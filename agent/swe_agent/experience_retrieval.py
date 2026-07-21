from __future__ import annotations

import json
import logging
import math
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any

from agent_rl.run_utils import (
    extract_last_json_object,
    freeform_thought_model_kwargs,
    route_completion_message,
)

from swe_agent.models.litellm_model import LitellmModel
from swe_agent.models.utils.retry import retry


logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*|\d+")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
PATH_RE = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")
IDENT_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning)|"
    r"[A-Za-z_][A-Za-z0-9_]{2,}(?=\s*\())\b"
)
QUOTED_RE = re.compile(r"[`'\"]([A-Za-z_][A-Za-z0-9_.:-]{2,})[`'\"]")
RRF_K = 60.0
QUERY_FIELDS = (
    "problem",
    "prior",
    "continuations",
    "stage",
    "llm_contract",
    "llm_state",
    "symbols",
)
DOCUMENT_FIELDS = ("full", "routing", "lesson", "features", "keywords")
SUMMARY_SYSTEM_PROMPT = (
    "You extract compact, visible retrieval features for SWE rubric experiences. "
    "Reason freely and append the requested JSON object."
)


def _rrf_contribution(rank: int) -> float:
    # The frozen weights were fitted on a float32 RRF contribution matrix.
    return struct.unpack("f", struct.pack("f", 1.0 / (RRF_K + rank)))[0]


def _text_values(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(
            f"{key}: {_text_values(item)}"
            for key, item in value.items()
            if item not in (None, "", [], {})
        )
    if isinstance(value, list):
        return "\n".join(_text_values(item) for item in value)
    return str(value)


def _dedupe(values: list[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return output


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN_RE.findall(text or ""):
        pieces = [raw]
        pieces.extend(
            part for part in re.split(r"[./:_-]+", raw) if part and part != raw
        )
        for piece in pieces:
            tokens.extend(
                part.casefold()
                for part in CAMEL_RE.split(piece)
                if len(part) > 1
            )
    return tokens


class BM25:
    def __init__(
        self,
        documents: list[list[str]],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.lengths = [len(document) for document in documents]
        self.term_frequencies = [Counter(document) for document in documents]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        frequencies: Counter[str] = Counter()
        for document in self.term_frequencies:
            frequencies.update(document.keys())
        total = len(documents)
        self.idf = {
            term: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in frequencies.items()
        }
        self.unknown_idf = math.log(1.0 + (total + 0.5) / 0.5)

    def score(self, query: list[str]) -> list[float]:
        query_terms = set(query)
        scores = []
        for length, frequencies in zip(self.lengths, self.term_frequencies):
            normalization = self.k1 * (
                1.0 - self.b + self.b * length / max(self.average_length, 1.0)
            )
            score = 0.0
            terms = query_terms if len(query_terms) <= len(frequencies) else frequencies
            for term in terms:
                if term not in query_terms:
                    continue
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += (
                        self.idf.get(term, self.unknown_idf)
                        * frequency
                        * (self.k1 + 1.0)
                        / (frequency + normalization)
                    )
            scores.append(score)
        return scores


def _normalized_strings(value: Any, limit: int = 8) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return _dedupe([str(item or "") for item in value])[:limit]


def parse_query_summary(parsed: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}
    output = {
        "task_contract": _normalized_strings(parsed.get("task_contract")),
        "implementation_boundaries": _normalized_strings(
            parsed.get("implementation_boundaries")
        ),
        "failure_modes": _normalized_strings(parsed.get("failure_modes")),
        "trajectory_stage": str(parsed.get("trajectory_stage") or "").strip(),
        "visible_branch_discriminators": _normalized_strings(
            parsed.get("visible_branch_discriminators")
        ),
        "abstention_risks": _normalized_strings(parsed.get("abstention_risks")),
        "retrieval_queries": _normalized_strings(parsed.get("retrieval_queries"), 10),
    }
    if not output["retrieval_queries"]:
        output["retrieval_queries"] = _dedupe(
            [
                *output["task_contract"],
                *output["implementation_boundaries"],
                *output["failure_modes"],
                *output["visible_branch_discriminators"],
            ]
        )[:10]
    return output


def _public_document(record: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Title: {record.get('title') or ''}",
            f"Description: {record.get('description') or ''}",
            f"Context: {record.get('context') or ''}",
            f"Experience: {record.get('experience') or ''}",
            "Metadata: " + _text_values(record.get("metadata") or {}),
        ]
    )


def _routing_document(record: dict[str, Any]) -> str:
    references = (record.get("metadata") or {}).get("reference_golden_rubrics") or []
    return "\n".join(
        [
            f"Title: {record.get('title') or ''}",
            f"Description: {record.get('description') or ''}",
            f"Activation and abstention: {record.get('context') or ''}",
            "Reference rubric titles and contracts: "
            + "\n".join(
                " | ".join(
                    [
                        str(item.get("title") or ""),
                        str(item.get("description") or ""),
                        _text_values(item.get("metadata") or {}),
                    ]
                )
                for item in references
                if isinstance(item, dict)
            ),
        ]
    )


def _lesson_document(record: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(record.get("experience") or ""),
            _text_values(
                (record.get("metadata") or {}).get("reference_golden_rubrics")
                or []
            ),
        ]
    )


def _feature_document(features: dict[str, list[str]]) -> str:
    return "\n".join(
        f"{field.replace('_', ' ').title()}: " + " | ".join(values)
        for field, values in features.items()
        if values
    )


def _feature_query(parts: dict[str, str]) -> str:
    values = []
    for pattern in (PATH_RE, IDENT_RE, QUOTED_RE):
        values.extend(pattern.findall(parts["all"]))
    return " ".join(_dedupe(values))


def _stage_query(context: dict[str, Any]) -> str:
    continuations = context.get("continuations") or []
    summaries = [item.get("summary") or {} for item in continuations]
    patched = sum(
        bool(summary.get("changed_files"))
        or int(summary.get("current_patch_chars") or 0) > 0
        for summary in summaries
    )
    unpatched = len(summaries) - patched
    changed_files = _dedupe(
        [
            str(path)
            for summary in summaries
            for path in (summary.get("changed_files") or [])
        ]
    )
    trajectory_text = _text_values(continuations).casefold()
    evidence = []
    for name, needles in (
        ("reproduction", ("reproduc", "trigger the", "steps to reproduce")),
        ("source inspection", ("grep ", "sed -n", "cat ", "inspect")),
        ("implementation", ("apply_patch", "git diff", "fix applied", "modified")),
        ("test execution", ("pytest", "test suite", "unittest", "tests pass")),
        ("environment blockage", ("modulenotfounderror", "dependency", "build failed")),
        ("submission artifact", ("submit", "patch.txt", "final patch")),
    ):
        if any(needle in trajectory_text for needle in needles):
            evidence.append(name)
    if patched == 0:
        population = "all continuations are pre-patch exploration"
    elif unpatched == 0:
        population = "all continuations have source patches"
    else:
        population = "mixed patched and unpatched continuation population"
    return "\n".join(
        [
            f"Trajectory stage portfolio: {population}.",
            f"Patched branches: {patched}; unpatched branches: {unpatched}.",
            "Visible evidence kinds: " + ", ".join(evidence or ["unclear"]),
            "Changed files across branches: " + ", ".join(changed_files),
            (
                "Retrieve experiences whose activation or abstention boundary matches this "
                "portfolio, preserves equivalent implementations, and supplies a criterion "
                "that can discriminate the visible branches now."
            ),
        ]
    )


def _summary_prompt(context_markdown: str, stage: str) -> str:
    return f"""Extract retrieval features from this visible SWE judging state. This is retrieval, not trajectory scoring: do not predict which branch passes and do not use hidden future outcomes.

Separate the literal task contract from current-stage branch evidence. Name concrete APIs, files, error modes, lifecycle or compatibility boundaries when visible. The retrieval queries should help find reusable judging experiences with matching activation and abstention boundaries; avoid generic words such as correctness, testing, implementation, or quality unless paired with the concrete contract.

Think freely, then end with one JSON object:
{{"task_contract":["..."],"implementation_boundaries":["..."],"failure_modes":["..."],"trajectory_stage":"...","visible_branch_discriminators":["..."],"abstention_risks":["..."],"retrieval_queries":["..."]}}

Use 2-8 concise items per relevant list and at most 10 retrieval queries.

## Compact visible state
{context_markdown}

## Deterministic portfolio features
{stage}"""


class WeightedKeywordExperienceRetriever:
    def __init__(self, checkpoint: Path, *, scope: str) -> None:
        self.checkpoint = Path(checkpoint)
        self.scope = scope
        config = json.loads(
            (self.checkpoint / "retrieval/recall_config.json").read_text(
                encoding="utf-8"
            )
        )
        if config.get("rerank") is not False:
            raise ValueError("Frozen weighted retrieval must have rerank=false")
        self.candidate_limit = int(config["candidate_limit"])
        self.query_weights = {
            field: float(config["query_weights"][field]) for field in QUERY_FIELDS
        }
        configured_document_weights = config["document_weights"]
        unknown_document_fields = set(configured_document_weights) - set(DOCUMENT_FIELDS)
        if unknown_document_fields:
            raise ValueError(
                f"Unknown frozen retrieval document fields: {sorted(unknown_document_fields)}"
            )
        self.document_fields = tuple(
            field for field in DOCUMENT_FIELDS if field in configured_document_weights
        )
        if not self.document_fields:
            raise ValueError("Frozen retrieval must configure at least one document field")
        self.document_weights = {
            field: float(configured_document_weights[field])
            for field in self.document_fields
        }
        self.summary_config = config.get("query_summary") or {}
        bank = json.loads(
            (self.checkpoint / "bank" / scope / "experience_bank.json").read_text(
                encoding="utf-8"
            )
        )
        keyword_bank = json.loads(
            (self.checkpoint / "retrieval/keyword_bank.json").read_text(
                encoding="utf-8"
            )
        )["bank"][scope]
        card_features = json.loads(
            (self.checkpoint / "retrieval/card_features.json").read_text(
                encoding="utf-8"
            )
        )["features"][scope]
        self.records = bank["experiences"]
        self.ids = [record["experience_id"] for record in self.records]
        if set(self.ids) != set(keyword_bank) or (
            "features" in self.document_fields and set(self.ids) != set(card_features)
        ):
            raise ValueError(f"Frozen retrieval coverage mismatch for {scope}")
        self.by_id = {record["experience_id"]: record for record in self.records}
        document_builders = {
            "full": lambda: [_public_document(record) for record in self.records],
            "routing": lambda: [_routing_document(record) for record in self.records],
            "lesson": lambda: [_lesson_document(record) for record in self.records],
            "features": lambda: [
                _feature_document(card_features[experience_id])
                for experience_id in self.ids
            ],
            "keywords": lambda: [
                "\n".join(keyword_bank[experience_id].get("keywords") or [])
                for experience_id in self.ids
            ],
        }
        self.bm25 = {
            field: BM25([tokenize(text) for text in document_builders[field]()])
            for field in self.document_fields
        }

    async def summarize(
        self,
        *,
        context_markdown: str,
        stage: str,
        model_name: str,
        top_p: float,
        model_kwargs: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": _summary_prompt(context_markdown, stage)},
        ]
        for _ in range(2):
            async for attempt in retry(
                logger=logger,
                abort_exceptions=LitellmModel.abort_exceptions,
                model_name=model_name,
                async_retry=True,
            ):
                with attempt:
                    assistant = await route_completion_message(
                        route_name="rubric_generation",
                        model_name=model_name,
                        messages=messages,
                        temperature=float(self.summary_config.get("temperature", 0.01)),
                        top_p=top_p,
                        max_tokens=int(self.summary_config.get("max_tokens", 4096)),
                        model_kwargs=freeform_thought_model_kwargs(model_kwargs),
                    )
            messages.append(assistant)
            response = assistant.get("content_no_thinking") or assistant.get("content") or ""
            parsed = extract_last_json_object(response)
            if parsed is None and assistant.get("content") != response:
                parsed = extract_last_json_object(assistant.get("content") or "")
            summary = parse_query_summary(parsed)
            if summary.get("retrieval_queries"):
                return summary, messages
            messages.append(
                {
                    "role": "user",
                    "content": "Append one valid final JSON object using the requested retrieval-feature schema.",
                }
            )
        return {}, messages

    def rank(
        self,
        *,
        context: dict[str, Any],
        summary: dict[str, Any],
        instance_id: str,
    ) -> list[str]:
        problem = _text_values(context.get("question"))
        prior = "\n".join(
            [
                _text_values(context.get("previous_state")),
                _text_values(context.get("parent_trajectory")),
            ]
        )
        continuations = _text_values(context.get("continuations"))
        parts = {
            "problem": problem,
            "prior": prior,
            "continuations": continuations,
            "all": "\n".join((problem, prior, continuations)),
        }
        query_fields = {
            "problem": problem,
            "prior": prior,
            "continuations": continuations,
            "stage": _stage_query(context),
            "llm_contract": _text_values(
                [
                    summary.get("task_contract") or [],
                    summary.get("implementation_boundaries") or [],
                    summary.get("failure_modes") or [],
                    summary.get("retrieval_queries") or [],
                ]
            ),
            "llm_state": _text_values(
                [
                    summary.get("trajectory_stage") or "",
                    summary.get("visible_branch_discriminators") or [],
                    summary.get("abstention_risks") or [],
                ]
            ),
            "symbols": _feature_query(parts),
        }
        scores: Counter[str] = Counter()
        for query_field in QUERY_FIELDS:
            query_weight = self.query_weights[query_field]
            if query_weight <= 0 or not query_fields[query_field].strip():
                continue
            query = tokenize(query_fields[query_field])
            for document_field in self.document_fields:
                document_weight = self.document_weights[document_field]
                if document_weight <= 0:
                    continue
                bm25_scores = self.bm25[document_field].score(query)
                ranking = sorted(
                    range(len(self.ids)),
                    key=lambda index: (-bm25_scores[index], self.ids[index]),
                )
                for rank, index in enumerate(ranking, start=1):
                    scores[self.ids[index]] += (
                        query_weight
                        * document_weight
                        * _rrf_contribution(rank)
                    )
        return sorted(
            self.ids,
            key=lambda experience_id: (-scores[experience_id], experience_id),
        )[: self.candidate_limit]

    async def retrieve(
        self,
        *,
        context: dict[str, Any],
        context_markdown: str,
        instance_id: str,
        model_name: str,
        top_p: float,
        model_kwargs: dict[str, Any] | None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        stage = _stage_query(context)
        summary, messages = await self.summarize(
            context_markdown=context_markdown,
            stage=stage,
            model_name=model_name,
            top_p=top_p,
            model_kwargs=model_kwargs,
        )
        return self.rank(
            context=context,
            summary=summary,
            instance_id=instance_id,
        ), messages
