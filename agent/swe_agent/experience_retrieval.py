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
RRF_K = 60.0
QUERY_WEIGHTS = {
    "prior": 0.6,
    "stage": 0.69,
    "llm_contract": 1.48,
    "llm_state": 4.01,
}
DOCUMENT_WEIGHTS = {"full": 0.19, "keywords": 1.39}
CANDIDATE_LIMIT = 6
SUMMARY_TEMPERATURE = 0.01
SUMMARY_MAX_TOKENS = 4096
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
                        self.idf[term]
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
    def __init__(
        self,
        checkpoint: Path,
        *,
        scope: str,
    ) -> None:
        checkpoint = Path(checkpoint)
        bank = json.loads(
            (checkpoint / "bank" / scope / "experience_bank.json").read_text(
                encoding="utf-8"
            )
        )
        keyword_bank = json.loads(
            (checkpoint / "retrieval/keyword_bank.json").read_text(
                encoding="utf-8"
            )
        )["bank"][scope]
        self.records = bank["experiences"]
        self.ids = [record["experience_id"] for record in self.records]
        self.instance_labels: dict[str, str | None] = {}
        for record in self.records:
            label = record.get("instance_label")
            if label is not None and not isinstance(label, str):
                raise ValueError(
                    f"Non-scalar instance label for {scope}:{record['experience_id']}"
                )
            self.instance_labels[record["experience_id"]] = label or None
        if set(self.ids) != set(keyword_bank):
            raise ValueError(f"Frozen retrieval coverage mismatch for {scope}")
        if any(
            set(keyword_bank[experience_id])
            != {"generated_keywords", "selected_keywords"}
            for experience_id in self.ids
        ):
            raise ValueError(f"Invalid frozen keyword schema for {scope}")
        documents = {
            "full": [_public_document(record) for record in self.records],
            "keywords": [
                "\n".join(
                    keyword_bank[experience_id].get("selected_keywords") or []
                )
                for experience_id in self.ids
            ],
        }
        self.bm25 = {
            field: BM25([tokenize(text) for text in documents[field]])
            for field in DOCUMENT_WEIGHTS
        }

    async def summarize(
        self,
        *,
        context_markdown: str,
        stage: str,
        model_name: str,
        top_p: float,
        model_kwargs: dict[str, Any] | None,
        max_format_correction_rounds: int = 2,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": _summary_prompt(context_markdown, stage)},
        ]
        for _ in range(max(1, int(max_format_correction_rounds))):
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
                        temperature=SUMMARY_TEMPERATURE,
                        top_p=top_p,
                        max_tokens=SUMMARY_MAX_TOKENS,
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
                    "content": (
                        "The previous response was not valid parseable JSON. "
                        "Do not repeat it. Escape every double quote inside a "
                        "string value (or omit the quoted code example), then "
                        "append one valid final JSON object using the requested "
                        "retrieval-feature schema."
                    ),
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
        prior = "\n".join(
            [
                _text_values(context.get("previous_state")),
                _text_values(context.get("parent_trajectory")),
            ]
        )
        query_fields = {
            "prior": prior,
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
        }
        scores: Counter[str] = Counter()
        for query_field, query_weight in QUERY_WEIGHTS.items():
            if not query_fields[query_field].strip():
                continue
            query = tokenize(query_fields[query_field])
            for document_field, document_weight in DOCUMENT_WEIGHTS.items():
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
        eligible_ids = [
            experience_id
            for experience_id in self.ids
            if not instance_id or self.instance_labels[experience_id] != instance_id
        ]
        return sorted(
            eligible_ids,
            key=lambda experience_id: (-scores[experience_id], experience_id),
        )[:CANDIDATE_LIMIT]

    async def retrieve(
        self,
        *,
        context: dict[str, Any],
        context_markdown: str,
        instance_id: str,
        model_name: str,
        top_p: float,
        model_kwargs: dict[str, Any] | None,
        max_format_correction_rounds: int = 2,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        stage = _stage_query(context)
        summary, messages = await self.summarize(
            context_markdown=context_markdown,
            stage=stage,
            model_name=model_name,
            top_p=top_p,
            model_kwargs=model_kwargs,
            max_format_correction_rounds=max_format_correction_rounds,
        )
        return self.rank(
            context=context,
            summary=summary,
            instance_id=instance_id,
        ), messages
