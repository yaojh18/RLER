from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from swe_agent.experience_retrieval import (
    WeightedKeywordExperienceRetriever,
    _text_values,
)
from swe_agent.prompt import (
    _seed_rubrics,
)

OBSERVATION_TRUNCATION_MARKER = "\n[... Observation truncated due to length ...]\n"
MAX_TERMINAL_PATCH_SECTION_CHARS = 4096
RETRIEVAL_SUMMARY_CONTEXT_CHARS = 48_000


def _truncate_middle(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(OBSERVATION_TRUNCATION_MARKER):
        return text[:limit]
    remaining = limit - len(OBSERVATION_TRUNCATION_MARKER)
    head = max(1, remaining // 2)
    tail = max(1, remaining - head)
    if head + tail >= len(text):
        return text
    return text[:head] + OBSERVATION_TRUNCATION_MARKER + text[-tail:]


def render_compact_markdown(value: Any) -> str:
    """Render prompt context without JSON punctuation or string escaping."""

    def scalar_text(item: Any) -> str:
        if item is None:
            return "None"
        if isinstance(item, bool):
            return "true" if item else "false"
        return str(item)

    def render(item: Any, indent: int) -> list[str]:
        prefix = " " * indent
        if isinstance(item, dict):
            if not item:
                return [prefix + "None"]
            rows: list[str] = []
            for key, child in item.items():
                if isinstance(child, (dict, list)):
                    rows.append(f"{prefix}- **{key}**:")
                    rows.extend(render(child, indent + 2))
                    continue
                text = scalar_text(child)
                if "\n" not in text:
                    rows.append(f"{prefix}- **{key}**: {text}")
                    continue
                rows.append(f"{prefix}- **{key}**:")
                rows.extend(f"{prefix}  {line}" for line in text.splitlines())
            return rows
        if isinstance(item, list):
            if not item:
                return [prefix + "None"]
            rows = []
            for child in item:
                if isinstance(child, (dict, list)):
                    rows.append(prefix + "-")
                    rows.extend(render(child, indent + 2))
                    continue
                text = scalar_text(child)
                lines = text.splitlines() or [""]
                rows.append(f"{prefix}- {lines[0]}")
                rows.extend(f"{prefix}  {line}" for line in lines[1:])
            return rows
        return [prefix + scalar_text(item)]

    return "\n".join(render(value, 0))


def _retrieval_trajectory_excerpt(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    cards = value.get("step_cards") or []
    return {
        "segment_step_range": value.get("segment_step_range"),
        "visible_step_card_count": value.get("visible_step_card_count"),
        "recent_steps": [
            {
                "step_index": card.get("step_index"),
                "assistant_message": _truncate_middle(
                    str(card.get("assistant_message") or ""), 900
                ),
                "observation": _truncate_middle(
                    str(card.get("observation") or ""), 900
                ),
            }
            for card in cards[-2:]
        ],
    }


def _retrieval_summary_context(
    context: dict[str, Any], *, instance_id: str, round_index: int
) -> dict[str, Any]:
    continuations = [
        {
            "node_id": item.get("node_id"),
            "parent_index": item.get("parent_index"),
            "workspace_summary": item.get("summary"),
            "recent_trajectory": _retrieval_trajectory_excerpt(
                item.get("raw_continuation", item.get("trajectory_continuation"))
            ),
        }
        for item in context.get("continuations") or []
    ]
    state = {
        "repository": instance_id.rsplit("-", 1)[0],
        "round": round_index,
        "problem": _truncate_middle(_text_values(context.get("question")), 10_000),
        "previous_persistent_state": _truncate_middle(
            _text_values(context.get("previous_state")), 8_000
        ),
        "parent_trajectory": _retrieval_trajectory_excerpt(
            context.get("parent_trajectory")
        ),
        "continuations": continuations,
    }
    if len(render_compact_markdown(state)) <= RETRIEVAL_SUMMARY_CONTEXT_CHARS:
        return state
    return {
        "repository": state["repository"],
        "round": round_index,
        "problem": state["problem"],
        "previous_persistent_state": state["previous_persistent_state"],
        "continuations": [
            {
                "node_id": item["node_id"],
                "parent_index": item["parent_index"],
                "workspace_summary": item["workspace_summary"],
            }
            for item in continuations
        ],
    }


@dataclass
class RubricRecord:
    rubric_id: str
    title: str
    direction: Literal["positive", "negative"]
    description: str
    scale: dict[str, str]
    weight: float
    source_round: int
    reward: float | None = None
    metadata: dict[str, str] | None = None


@dataclass
class RubricGenerationSample:
    sample_index: int
    rubric_list_id: str
    generated: list[RubricRecord]
    messages: list[dict[str, Any]]
    format_errors: list[dict[str, Any]] | None = None
    terminal_error: str | None = None


@dataclass
class RubricExperience:
    experience_id: str
    title: str
    description: str
    metadata: dict[str, Any]
    context: str
    experience: str


@dataclass
class RubricBankGenerationContext:
    existing_rubrics: list[RubricRecord]
    extra_prompt_sections: list[str]
    retrieved: list[RubricExperience]
    retrieve_messages: list[dict[str, Any]]


@dataclass
class RubricBankRoundUpdate:
    rubrics: list[RubricRecord]
    active_before: list[RubricRecord]
    active_after: list[RubricRecord]
    inactive_after: list[RubricRecord]


def build_terminal_update_evidence(
    *,
    parent_patch: str = "",
    parent_evaluation: dict[str, Any] | None = None,
    continuations: list[dict[str, Any]],
) -> dict[str, str]:
    items = [
        {
            "title": "Parent",
            "patch": parent_patch or "",
            "evaluation": parent_evaluation,
        }
    ]
    for index, continuation in enumerate(continuations, start=1):
        items.append(
            {
                "title": f"Continuation {index}",
                "patch": str(continuation.get("patch") or ""),
                "evaluation": continuation.get("evaluation"),
            }
        )
    test_rows: list[tuple[str, set[str], bool]] = []
    evaluated_sets: list[set[str]] = []
    for item in items:
        evaluation = item.get("evaluation")
        has_evaluation = isinstance(evaluation, dict)
        tests = (
            {
                str(test)
                for test in evaluation.get("passed_tests", [])
                if str(test)
            }
            if has_evaluation
            else set()
        )
        if has_evaluation:
            evaluated_sets.append(tests)
        test_rows.append((str(item.get("title", "")), tests, has_evaluation))
    common_tests = (
        set.intersection(*evaluated_sets) if evaluated_sets else set()
    )
    passed_tests_by_title = {
        title: sorted(tests - common_tests) if has_evaluation else []
        for title, tests, has_evaluation in test_rows
    }
    terminal_patch_rows: list[str] = []
    passed_test_rows: list[str] = []
    for item in items:
        patch = _truncate_middle(str(item["patch"] or ""), MAX_TERMINAL_PATCH_SECTION_CHARS)
        diff_tests = passed_tests_by_title.get(item["title"], [])
        terminal_patch_rows.extend([f"## {item['title']}:", patch.strip() if patch.strip() else "<empty>"])
        passed_test_rows.extend(
            [
                f"## {item['title']}:",
                "\n".join(diff_tests) if diff_tests else "<none>",
            ]
        )
    return {
        "terminal_patch": "\n\n".join(terminal_patch_rows),
        "passed_tests": "\n\n".join(passed_test_rows),
    }


def _convert_rubric_item(
    task: str,
    item: dict[str, Any],
    round_index: int,
) -> RubricRecord | None:
    if not isinstance(item, dict):
        return None
    normalized_keys = {
        re.sub(r"[^a-z0-9]+", "_", str(key).strip().casefold()).strip("_"): value
        for key, value in item.items()
    }
    aliases = {
        "direction": "polarity",
        "type": "polarity",
        "name": "title",
        "criterion": "description",
        "importance": "weight",
    }
    for source, target in aliases.items():
        if target not in normalized_keys and source in normalized_keys:
            normalized_keys[target] = normalized_keys[source]
    item = normalized_keys
    direction = item.get("polarity", None)
    if isinstance(direction, str):
        direction = direction.strip().lower()
    if direction not in {"positive", "negative"}:
        return None
    title = item.get("title", None)
    if title is None or not isinstance(title, str) or not title.strip():
        return None
    title = title.strip()
    description = item.get("description", None)
    if description is None or not isinstance(description, str) or not description.strip():
        return None
    description = description.strip()
    raw_scale = item.get("scale") or {}
    if isinstance(raw_scale, list) and len(raw_scale) == 5:
        raw_scale = {str(index): text for index, text in enumerate(raw_scale, start=1)}
    if not isinstance(raw_scale, dict):
        return None
    scale = {}
    for score, text in raw_scale.items():
        score_match = re.fullmatch(r"(?:score[_\s-]*)?([1-5])", str(score).strip().casefold())
        if score_match is not None:
            scale[score_match.group(1)] = text
    if set(scale) != {"1", "2", "3", "4", "5"}:
        return None
    scale_text = {isinstance(text, str) for text in scale.values()}
    if not scale_text or not all(scale_text):
        return None
    metadata_raw = item.get("metadata") or {}
    if not isinstance(metadata_raw, dict):
        return None
    metadata = {str(key): str(value) for key, value in metadata_raw.items()}
    if "weight" not in item:
        return None
    try:
        weight = float(item["weight"])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(weight) or weight <= 0.0:
        return None
    rubric_id = hashlib.md5(
        json.dumps(
            {
                "task": task,
                "direction": direction,
                "title": title,
                "description": description,
                "metadata": metadata,
                "scale": scale,
                "weight": weight,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return RubricRecord(
        rubric_id=rubric_id,
        title=title,
        direction=direction,
        description=description,
        scale=scale,
        weight=weight,
        source_round=round_index,
        metadata=metadata,
    )


class ScoreRubricBank:
    def __init__(self, *, max_active_rubrics: int, scope: str = "siblings") -> None:
        self.max_active_rubrics = max_active_rubrics
        self.scope = scope
        self.active_bank: list[RubricRecord] = []
        self.inactive_bank: list[RubricRecord] = []

    def initialize(self, task: str) -> None:
        self.active_bank = [
            rubric
            for seed in _seed_rubrics(scope=self.scope)
            if (rubric := _convert_rubric_item(task, seed, 0)) is not None
        ]
        self.inactive_bank = []

    def set_state(self, *, active_bank: list[RubricRecord], inactive_bank: list[RubricRecord]) -> None:
        self.active_bank = copy.deepcopy(active_bank)
        self.inactive_bank = copy.deepcopy(inactive_bank)

    def build_generation_context(self, **_: Any) -> RubricBankGenerationContext:
        sections = []
        if self.active_bank:
            sections.append(
                "## Existing Rubrics:\n"
                + render_compact_markdown(
                    [
                        {
                            "polarity": rubric.direction,
                            "weight": rubric.weight,
                            "title": rubric.title,
                            "description": rubric.description,
                            "scale": rubric.scale,
                            "metadata": rubric.metadata,
                        }
                        for rubric in self.active_bank
                    ]
                )
            )
        return RubricBankGenerationContext(
            existing_rubrics=copy.deepcopy(self.active_bank),
            extra_prompt_sections=sections,
            retrieved=[],
            retrieve_messages=[],
        )

    def update_from_model(self, *, generated: list[RubricRecord]) -> RubricBankRoundUpdate:
        active_before = copy.deepcopy(self.active_bank)
        selected_by_title: dict[str, RubricRecord] = {}
        for rubric in generated:
            selected_by_title[rubric.title.strip().casefold()] = copy.deepcopy(rubric)
        active_after = list(selected_by_title.values())[: self.max_active_rubrics]
        active_ids = {rubric.rubric_id for rubric in active_after}
        inactive_after: list[RubricRecord] = []
        inactive_ids: set[str] = set()
        for rubric in active_before + self.inactive_bank:
            if rubric.rubric_id in active_ids or rubric.rubric_id in inactive_ids:
                continue
            inactive_after.append(copy.deepcopy(rubric))
            inactive_ids.add(rubric.rubric_id)
        return RubricBankRoundUpdate(
            rubrics=active_after,
            active_before=active_before,
            active_after=active_after,
            inactive_after=inactive_after,
        )


class ExperienceRubricBank:
    def __init__(
        self,
        *,
        bank_path: Path,
        scope: str = "siblings",
    ) -> None:
        checkpoint = Path(bank_path)
        if not checkpoint.is_dir():
            raise ValueError(
                "Experience retrieval requires a frozen checkpoint directory"
            )
        self.retriever = WeightedKeywordExperienceRetriever(
            checkpoint,
            scope=scope,
        )
        self.bank_path = checkpoint / "bank" / scope / "experience_bank.json"
        self.scope = scope
        self.experiences = self.load()

    def load(self) -> dict[str, RubricExperience]:
        payload = json.loads(self.bank_path.read_text(encoding="utf-8"))
        items = payload.get("experiences") if isinstance(payload, dict) else None
        if not isinstance(items, list) or not items:
            raise ValueError("Frozen experience bank must contain experiences")
        experiences: dict[str, RubricExperience] = {}
        for item in items:
            required = (
                "experience_id",
                "title",
                "description",
                "context",
                "experience",
            )
            if not isinstance(item, dict) or any(
                not str(item.get(key) or "").strip() for key in required
            ):
                raise ValueError("Frozen experience is missing a required text field")
            if not isinstance(item.get("metadata"), dict):
                raise ValueError("Frozen experience metadata must be an object")
            experience = RubricExperience(
                experience_id=str(item["experience_id"]).strip(),
                title=str(item["title"]).strip(),
                description=str(item["description"]).strip(),
                context=str(item["context"]).strip(),
                experience=str(item["experience"]).strip(),
                metadata=copy.deepcopy(item["metadata"]),
            )
            experiences[experience.experience_id] = experience
        return experiences

    def to_list(self) -> list[dict[str, Any]]:
        return [asdict(experience) for experience in self.experiences.values()]

    async def build_generation_context(
        self,
        *,
        question: dict[str, Any],
        previous_state: dict[str, Any],
        latest_shared_segment: dict[str, Any] | None,
        continuations: list[dict[str, Any]],
        model_name: str,
        top_p: float,
        model_kwargs: dict[str, Any] | None = None,
        retrieval_format_correction_rounds: int = 2,
        instance_id: str = "",
        round_index: int = 0,
    ) -> RubricBankGenerationContext:
        context = {
            "question": question,
            "previous_state": previous_state,
            "parent_trajectory": latest_shared_segment,
            "continuations": continuations,
        }
        experience_ids, retrieve_messages = await self.retriever.retrieve(
            context=context,
            context_markdown=render_compact_markdown(
                _retrieval_summary_context(
                    context,
                    instance_id=instance_id,
                    round_index=round_index,
                )
            ),
            instance_id=instance_id,
            model_name=model_name,
            top_p=top_p,
            model_kwargs=model_kwargs,
            max_format_correction_rounds=(
                retrieval_format_correction_rounds
            ),
        )
        by_id = {
            experience.experience_id: experience
            for experience in self.experiences.values()
        }
        retrieved = [
            by_id[experience_id]
            for experience_id in experience_ids
            if experience_id in by_id
        ]
        sections = []
        if retrieved:
            sections.append(
                "## Retrieved Rubric Experiences:\n"
                + render_compact_markdown(
                    [
                        {
                            "title": item.title,
                            "description": item.description,
                            "context": item.context,
                            "experience": item.experience,
                            "reference_golden_rubrics": item.metadata.get(
                                "reference_golden_rubrics"
                            ),
                        }
                        for item in retrieved
                    ]
                )
            )
        return RubricBankGenerationContext(
            existing_rubrics=[],
            extra_prompt_sections=sections,
            retrieved=copy.deepcopy(retrieved),
            retrieve_messages=copy.deepcopy(retrieve_messages),
        )
