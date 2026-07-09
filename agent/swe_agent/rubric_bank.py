from __future__ import annotations

import copy
import difflib
import hashlib
import json
import math
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from agent_rl.run_utils import extract_last_json_object, freeform_thought_model_kwargs, route_completion_message

from swe_agent.models.litellm_model import LitellmModel
from swe_agent.models.utils.retry import retry
from swe_agent.prompt import (
    RUBRIC_EXPERIENCE_RETRIEVAL_PROMPT,
    RUBRIC_EXPERIENCE_UPDATE_PROMPT,
    _seed_experiences,
    _seed_rubrics,
)


logger = logging.getLogger(__name__)
OBSERVATION_TRUNCATION_MARKER = "\n[... Observation truncated due to length ...]\n"
MAX_TERMINAL_PATCH_SECTION_CHARS = 4096


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


def _experience_id(payload: dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()[:12]


def _rounded_score_list(scores: dict[str, Any], ordered_ids: list[str]) -> list[float]:
    return [round(float(scores.get(node_id, 0.0)), 3) for node_id in ordered_ids]


def _public_experience(experience: RubricExperience | dict[str, Any]) -> dict[str, Any]:
    payload = asdict(experience) if isinstance(experience, RubricExperience) else copy.deepcopy(experience)
    payload.pop("experience_id", None)
    return payload


def gold_patch_skeleton(gold_patch: str, max_chars: int = 4096) -> str:
    rows: list[str] = []
    current_file = ""
    for line in (gold_patch or "").splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            current_file = parts[2][2:] if len(parts) >= 3 and parts[2].startswith("a/") else (parts[2] if len(parts) >= 3 else "")
            rows.append(line)
        elif line.startswith("@@"):
            rows.append(f"{current_file}: {line}")
        elif line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            stripped = line[1:].strip()
            if stripped:
                signature = re.sub(r"\s+", " ", stripped)
                rows.append(f"{line[0]} {signature[:160]}")
    text = "\n".join(rows)
    if len(text) <= max_chars:
        return text
    return _truncate_middle(text, max_chars)


def evaluation_tests_not_shared(items: list[dict[str, Any]], *, test_key: str) -> dict[str, list[str]]:
    rows: list[tuple[str, set[str], bool]] = []
    evaluated_sets: list[set[str]] = []
    for item in items:
        label = str(item.get("title", ""))
        evaluation = item.get("evaluation")
        has_evaluation = isinstance(evaluation, dict)
        tests = {str(test) for test in evaluation.get(test_key, []) if str(test)} if has_evaluation else set()
        if has_evaluation:
            evaluated_sets.append(tests)
        rows.append((label, tests, has_evaluation))
    common_tests = set.intersection(*evaluated_sets) if evaluated_sets else set()
    return {label: sorted(tests - common_tests) if has_evaluation else [] for label, tests, has_evaluation in rows}


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
    passed_tests_by_title = evaluation_tests_not_shared(items, test_key="passed_tests")
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


def _convert_experience(payload: dict[str, Any], experience_id: str | None = None) -> RubricExperience | None:
    required_keys = {"title", "description", "metadata", "context", "experience"}
    if required_keys - set(payload):
        return None
    title = payload.get("title")
    description = payload.get("description")
    experience = payload.get("experience")
    context = payload.get("context")
    if not all(value is not None and str(value).strip() for value in (title, description, experience, context)):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    legacy_metadata_types = {
        "generated_rubrics": list,
        "gt_skeleton": str,
        "generated_rubric_accuracy": dict,
        "gt_scores": list,
        "analysis": str,
        "reference_golden_rubrics": list,
    }
    compact_metadata_types = {"reference_golden_rubrics": list}
    if set(metadata) == set(legacy_metadata_types):
        required_metadata_types = legacy_metadata_types
    elif set(metadata) == set(compact_metadata_types):
        # Offline PRM analysis emits deployable experiences without retaining
        # privileged reward, score, or golden-patch diagnostics. Continue to
        # accept the legacy update artifacts while allowing this compact form.
        required_metadata_types = compact_metadata_types
    else:
        return None
    for key, expected_type in required_metadata_types.items():
        if not isinstance(metadata.get(key), expected_type):
            return None
    for rubric in metadata.get("generated_rubrics", []):
        if not isinstance(rubric, dict) or _convert_rubric_item("generated", rubric, 0) is None:
            return None
    for rubric in metadata["reference_golden_rubrics"]:
        if not isinstance(rubric, dict) or _convert_rubric_item("reference", rubric, 0) is None:
            return None
    body = {
        "title": str(title).strip(),
        "description": str(description).strip(),
        "context": str(context).strip(),
        "experience": str(experience).strip(),
        "metadata": copy.deepcopy(metadata),
    }
    return RubricExperience(
        experience_id=experience_id or _experience_id(body),
        **body,
    )


def _convert_action(
    payload: dict[str, Any],
    *,
    experiences: dict[str, RubricExperience],
    attempt_evidence: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    action = payload.get("action")
    if not isinstance(action, str):
        return None
    if action == "retrieve":
        if {"action", "titles"} - set(payload):
            return None
        titles = payload.get("titles")
        if not isinstance(titles, list):
            return None
        normalized_titles = []
        for title in titles:
            if title is None or not str(title).strip():
                return None
            normalized_title = str(title).strip()
            if normalized_title not in experiences:
                return None
            normalized_titles.append(normalized_title)
        return {"action": "retrieve", "titles": list(dict.fromkeys(normalized_titles))}
    if action == "delete":
        if {"action", "title"} - set(payload):
            return None
        title = payload.get("title")
        if title is None or not str(title).strip():
            return None
        title = str(title).strip()
        if title not in experiences:
            return None
        return {"action": "delete", "title": title}
    if action not in {"add", "update"}:
        return None
    expected_keys = {"action", "title", "description", "metadata", "context", "experience"}
    if action == "update":
        expected_keys.add("target_title")
    if expected_keys - set(payload):
        return None
    target_title = payload.get("target_title")
    experience_id = None
    if action == "update":
        if target_title is None or not str(target_title).strip():
            return None
        target_title = str(target_title).strip()
        if target_title not in experiences:
            return None
        experience_id = experiences[target_title].experience_id
    experience_payload = {
        key: copy.deepcopy(payload.get(key))
        for key in ("title", "description", "metadata", "context", "experience")
    }
    title = experience_payload.get("title")
    if title is None or not str(title).strip():
        return None
    title = str(title).strip()
    experience_payload["title"] = title
    if action == "add" and title in experiences:
        return None
    if action == "update" and title != target_title and title in experiences:
        return None
    metadata = experience_payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    analysis = metadata.get("analysis", "")
    reference_golden_rubrics = metadata.get("reference_golden_rubrics")
    if not isinstance(reference_golden_rubrics, list) or not reference_golden_rubrics:
        return None
    validated_reference_golden_rubrics = []
    for reference in reference_golden_rubrics:
        if not isinstance(reference, dict):
            return None
        rubric = _convert_rubric_item("reference", reference, 0)
        if rubric is None:
            return None
        validated_reference_golden_rubrics.append(
            {
                "polarity": rubric.direction,
                "weight": rubric.weight,
                "title": rubric.title,
                "description": rubric.description,
                "metadata": copy.deepcopy(rubric.metadata or {}),
                "scale": copy.deepcopy(rubric.scale),
            }
        )
    experience_payload["metadata"] = {
        "generated_rubrics": copy.deepcopy(attempt_evidence["generated_rubrics"]),
        "gt_skeleton": attempt_evidence["gt_skeleton"],
        "generated_rubric_accuracy": copy.deepcopy(attempt_evidence["generated_rubric_accuracy"]),
        "gt_scores": copy.deepcopy(attempt_evidence["gt_scores"]),
        "analysis": analysis,
        "reference_golden_rubrics": validated_reference_golden_rubrics,
    }
    experience = _convert_experience(experience_payload, experience_id)
    if experience is None:
        return None
    converted = {"action": action, "experience": experience}
    if action == "update":
        converted["target_title"] = target_title
    return converted


def _pairwise_accuracy(scores: dict[str, float], gt_scores: dict[str, float], node_ids: list[str]) -> float:
    correct = 0.0
    total = 0
    for left_index in range(len(node_ids)):
        for right_index in range(left_index + 1, len(node_ids)):
            left = node_ids[left_index]
            right = node_ids[right_index]
            if left not in scores or right not in scores or left not in gt_scores or right not in gt_scores:
                continue
            score_delta = float(scores[left]) - float(scores[right])
            gt_delta = float(gt_scores[left]) - float(gt_scores[right])
            total += 1
            if abs(gt_delta) <= 1e-9 or abs(score_delta) <= 1e-9:
                correct += 0.5
            elif score_delta * gt_delta > 0:
                correct += 1.0
    return correct / total if total else 0.5


def _has_reward_variance(scores: dict[str, float], node_ids: list[str]) -> bool:
    values = [float(scores[node_id]) for node_id in node_ids if node_id in scores]
    return len(values) >= 2 and max(values) - min(values) > 1e-9


def _rubric_accuracy_payload(
    *,
    generated: list[dict[str, Any]],
    score_by_rubric: dict[str, dict[str, float]],
    gt_scores: dict[str, float],
    ordered_node_ids: list[str],
    scope: str = "siblings",
) -> dict[str, dict[str, Any]]:
    accuracy: dict[str, dict[str, Any]] = {}
    for rubric in generated:
        rubric_id = rubric.get("rubric_id")
        title = rubric.get("title")
        raw_scores = score_by_rubric.get(rubric_id)
        if not raw_scores:
            continue
        direction = rubric.get("direction")
        aligned_scores = {
            node_id: (
                0.5 * (1.0 - float(raw_scores.get(node_id)))
                if scope == "pc" and direction == "negative"
                else (1.0 - float(raw_scores.get(node_id)) if direction == "negative" else float(raw_scores.get(node_id)))
            )
            for node_id in ordered_node_ids
            if node_id in raw_scores
        }
        accuracy[title] = {
            "overall_accuracy": round(_pairwise_accuracy(aligned_scores, gt_scores, ordered_node_ids), 3),
            "judging_diff_per_sample": [
                round(aligned_scores.get(node_id) - gt_scores.get(node_id), 3)
                for node_id in ordered_node_ids
            ],
        }
    return accuracy


def _seed_experience_records(scope: str = "siblings") -> dict[str, RubricExperience]:
    return {
        experience.title: experience
        for seed in _seed_experiences(scope=scope)
        if (experience := _convert_experience(seed)) is not None
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


def _seed_rubric_records(task: str, scope: str = "siblings") -> list[RubricRecord]:
    return [rubric for seed in _seed_rubrics(scope=scope) if (rubric := _convert_rubric_item(task, seed, 0)) is not None]


class ScoreRubricBank:
    def __init__(self, *, max_active_rubrics: int, scope: str = "siblings") -> None:
        self.max_active_rubrics = max_active_rubrics
        self.scope = scope
        self.active_bank: list[RubricRecord] = []
        self.inactive_bank: list[RubricRecord] = []

    def initialize(self, task: str) -> None:
        self.active_bank = _seed_rubric_records(task, scope=self.scope)
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

    def update_after_round(
        self,
        *,
        generated: list[RubricRecord],
        rewards: dict[str, float],
    ) -> RubricBankRoundUpdate:
        active_before = copy.deepcopy(self.active_bank)
        deduped_by_title: dict[str, RubricRecord] = {}
        kept_active_bank = [rubric for rubric in self.active_bank if rewards.get(rubric.rubric_id, 0.0) > 0.0]
        for rubric in kept_active_bank + generated:
            candidate = RubricRecord(**{**asdict(rubric), "reward": rewards.get(rubric.rubric_id, 0.0)})
            title_key = candidate.title.strip().casefold()
            existing = deduped_by_title.get(title_key)
            if existing is None or (candidate.reward) > (existing.reward) or (
                candidate.reward == existing.reward and candidate.source_round > existing.source_round
            ):
                deduped_by_title[title_key] = candidate
        ranked = list(deduped_by_title.values())
        ranked.sort(key=lambda rubric: rubric.reward, reverse=True)
        active_after = ranked[: self.max_active_rubrics]
        active_ids = {rubric.rubric_id for rubric in active_after}
        inactive_after = [rubric for rubric in ranked + self.inactive_bank if rubric.rubric_id not in active_ids]
        return RubricBankRoundUpdate(
            rubrics=active_after,
            active_before=active_before,
            active_after=active_after,
            inactive_after=inactive_after,
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


def _normalize_experience_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title)
    normalized = normalized.translate(str.maketrans({"‘": '"', "’": '"', "“": '"', "”": '"', "'": '"'}))
    return " ".join(normalized.split()).casefold()


def _fuzzy_experience_title(title: str) -> str:
    return " ".join(re.findall(r"\w+", _normalize_experience_title(title)))


def _match_experience_title(title: str, available_titles: list[str]) -> str | None:
    normalized = _normalize_experience_title(title)
    exact_matches = [candidate for candidate in available_titles if _normalize_experience_title(candidate) == normalized]
    if len(exact_matches) == 1:
        return exact_matches[0]
    query = _fuzzy_experience_title(title)
    containment_matches = [
        candidate
        for candidate in available_titles
        if query
        and (
            query in _fuzzy_experience_title(candidate)
            or _fuzzy_experience_title(candidate) in query
        )
    ]
    if len(containment_matches) == 1:
        return containment_matches[0]
    ranked = sorted(
        (
            difflib.SequenceMatcher(None, query, _fuzzy_experience_title(candidate)).ratio(),
            candidate,
        )
        for candidate in available_titles
    )
    if not ranked:
        return None
    best_score, best_title = ranked[-1]
    second_score = ranked[-2][0] if len(ranked) > 1 else 0.0
    if best_score >= 0.88 and best_score - second_score >= 0.15:
        return best_title
    return None


class ExperienceRubricBank:
    def __init__(
        self,
        *,
        bank_path: Path | None = None,
        retrieve_top_k: int = 4,
        retrieval_prompt: str = RUBRIC_EXPERIENCE_RETRIEVAL_PROMPT,
        update_prompt: str = RUBRIC_EXPERIENCE_UPDATE_PROMPT,
        scope: str = "siblings",
    ) -> None:
        self.bank_path = Path(bank_path) if bank_path is not None else None
        self.retrieve_top_k = retrieve_top_k
        self.retrieval_prompt = retrieval_prompt
        self.update_prompt = update_prompt
        self.scope = scope
        self.experiences: dict[str, RubricExperience] = (
            self.load() if self.bank_path and self.bank_path.exists() else _seed_experience_records(scope=self.scope)
        )

    def load(self) -> dict[str, RubricExperience]:
        payload = json.loads(self.bank_path.read_text(encoding="utf-8")) if self.bank_path is not None else {}
        if isinstance(payload, dict):
            items = payload.get("experiences") or payload.get("after") or payload
        else:
            items = payload
        experiences = {}
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict):
                experience = _convert_experience(item, str(item.get("experience_id") or "") or None)
                if experience is not None:
                    experiences[experience.title] = experience
        return experiences or _seed_experience_records(scope=self.scope)

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
        temperature: float,
        top_p: float,
        max_tokens: int,
        model_kwargs: dict[str, Any] | None = None,
    ) -> RubricBankGenerationContext:
        retrieved, retrieve_messages = await self._retrieve(
            question=question,
            previous_state=previous_state,
            latest_shared_segment=latest_shared_segment,
            continuations=continuations,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
        )
        sections = []
        if retrieved:
            sections.append(
                "## Retrieved Rubric Experiences:\n" +
                render_compact_markdown(
                    [
                        {
                            "title": item.title,
                            "description": item.description,
                            "context": item.context,
                            "experience": item.experience,
                            "reference_golden_rubrics": item.metadata.get("reference_golden_rubrics", None),
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

    async def _retrieve(
        self,
        *,
        question: dict[str, Any],
        previous_state: dict[str, Any],
        latest_shared_segment: dict[str, Any] | None,
        continuations: list[dict[str, Any]],
        model_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        model_kwargs: dict[str, Any] | None,
    ) -> tuple[list[RubricExperience], list[dict[str, Any]]]:
        if not self.experiences:
            return [], []
        prompt = "\n\n".join(
            [
                self.retrieval_prompt.strip(),
                "## Experience Short-view:",
                "\n\n".join(
                    f"### {item.title}\n{item.description}"
                    for item in self.experiences.values()
                ),
                "## Current Round Context:",
                render_compact_markdown(
                    {
                        "question": question,
                        "previous_state": previous_state,
                        "parent trajectory": latest_shared_segment,
                        "continuations": continuations,
                    }
                ),
            ]
        )
        messages = [{"role": "user", "content": prompt}]
        requested_titles = []
        available_titles = list(self.experiences)
        for _ in range(4):
            async for attempt in retry(
                logger=logger,
                abort_exceptions=LitellmModel.abort_exceptions,
                model_name=model_name,
                async_retry=True,
            ):
                with attempt:
                    assistant_message = await route_completion_message(
                        route_name="rubric_generation",
                        model_name=model_name,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        model_kwargs=freeform_thought_model_kwargs(model_kwargs),
                    )
            response = assistant_message.get("content_no_thinking") or assistant_message.get("content") or ""
            messages.append(assistant_message)
            parsed = extract_last_json_object(response)
            full_response = assistant_message.get("content") or ""
            if parsed is None and full_response != response:
                parsed = extract_last_json_object(full_response)
            parsed_titles = None
            if isinstance(parsed, dict):
                for key in ("titles", "experience_titles", "selected_titles"):
                    if key in parsed:
                        parsed_titles = parsed[key]
                        break
            if isinstance(parsed_titles, str):
                parsed_titles = [parsed_titles]
            if isinstance(parsed_titles, list) and all(
                isinstance(item, dict) and isinstance(item.get("title"), str) for item in parsed_titles
            ):
                parsed_titles = [item["title"] for item in parsed_titles]
            if isinstance(parsed_titles, list) and all(isinstance(title, str) for title in parsed_titles):
                requested_titles = []
                for title in parsed_titles:
                    matched_title = _match_experience_title(title, available_titles)
                    if matched_title is not None and matched_title not in requested_titles:
                        requested_titles.append(matched_title)
                    if len(requested_titles) >= self.retrieve_top_k:
                        break
                if not parsed_titles or requested_titles:
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Some requested titles do not exist in the experience index. Follow the output format example "
                            "above using only existing titles, or {\"titles\": []}."
                        ),
                    }
                )
                continue
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Expected {\"titles\": [\"existing experience title\"]} in the final JSON. Follow the output "
                        "format example above."
                    ),
                }
            )
        return [self.experiences[title] for title in requested_titles], messages

    async def update_after_instance(
        self,
        *,
        instance: dict[str, Any],
        rubric_payloads: list[dict[str, Any]] | None = None,
        model_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        before = self.to_list()
        grouped_payloads = self._group_payloads_by_index(rubric_payloads or [])
        if grouped_payloads is None:
            update = await self._update_from_payloads(
                instance=instance,
                rubric_payloads=rubric_payloads or [],
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                model_kwargs=model_kwargs,
            )
            payload = {
                "before": before,
                "actions": update["actions"],
                "after": update["after"],
                "messages": update["messages"],
            }
        else:
            index_key, grouped_items = grouped_payloads
            group_updates: list[dict[str, Any]] = []
            actions: list[dict[str, Any]] = []
            messages: list[dict[str, Any]] = []
            for group_index, group_payloads in grouped_items:
                update = await self._update_from_payloads(
                    instance=instance,
                    rubric_payloads=group_payloads,
                    model_name=model_name,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    model_kwargs=model_kwargs,
                )
                if not update["has_evidence"]:
                    continue
                group_update = {
                    index_key: group_index,
                    "before": update["before"],
                    "actions": update["actions"],
                    "after": update["after"],
                    "messages": update["messages"],
                }
                group_updates.append(group_update)
                actions.extend(
                    {
                        **copy.deepcopy(action),
                        index_key: group_index,
                    }
                    for action in update["actions"]
                )
                messages.append(
                    {
                        index_key: group_index,
                        "messages": copy.deepcopy(update["messages"]),
                    }
                )
            payload = {
                "before": before,
                "actions": actions,
                "after": self.to_list(),
                "messages": messages,
                "groups": group_updates,
            }
        return payload

    @staticmethod
    def _group_payloads_by_index(
        rubric_payloads: list[dict[str, Any]],
    ) -> tuple[str, list[tuple[int, list[dict[str, Any]]]]] | None:
        if not rubric_payloads:
            return None
        index_key = None
        for candidate in ("group_index", "round_index"):
            has_index = [candidate in payload for payload in rubric_payloads]
            if any(has_index):
                if not all(has_index):
                    raise ValueError(f"rubric_payloads must either all include {candidate} or none include {candidate}")
                index_key = candidate
                break
        if index_key is None:
            return None
        grouped: dict[int, list[dict[str, Any]]] = {}
        order: list[int] = []
        for payload in rubric_payloads:
            group_index = int(payload[index_key])
            if group_index not in grouped:
                grouped[group_index] = []
                order.append(group_index)
            grouped[group_index].append(
                {
                    key: copy.deepcopy(value)
                    for key, value in payload.items()
                    if key != index_key
                }
            )
        return index_key, [(group_index, grouped[group_index]) for group_index in order]

    async def _update_from_payloads(
        self,
        *,
        instance: dict[str, Any],
        rubric_payloads: list[dict[str, Any]],
        model_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        model_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        before = self.to_list()
        evidence = self._build_instance_evidence(instance=instance, rubric_payloads=rubric_payloads)
        if evidence.get("rubric_attempts"):
            actions, messages = await self._update_bank_from_evidence(
                evidence=evidence,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                model_kwargs=model_kwargs,
            )
        else:
            actions = []
            messages = []
        return {
            "before": before,
            "actions": actions,
            "after": self.to_list(),
            "messages": messages,
            "has_evidence": bool(evidence.get("rubric_attempts")),
        }

    def _build_instance_evidence(
        self,
        *,
        instance: dict[str, Any],
        rubric_payloads: list[dict[str, Any]],
    ) -> dict[str, Any]:
        gt_skeleton = gold_patch_skeleton(instance.get("patch", ""))
        attempts = []
        for payload in rubric_payloads:
            generation_context = payload.get("generation_context")
            if not isinstance(generation_context, dict):
                generation_context = {}
            avg_scores = payload.get("average_rubric_judged_scores", {})
            ordered_node_ids = [str(node_id) for node_id in avg_scores]
            generated = payload.get("generated", [])
            ground_truth_by_node = (
                payload.get("gt_by_rubric", {}).get(str(generated[0].get("rubric_id")), {}).get("ground_truth_by_node", {})
                if generated else {}
            )
            if not _has_reward_variance(ground_truth_by_node, ordered_node_ids):
                continue
            generated_rubrics = [
                {
                    "polarity": rubric["direction"],
                    "weight": rubric["weight"],
                    "title": rubric["title"],
                    "description": rubric["description"],
                    "metadata": copy.deepcopy(rubric["metadata"]),
                    "scale": copy.deepcopy(rubric["scale"]),
                }
                for rubric in generated
            ]
            generated_rubric_accuracy = _rubric_accuracy_payload(
                generated=generated,
                score_by_rubric={
                    str(rubric_id): {str(node_id): float(score) for node_id, score in node_scores.items()}
                    for rubric_id, node_scores in (payload.get("score_by_rubric") or {}).items()
                    if isinstance(node_scores, dict)
                },
                gt_scores=ground_truth_by_node,
                ordered_node_ids=ordered_node_ids,
                scope=self.scope,
            )
            attempts.append(
                {
                    "generation_context": copy.deepcopy(generation_context),
                    "retrieved_experience": [_public_experience(item) for item in payload.get("retrieved", [])],
                    "generated_rubrics": generated_rubrics,
                    "gt_skeleton": gt_skeleton,
                    "terminal_patch": str(payload.get("terminal_patch") or ""),
                    "passed_tests": str(payload.get("passed_tests") or ""),
                    "gt_scores": _rounded_score_list(ground_truth_by_node, ordered_node_ids),
                    "average_rubric_judged_scores": _rounded_score_list(avg_scores, ordered_node_ids),
                    "generated_rubric_accuracy": generated_rubric_accuracy,
                }
            )
        return {
            "instance_id": instance.get("instance_id"),
            "rubric_attempts": attempts,
        }

    async def _update_bank_from_evidence(
        self,
        *,
        evidence: dict[str, Any],
        model_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        model_kwargs: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        applied = []
        update_messages = []
        for attempt_evidence in evidence["rubric_attempts"]:
            initial_prompt = "\n\n".join(
                [
                    self.update_prompt.strip(),
                    "## Current Rubric Bank:",
                    render_compact_markdown(
                        [
                            {"title": item.title, "description": item.description}
                            for item in self.experiences.values()
                        ]
                    ),
                    "## Previous Rubric Generation Attempt:",
                    render_compact_markdown(
                        {
                            "instance_id": evidence.get("instance_id"),
                            **attempt_evidence,
                        }
                    ),
                ]
            )
            messages = [{"role": "user", "content": initial_prompt}]
            add_update_count = 0
            for _ in range(8):
                async for attempt in retry(
                    logger=logger,
                    abort_exceptions=LitellmModel.abort_exceptions,
                    model_name=model_name,
                    async_retry=True,
                ):
                    with attempt:
                        assistant_message = await route_completion_message(
                            route_name="rubric_generation",
                            model_name=model_name,
                            messages=messages,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            model_kwargs=freeform_thought_model_kwargs(model_kwargs),
                        )
                response = assistant_message.get("content_no_thinking") or assistant_message.get("content") or ""
                messages.append(assistant_message)
                parsed = extract_last_json_object(response)
                if parsed == {}:
                    if add_update_count == 0:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Generate at least one add or update action with concrete observable rubric-generation guidance, "
                                    "or retrieve an existing experience first if needed. Follow the output format example above."
                                ),
                            }
                        )
                        continue
                    break
                converted_action = _convert_action(parsed, experiences=self.experiences, attempt_evidence=attempt_evidence)
                if converted_action is None:
                    valid_titles = [item.title for item in self.experiences.values()]
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The response did not match a valid retrieve/add/update/delete action schema, or it referenced an invalid bank title. "
                                "For add/update, title, description, context, experience, metadata.analysis, and "
                                "metadata.reference_golden_rubrics must be top-level fields in the new flat format. "
                                "Current valid experience titles are:\n"
                                + render_compact_markdown(valid_titles)
                                + "\nFollow the output format example above with a corrected single action or {}."
                            ),
                        }
                    )
                    continue
                action = converted_action["action"]
                if action == "retrieve":
                    titles = converted_action["titles"]
                    retrieved = [_public_experience(self.experiences[title]) for title in titles]
                    applied.append(copy.deepcopy(converted_action))
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "## Retrieved Existing Experiences:\n"
                                + render_compact_markdown(retrieved)
                                + "\n\nUse this retrieved context to decide the next add, update, delete, or retrieve action. "
                                + "Follow the output format example above, or return {} when no "
                                + "more bank changes are needed for this rubric generation attempt."
                            ),
                        }
                    )
                    continue
                if action == "delete":
                    title = converted_action["title"]
                    del self.experiences[title]
                    applied.append(copy.deepcopy(converted_action))
                elif action in {"add", "update"}:
                    experience = converted_action["experience"]
                    if action == "update":
                        del self.experiences[converted_action["target_title"]]
                    self.experiences[experience.title] = experience
                    applied_action = {"action": action, **_public_experience(experience)}
                    if action == "update":
                        applied_action["target_title"] = converted_action["target_title"]
                    applied.append(applied_action)
                    add_update_count += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Applied the action to the rubric experience bank successfully.\nGenerate the next action or return "
                            "an empty object. Follow the output format example above."
                        ),
                    }
                )
            update_messages.append(
                {
                    "rubric_list_id": attempt_evidence.get("rubric_list_id"),
                    "messages": copy.deepcopy(messages),
                }
            )
        return applied, update_messages
