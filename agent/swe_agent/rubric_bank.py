from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from agent_rl.run_utils import extract_json_from_response, run_chat_with_route_async

from swe_agent.models.litellm_model import LitellmModel
from swe_agent.models.utils.retry import retry
from swe_agent.parallel_utils import _atomic_write_json
from swe_agent.prompt import (
    RUBRIC_EXPERIENCE_RETRIEVAL_PROMPT,
    RUBRIC_EXPERIENCE_RETRIEVAL_RESPONSE_FORMAT,
    RUBRIC_EXPERIENCE_UPDATE_PROMPT,
    RUBRIC_EXPERIENCE_UPDATE_RESPONSE_FORMAT,
    _seed_experiences,
    _seed_rubrics,
)


logger = logging.getLogger(__name__)


@dataclass
class RubricRecord:
    rubric_id: str
    title: str
    direction: Literal["positive", "negative"]
    description: str
    scale: dict[str, str]
    weight: int
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


def _first_user_message(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content", "")
            return content if isinstance(content, str) else ""
    return ""


def _section_text(text: str, header: str, next_header: str | None = None) -> str:
    start = text.find(header)
    if start < 0:
        return ""
    start += len(header)
    end = text.find(next_header, start) if next_header else -1
    return text[start:end if end >= 0 else len(text)].strip()


def _json_after_header(text: str, header: str) -> Any:
    start = text.find(header)
    if start < 0:
        return None
    json_start = text.find("{", start + len(header))
    if json_start < 0:
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text[json_start:])
        return parsed
    except json.JSONDecodeError:
        return None


def _json_from_section(text: str, header: str, next_header: str | None = None) -> Any:
    section = _section_text(text, header, next_header)
    if not section or section == "None":
        return None
    starts = [position for position in (section.find("{"), section.find("[")) if position >= 0]
    if not starts:
        return None
    json_start = min(starts)
    try:
        parsed, _ = json.JSONDecoder().raw_decode(section[json_start:])
        return parsed
    except json.JSONDecodeError:
        return None


def _extract_continuations_from_prompt(prompt: str) -> list[dict[str, Any]]:
    continuations = []
    for match in re.finditer(r"^## Continuation (\d+):\s*$", prompt, flags=re.MULTILINE):
        parsed = _json_after_header(prompt[match.start():], match.group(0))
        if isinstance(parsed, dict):
            continuations.append(
                {
                    "sample_index": int(match.group(1)),
                    **parsed,
                }
            )
    return continuations


def _extract_generation_context_from_messages(messages: Any) -> dict[str, Any]:
    prompt = _first_user_message(messages)
    if not prompt:
        return {}
    context: dict[str, Any] = {
        "question": _section_text(prompt, "## Question:", "## Previous Persistent State:"),
        "previous_state": _json_from_section(prompt, "## Previous Persistent State:", "## Latest Agent Trajectory:"),
        "latest_shared_segment": _json_from_section(prompt, "## Latest Agent Trajectory:", "## Agent Trajectory Continuations:"),
        "continuations": _extract_continuations_from_prompt(prompt),
    }
    retrieved = _json_from_section(prompt, "## Retrieved Rubric Experiences:")
    if isinstance(retrieved, list):
        context["retrieved_rubric_experiences"] = retrieved
    previous_generated_rubrics = _json_from_section(prompt, "## Existing Rubrics:")
    if isinstance(previous_generated_rubrics, list):
        context["previous_generated_rubrics"] = previous_generated_rubrics
    return context


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
    half = max_chars // 2
    return text[:half].rstrip() + "\n...[truncated]...\n" + text[-half:].lstrip()


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
    required_metadata_types = {
        "generated_rubrics": list,
        "gt_skeleton": str,
        "generated_rubric_accuracy": dict,
        "gt_scores": list,
        "analysis": str,
        "reference_golden_rubrics": list,
    }
    if set(metadata) != set(required_metadata_types):
        return None
    for key, expected_type in required_metadata_types.items():
        if not isinstance(metadata.get(key), expected_type):
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
    expected_keys = {"action", "experience"} if action == "add" else {"action", "target_title", "experience"}
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
    experience_payload = copy.deepcopy(payload.get("experience"))
    if not isinstance(experience_payload, dict):
        return None
    title = experience_payload.get("title")
    if title is None or not str(title).strip():
        return None
    title = str(title).strip()
    experience_payload["title"] = title
    if action == "add" and title in experiences:
        return None
    if action == "update" and title != target_title and title in experiences:
        return None
    metadata = experience_payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    analysis = metadata.get("analysis")
    reference_golden_rubrics = metadata.get("reference_golden_rubrics")
    if not isinstance(analysis, str) or not analysis.strip():
        return None
    if not isinstance(reference_golden_rubrics, list) or not reference_golden_rubrics:
        return None
    for reference in reference_golden_rubrics:
        if not isinstance(reference, dict) or _convert_generated_rubric("reference", reference, 0) is None:
            return None
    experience_payload["metadata"] = {
        "generated_rubrics": copy.deepcopy(attempt_evidence["generated_rubrics"]),
        "gt_skeleton": attempt_evidence["gt_skeleton"],
        "generated_rubric_accuracy": copy.deepcopy(attempt_evidence["generated_rubric_accuracy"]),
        "gt_scores": copy.deepcopy(attempt_evidence["gt_scores"]),
        "analysis": analysis,
        "reference_golden_rubrics": copy.deepcopy(reference_golden_rubrics),
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
            node_id: (1.0 - float(raw_scores.get(node_id)) if direction == "negative" else float(raw_scores.get(node_id)))
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


def _seed_experience_records() -> dict[str, RubricExperience]:
    return {
        experience.title: experience
        for seed in _seed_experiences()
        if (experience := _convert_experience(seed)) is not None
    }


def _convert_generated_rubric(task: str, payload: dict[str, Any], round_index: int) -> RubricRecord | None:
    item = payload.get("rubric", None)
    if item is None or not isinstance(item, dict):
        return None
    direction = item.get("polarity", None)
    if direction not in {"positive", "negative"}:
        return None
    title = item.get("title", None)
    if title is None or not isinstance(title, str):
        return None
    description = item.get("description", None)
    if description is None or not isinstance(description, str):
        return None
    scale = {str(score): text for score, text in (item.get("scale") or {}).items()}
    if set(scale) != {"1", "2", "3", "4", "5"}:
        return None
    scale_text = {isinstance(text, str) for _, text in (item.get("scale") or {}).items()}
    if not scale_text or not all(scale_text):
        return None
    metadata_raw = item.get("metadata") or {}
    if not isinstance(metadata_raw, dict):
        return None
    metadata = {str(key): str(value) for key, value in metadata_raw.items()}
    rubric_id = hashlib.md5(
        json.dumps(
            {
                "task": task,
                "direction": direction,
                "title": title,
                "description": description,
                "metadata": metadata,
                "scale": scale,
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
        weight=1 if direction == "positive" else -1,
        source_round=round_index,
        metadata=metadata,
    )


def _seed_rubric_records(task: str) -> list[RubricRecord]:
    return [rubric for seed in _seed_rubrics() if (rubric := _convert_generated_rubric(task, seed, 0)) is not None]


class ScoreRubricBank:
    def __init__(self, *, max_active_rubrics: int) -> None:
        self.max_active_rubrics = max_active_rubrics
        self.active_bank: list[RubricRecord] = []
        self.inactive_bank: list[RubricRecord] = []

    def initialize(self, task: str) -> None:
        self.active_bank = _seed_rubric_records(task)
        self.inactive_bank = []

    def set_state(self, *, active_bank: list[RubricRecord], inactive_bank: list[RubricRecord]) -> None:
        self.active_bank = copy.deepcopy(active_bank)
        self.inactive_bank = copy.deepcopy(inactive_bank)

    def build_generation_context(self, **_: Any) -> RubricBankGenerationContext:
        sections = []
        if self.active_bank:
            sections.append(
                "## Existing Rubrics:\n"
                + json.dumps(
                    [{
                            "polarity": rubric.direction,
                            "title": rubric.title,
                            "description": rubric.description,
                            "scale": rubric.scale,
                            "metadata": rubric.metadata,
                        }
                        for rubric in self.active_bank
                    ])
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
        deduped_by_title: dict[str, RubricRecord] = {}
        kept_active_bank = [rubric for rubric in self.active_bank if rubric.reward is not None and rubric.reward > 0.0]
        for rubric in kept_active_bank + generated:
            candidate = RubricRecord(**{**asdict(rubric), "reward": rewards.get(rubric.rubric_id, 0.0)})
            title_key = candidate.title.strip().casefold()
            existing = deduped_by_title.get(title_key)
            if existing is None or (candidate.reward or 0.0) > (existing.reward or 0.0) or (
                (candidate.reward or 0.0) == (existing.reward or 0.0) and candidate.source_round > existing.source_round
            ):
                deduped_by_title[title_key] = candidate
        ranked = list(deduped_by_title.values())
        ranked.sort(key=lambda rubric: (rubric.reward or 0.0), reverse=True)
        active_after = ranked[: self.max_active_rubrics]
        active_ids = {rubric.rubric_id for rubric in active_after}
        inactive_after = [rubric for rubric in ranked + self.inactive_bank if rubric.rubric_id not in active_ids]
        return RubricBankRoundUpdate(
            rubrics=active_after,
            active_after=active_after,
            inactive_after=inactive_after,
        )


class ExperienceRubricBank:
    def __init__(self, *, bank_path: Path | None = None, retrieve_top_k: int = 4, write_artifacts: bool = True) -> None:
        self.bank_path = Path(bank_path) if bank_path is not None else None
        self.retrieve_top_k = retrieve_top_k
        self.write_artifacts = write_artifacts
        self.experiences: dict[str, RubricExperience] = (
            self.load() if self.bank_path and self.bank_path.exists() else _seed_experience_records()
        )

    def load(self) -> dict[str, RubricExperience]:
        payload = json.loads(self.bank_path.read_text(encoding="utf-8")) if self.bank_path is not None else {}
        items = payload.get("experiences", payload) if isinstance(payload, dict) else payload
        experiences = {}
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict):
                experience = _convert_experience(item, str(item.get("experience_id") or "") or None)
                if experience is not None:
                    experiences[experience.title] = experience
        return experiences or _seed_experience_records()

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
                json.dumps([
                    {
                        "title": item.title,
                        "description": item.description,
                        "context": item.context,
                        "experience": item.experience,
                        "reference_golden_rubric": item.metadata.get("reference_golden_rubrics", None),
                    }
                    for item in retrieved
                ])
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
                RUBRIC_EXPERIENCE_RETRIEVAL_PROMPT.strip(),
                "## Experience Short-view:",
                json.dumps([
                    {
                        "title": item.title,
                        "description": item.description,
                    }
                    for item in self.experiences.values()
                ]),
                "## Current Round Context:",
                json.dumps({
                    "question": question,
                    "previous_state": previous_state,
                    "latest_shared_segment": latest_shared_segment,
                    "continuations": continuations,
                })
            ]
        )
        messages = [{"role": "user", "content": prompt}]
        requested_titles = []
        for _ in range(4):
            async for attempt in retry(
                logger=logger,
                abort_exceptions=LitellmModel.abort_exceptions,
                model_name=model_name,
                async_retry=True,
            ):
                with attempt:
                    response = await run_chat_with_route_async(
                        "rubric_generation",
                        model_name=model_name,
                        user_prompt=messages[-1]["content"] if len(messages) == 1 else None,
                        messages=messages if len(messages) > 1 else None,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        response_format=copy.deepcopy(RUBRIC_EXPERIENCE_RETRIEVAL_RESPONSE_FORMAT),
                        enable_json_schema_validation=True,
                        **(model_kwargs or {}),
                    )
            messages.append({"role": "assistant", "content": response or ""})
            parsed = extract_json_from_response(response or "")
            if isinstance(parsed, dict) and isinstance(parsed.get("titles"), list):
                requested_titles = [title for title in parsed["titles"][: self.retrieve_top_k] if title in self.experiences]
                if len(requested_titles) == len(parsed["titles"][: self.retrieve_top_k]):
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": "Some requested titles do not exist in the experience index. Return corrected JSON using only existing titles, or {\"titles\": []}.",
                    }
                )
                continue
            messages.append(
                {
                    "role": "user",
                    "content": "Expected JSON exactly in the format {\"titles\": [\"existing experience title\"]}. Return corrected JSON only.",
                }
            )
        return [self.experiences[title] for title in requested_titles], messages

    async def update_after_instance(
        self,
        *,
        run_dir: Path,
        instance: dict[str, Any],
        rubric_payloads: list[dict[str, Any]] | None = None,
        model_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        before = self.to_list()
        evidence = self._build_instance_evidence(instance=instance, rubric_payloads=rubric_payloads or [])
        if evidence.get("rubric_attempts"):
            applied, update_messages = await self._update_bank_from_evidence(
                evidence=evidence,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                model_kwargs=model_kwargs,
            )
        else:
            applied = []
            update_messages = []
        if self.write_artifacts and self.bank_path is not None:
            _atomic_write_json(self.bank_path, {"experiences": self.to_list()})
        payload = {
            "before": before,
            "actions": applied,
            "after": self.to_list(),
            "messages": update_messages,
        }
        if self.write_artifacts:
            _atomic_write_json(Path(run_dir) / "rubric_bank.json", payload)
            _atomic_write_json(Path(run_dir) / "rubric_bank_message.json", update_messages)
        return payload

    def _build_instance_evidence(
        self,
        *,
        instance: dict[str, Any],
        rubric_payloads: list[dict[str, Any]],
    ) -> dict[str, Any]:
        gt_skeleton = gold_patch_skeleton(instance.get("patch", ""))
        attempts = []
        for payload in rubric_payloads:
            generation_context = _extract_generation_context_from_messages(payload.get("messages"))
            child_rewards = payload.get("child_rewards", {})
            ordered_node_ids = [str(node_id) for node_id in child_rewards]
            generated = payload.get("generated", [])
            ground_truth_by_node = (
                payload.get("gt_by_rubric", {}).get(str(generated[0].get("rubric_id")), {}).get("ground_truth_by_node", {})
                if generated else {}
            )
            if not _has_reward_variance(ground_truth_by_node, ordered_node_ids):
                continue
            generated_rubrics = [
                {
                    "rubric": {
                        "polarity": rubric["direction"],
                        "title": rubric["title"],
                        "description": rubric["description"],
                        "metadata": copy.deepcopy(rubric["metadata"]),
                        "scale": copy.deepcopy(rubric["scale"]),
                    }
                }
                for rubric in generated
            ]
            generated_rubric_accuracy = _rubric_accuracy_payload(
                generated=generated,
                score_by_rubric={
                    str(rubric_id): {str(node_id): float(score) for node_id, score in node_scores.items()}
                    for rubric_id, node_scores in (payload.get("child_score_by_rubric") or {}).items()
                    if isinstance(node_scores, dict)
                },
                gt_scores=ground_truth_by_node,
                ordered_node_ids=ordered_node_ids,
            )
            attempts.append(
                {
                    "generation_context": copy.deepcopy(generation_context),
                    "retrieved_experience": [_public_experience(item) for item in payload.get("retrieved", [])],
                    "generated_rubrics": generated_rubrics,
                    "gt_skeleton": gt_skeleton,
                    "gt_scores": _rounded_score_list(ground_truth_by_node, ordered_node_ids),
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
                    RUBRIC_EXPERIENCE_UPDATE_PROMPT.strip(),
                    "## Current Rubric Bank:",
                    json.dumps([
                        {"title": item.title, "description": item.description}
                        for item in self.experiences.values()
                    ]),
                    "## Previous Rubric Generation Attempt:",
                    json.dumps(
                        {
                            "instance_id": evidence.get("instance_id"),
                            **attempt_evidence,
                        }
                    ),
                ]
            )
            messages = [{"role": "user", "content": initial_prompt}]
            for _ in range(8):
                async for attempt in retry(
                    logger=logger,
                    abort_exceptions=LitellmModel.abort_exceptions,
                    model_name=model_name,
                    async_retry=True,
                ):
                    with attempt:
                        response = await run_chat_with_route_async(
                            "rubric_generation",
                            model_name=model_name,
                            user_prompt=messages[-1]["content"] if len(messages) == 1 else None,
                            messages=messages if len(messages) > 1 else None,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            response_format=copy.deepcopy(RUBRIC_EXPERIENCE_UPDATE_RESPONSE_FORMAT),
                            enable_json_schema_validation=False,
                            **(model_kwargs or {}),
                        )
                response = response or ""
                messages.append({"role": "assistant", "content": response})
                parsed = extract_json_from_response(response)
                if parsed == {}:
                    break
                converted_action = _convert_action(parsed, experiences=self.experiences, attempt_evidence=attempt_evidence)
                if converted_action is None:
                    valid_titles = [item.title for item in self.experiences.values()]
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The response did not match a valid retrieve/add/update/delete action schema, or it referenced an invalid bank title. "
                                "For add/update, experience.metadata.analysis must be non-empty and "
                                "experience.metadata.reference_golden_rubrics must be in the correct rubric format. "
                                "Current valid experience titles are: "
                                + json.dumps(valid_titles, ensure_ascii=False)
                                + ". Generate a corrected single action or return {}."
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
                                + json.dumps(retrieved)
                                + "\n\nUse this retrieved context to decide the next add, update, delete, or retrieve action. "
                                + "Return {} when no more bank changes are needed for this rubric generation attempt."
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
                    applied_action = {"action": action, "experience": _public_experience(experience)}
                    if action == "update":
                        applied_action["target_title"] = converted_action["target_title"]
                    applied.append(applied_action)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Applied the action to the rubric experience bank successfully.\nGenerate the next action or return an empty object."
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
