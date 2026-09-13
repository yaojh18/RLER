"""Training-only direct judging with frozen per-instance golden rubrics.

This module deliberately has no experience retrieval, persistent-state update,
or rubric-generation path.  It is shared by the training-only lane and naive
collectors; the TTS trajectory-search implementation keeps its existing judge
pipeline.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from agent_rl.run_utils import (
    extract_last_json_object,
    freeform_thought_model_kwargs,
    route_completion_message,
)
from swe_agent.models.litellm_model import LitellmModel
from swe_agent.models.utils.retry import retry
from swe_agent.rubric_bank import RubricRecord, _convert_rubric_item, _truncate_middle
from swe_agent.trajectory_search import (
    _render_continuation_view,
    _score_round,
)
logger = logging.getLogger("swe_agent.direct_rubric_judge")

DEFAULT_RUBRIC_BANK = (
    Path(__file__).resolve().parents[2]
    / "rubric"
    / "qwen"
    / "golden_reference_rubrics.json"
)
PREFIX_VIEW_PROFILES = (
    ("raw", None, None),
    ("compact-512-80k", 512, 80_000),
    ("compact-384-60k", 384, 60_000),
    ("compact-256-40k", 256, 40_000),
    ("compact-192-24k", 192, 24_000),
    ("compact-128-16k", 128, 16_000),
    ("compact-96-12k", 96, 12_000),
    ("compact-64-8k", 64, 8_000),
)

DIRECT_ABSTAIN_SYSTEM_PROMPT = """
You are the conservative same-reward gate for an early-trajectory reward predictor. The eight continuations are independent prefixes from the same root. Judge only their current visible behavior, command observations, repository state, tests, evaluator-facing artifacts, and the supplied golden reference rubrics. Never predict later actions or outcomes.

`should_abstain=true` requires strong positive visible evidence that all eight prefixes are materially equivalent for final evaluator reward. If any prefix has a credible task-relevant semantic, artifact, failure, recovery, or discriminative-validation difference, return `should_abstain=false`. If the supplied reference rubrics do not cover a visible difference, do not assume equality; return false. Optimize for recall of truly different-reward groups because a false abstention cannot be recovered by the numeric detector.

Workflow stage, exploration depth, verbosity, generic validation volume, and harmless temporary files do not establish reward equality or difference by themselves. A reversible intermediate mistake is not a terminal flaw after visible recovery, and an unfinished prefix is not an empty submission.

For `terminal_at_cutoff=true`, the terminal assistant payload is the
evaluator-facing submission. A `non_patch` terminal payload cannot be rescued
by a valid workspace source or prose. If one terminal prefix is `unified_diff`
and another is `non_patch`, that visible format difference must block
abstention. `terminal_at_cutoff=false` means terminal-artifact criteria are
inapplicable.

Return exactly one JSON object:
{
  "should_abstain": <boolean>,
  "confidence": <number from 0 to 1>,
  "same_reward_evidence": ["visible evidence supporting material equivalence"],
  "variance_evidence": ["visible evidence that prevents abstention"]
}
""".strip()

DIRECT_RUBRIC_JUDGE_PROMPT = """
You are an expert evaluator scoring one SWE-agent trajectory prefix against one rubric.

Judge only the specified criterion using evidence visible in the continuation. Use the evidence hierarchy that matches the trajectory state:
- When an irreversible or unmistakably terminal submission contains a parseable patch, that submitted patch is the evaluator-facing implementation and takes precedence over an empty or stale workspace summary.
- For an unfinished nonterminal prefix, the latest workspace summary and optional cutoff git diff describe repository state and take precedence over superseded intermediate edits or unsupported claims.
- A later visible source or diff supersedes an earlier edit; prose never supersedes a concrete artifact.
- A concrete edit that visibly succeeded remains part of the repository state unless a later visible edit, revert, source view, or diff supersedes it. Do not erase an observed edit merely because a later workspace summary is empty or unavailable.

The stage named by the rubric is an evidence category, not a prediction of what the trajectory will do next and not a scalar measure of progress. Score the concrete behavior and evidence already visible at that stage. Do not fill missing evidence with an imagined future, and do not treat uncertainty about later behavior as either success or failure.

Apply the rubric only where its stated criterion is observable. Missing a future action in an unfinished prefix is unknown, not a flaw. In particular, an unfinished branch with no terminal submission must receive the no-flaw anchor on a negative final-artifact rubric; it must not be scored as though it submitted an empty or invalid patch. When a positive final-artifact rubric must compare terminal and unfinished branches, use its explicitly neutral unfinished anchor rather than the lowest score.

The structured `terminal_payload_format` field is an observable format check, not a semantic score. Trust `unified_diff` versus `non_patch` for payload-format applicability, then inspect the actual diff separately for task correctness.

Do not infer hidden tests, future edits, later recovery, or final outcomes. When the rubric explicitly evaluates causal recovery, score only the diagnosis and correction already made concrete in the prefix; do not require the branch to be at a later workflow stage.

Use the supplied scale exactly. For a positive rubric, 1 is weakest and 5 is strongest. For a negative rubric, 1 means the flaw is absent and 5 means it is severe. Ground the score in exact visible evidence.

Match the most specific applicable anchor, including any stated precedence rule. If an anchor explicitly names the visible implementation or behavior, do not choose a neighboring anchor merely because a more general part of that neighboring description could also apply.

Return exactly one JSON object:
{
  "evidence": "exact visible evidence supporting the selected anchor",
  "score": 1
}
where `score` is an integer from 1 to 5.
""".strip()


@lru_cache(maxsize=8)
def _cached_bank(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


class DirectRubricBank:
    """Process-cached view of the frozen direct-rubric bank."""

    def __init__(self, path: Path | str = DEFAULT_RUBRIC_BANK) -> None:
        self.path = Path(path).resolve()
        self.payload = _cached_bank(str(self.path))
        self._score_mapping = tuple(
            float(value) for value in self.payload["score_mapping"]
        )

    @property
    def score_mapping(self) -> tuple[float, float, float, float, float]:
        return self._score_mapping  # type: ignore[return-value]

    def is_eligible(self, instance_id: str) -> bool:
        return instance_id in self.payload["eligible_instance_ids"]

    def raw_references(self, instance_id: str) -> list[dict[str, Any]]:
        return copy.deepcopy(
            self.payload["instances"][instance_id]["reference_golden_rubrics"]
        )

    def rubrics(
        self,
        instance_id: str,
        task_text: str,
        *,
        references: list[dict[str, Any]] | None = None,
    ) -> list[RubricRecord]:
        raw = self.raw_references(instance_id) if references is None else references
        return cast(
            list[RubricRecord],
            [_convert_rubric_item(task_text, item, 1) for item in raw],
        )


def _render_section(title: str, payload: Any) -> str:
    return f"## {title}\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _terminal_payload_format(branch: dict[str, Any]) -> str:
    if not branch["ended_by_prefix"]:
        return "not_terminal"
    cards = branch["step_cards"]
    payload = str(cards[-1].get("assistant_message") or "") if cards else ""
    lines = payload.splitlines()
    return (
        "unified_diff"
        if any(line.startswith("diff --git ") for line in lines)
        and any(line.startswith("--- ") for line in lines)
        and any(line.startswith("+++ ") for line in lines)
        else "non_patch"
    )


def _continuation_views(prefix: dict[str, Any]) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for branch in prefix["branches"]:
        cards = copy.deepcopy(branch["step_cards"])
        workspace = branch.get("workspace_meta") or {}
        views.append(
            {
                "node_id": branch["node_id"],
                "summary": {
                    "step_count": len(cards),
                    "terminal_at_cutoff": bool(branch["ended_by_prefix"]),
                    "terminal_payload_format": _terminal_payload_format(branch),
                    "changed_files": list(workspace.get("changed_files") or [])[:8],
                    "untracked_files": list(workspace.get("untracked_files") or [])[:8],
                    "git_diff": workspace.get("git_diff") or "",
                },
                "raw_continuation": {
                    "segment_step_range": [1, len(cards)],
                    "step_cards": cards,
                },
            }
        )
    return views


def _compact_prefix(
    prefix: dict[str, Any], field_limit: int | None, diff_limit: int | None
) -> dict[str, Any]:
    if field_limit is None or diff_limit is None:
        return copy.deepcopy(prefix)
    compact = copy.deepcopy(prefix)
    for branch in compact["branches"]:
        for card in branch["step_cards"]:
            card["assistant_message"] = _truncate_middle(
                str(card.get("assistant_message") or ""), field_limit
            )
            card["observation"] = _truncate_middle(
                str(card.get("observation") or ""), field_limit
            )
        workspace = branch.get("workspace_meta") or {}
        if workspace.get("git_diff"):
            workspace["git_diff"] = _truncate_middle(
                str(workspace["git_diff"]), diff_limit
            )
        branch["workspace_meta"] = workspace
    return compact


def _render_group(prefix: dict[str, Any]) -> str:
    parts: list[str] = []
    for branch in prefix["branches"]:
        parts.extend(
            [
                f"## {branch['node_id']}",
                f"terminal_at_cutoff: {str(bool(branch['ended_by_prefix'])).lower()}",
                f"terminal_payload_format: {_terminal_payload_format(branch)}",
            ]
        )
        for card in branch["step_cards"]:
            parts.extend(
                [
                    f"### Step {card['step_index']}",
                    "Assistant:",
                    str(card.get("assistant_message") or ""),
                    "Observation:",
                    str(card.get("observation") or ""),
                ]
            )
        workspace = branch.get("workspace_meta") or {}
        if workspace.get("git_diff"):
            parts.extend(["### Git diff at cutoff", str(workspace["git_diff"])])
    return "\n".join(parts)


def _approximate_tokens(messages: list[dict[str, Any]], model_name: str) -> int:
    text = "\n\n".join(str(message.get("content") or "") for message in messages)
    try:
        import litellm

        return int(litellm.token_counter(model=model_name, messages=messages))
    except Exception:
        return max(1, (len(text) + 3) // 4)


def _choose_view(
    *,
    prefix: dict[str, Any],
    task: str,
    references: list[dict[str, Any]],
    context_length: int,
    model_name: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    handbook_view = {"reference_golden_rubrics": references}
    rubric_chars = max(
        (len(json.dumps(rubric, ensure_ascii=False)) for rubric in references),
        default=0,
    )
    static_judge_chars = len(task) + len(DIRECT_RUBRIC_JUDGE_PROMPT)
    for profile, field_limit, diff_limit in PREFIX_VIEW_PROFILES:
        candidate = _compact_prefix(prefix, field_limit, diff_limit)
        abstain_messages = [
            {"role": "system", "content": DIRECT_ABSTAIN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        _render_section("Task", task),
                        _render_section("Golden reference rubrics", handbook_view),
                        "## Prefix group",
                        _render_group(candidate),
                    ]
                ),
            },
        ]
        continuations = _continuation_views(candidate)
        continuation_chars = max(
            (len(_render_continuation_view(item)) for item in continuations),
            default=0,
        )
        largest_judge_chars = (
            continuation_chars + rubric_chars + static_judge_chars
        )
        abstain_tokens = _approximate_tokens(abstain_messages, model_name)
        approximate_judge_tokens = max(1, (largest_judge_chars + 3) // 4)
        if max(abstain_tokens, approximate_judge_tokens) < context_length:
            return abstain_messages, continuations, {
                "profile": profile,
                "assistant_observation_chars": field_limit,
                "git_diff_chars": diff_limit,
                "abstain_prompt_tokens": abstain_tokens,
                "largest_judge_prompt_tokens_approx": approximate_judge_tokens,
                "context_tokens": context_length,
            }
    raise ValueError("direct rubric judge exceeds context after all view profiles")


async def _explicit_abstain(
    *,
    messages: list[dict[str, Any]],
    model_name: str,
    model_kwargs: dict[str, Any],
    max_tokens: int,
    format_correction_rounds: int,
) -> dict[str, Any]:
    conversation = copy.deepcopy(messages)
    format_errors: list[str] = []
    for _ in range(max(1, format_correction_rounds)):
        async for attempt in retry(
            logger=logger,
            abort_exceptions=LitellmModel.abort_exceptions,
            model_name=model_name,
            async_retry=True,
        ):
            with attempt:
                response = await route_completion_message(
                    route_name="rubric_abstain",
                    model_name=model_name,
                    messages=conversation,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=max_tokens,
                    model_kwargs=freeform_thought_model_kwargs(model_kwargs),
                )
        conversation.append(response)
        content = response.get("content_no_thinking") or response.get("content") or ""
        parsed = extract_last_json_object(content)
        if parsed is None and response.get("content") != content:
            parsed = extract_last_json_object(response.get("content") or "")
        if (
            isinstance(parsed, dict)
            and isinstance(parsed.get("should_abstain"), bool)
            and isinstance(parsed.get("confidence"), (int, float))
            and not isinstance(parsed.get("confidence"), bool)
            and isinstance(parsed.get("same_reward_evidence"), list)
            and isinstance(parsed.get("variance_evidence"), list)
        ):
            return {
                "parsed": parsed,
                "messages": conversation,
                "format_errors": format_errors,
            }
        format_errors.append("invalid abstain response schema")
        conversation.append(
            {
                "role": "user",
                "content": (
                    "Return exactly one JSON object with boolean `should_abstain`, "
                    "numeric `confidence`, and list-valued `same_reward_evidence` "
                    "and `variance_evidence`. Do not add prose."
                ),
            }
        )
    raise ValueError("invalid abstain response after format corrections")


def _weighted_scores(
    *,
    node_ids: list[str],
    score_records: list[list[dict[str, Any]]],
    rubrics: list[RubricRecord],
    mapping: tuple[float, float, float, float, float],
) -> tuple[dict[str, float], dict[str, float], float]:
    denominator = sum(float(rubric.weight) for rubric in rubrics)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("invalid direct rubric weight denominator")
    lookup = {
        node_id: {record["rubric_id"]: record for record in records}
        for node_id, records in zip(node_ids, score_records)
    }
    signed_scores: dict[str, float] = {}
    rewards: dict[str, float] = {}
    for node_id in node_ids:
        signed = 0.0
        for rubric in rubrics:
            record = lookup[node_id].get(rubric.rubric_id)
            if record is None:
                raise ValueError(f"missing judge score for {node_id}/{rubric.title}")
            raw = int(record["score_raw"])
            if raw not in range(1, 6):
                raise ValueError(f"invalid raw judge score {raw}")
            mapped = mapping[raw - 1]
            signed += mapped * float(rubric.weight) * (
                -1.0 if rubric.direction == "negative" else 1.0
            )
        signed /= denominator
        signed_scores[node_id] = signed
        rewards[node_id] = 0.5 + signed / (2.0 * mapping[-1])

    scale_span = mapping[-1] - mapping[0]
    weighted_range = 0.0
    for rubric in rubrics:
        values = [
            mapping[int(lookup[node_id][rubric.rubric_id]["score_raw"]) - 1]
            for node_id in node_ids
        ]
        weighted_range += (
            (max(values) - min(values)) / scale_span * float(rubric.weight)
        )
    return rewards, signed_scores, weighted_range / denominator


def _numeric_prediction(feature: float, detector: dict[str, Any]) -> dict[str, Any]:
    threshold = float(detector["threshold"])
    direction = detector.get("variance_direction")
    if direction == "ge":
        predicts_variance = feature >= threshold
    elif direction == "gt":
        predicts_variance = feature > threshold
    elif direction == "le":
        predicts_variance = feature <= threshold
    else:
        raise ValueError("unsupported numeric variance direction")
    return {
        "feature": "weighted_rubric_range_mean",
        "value": feature,
        "threshold": threshold,
        "variance_direction": direction,
        "predicts_variance": predicts_variance,
        "predicts_zero_variance": not predicts_variance,
    }


async def judge_direct_rubric_group(
    *,
    instance_id: str,
    problem_statement: str,
    prefix: dict[str, Any],
    bank: DirectRubricBank,
    judge_model_name: str,
    judge_model_kwargs: dict[str, Any],
    detector: dict[str, Any] | None,
    context_length: int,
    max_tokens: int,
    judge_temperature: float = 0.02,
    judge_top_p: float = 1.0,
    format_correction_rounds: int = 8,
) -> dict[str, Any]:
    raw_references = bank.raw_references(instance_id)
    rubrics = bank.rubrics(
        instance_id,
        problem_statement,
        references=raw_references,
    )
    abstain_messages, continuations, view_budget = _choose_view(
        prefix=prefix,
        task=problem_statement,
        references=raw_references,
        context_length=context_length,
        model_name=judge_model_name,
    )
    question = {"system_prompt": "", "user_prompt": problem_statement}
    score_task = _score_round(
        question=question,
        shared_context={
            "previous_persistent_state": None,
            "latest_agent_trajectory": None,
        },
        continuations=continuations,
        rubrics=rubrics,
        model_name=judge_model_name,
        temperature=judge_temperature,
        top_p=judge_top_p,
        max_tokens=max_tokens,
        model_kwargs=judge_model_kwargs,
        judge_prompt=DIRECT_RUBRIC_JUDGE_PROMPT,
        max_format_correction_rounds=format_correction_rounds,
    )
    if detector is None:
        explicit_task = asyncio.sleep(0, result=None)
    else:
        explicit_task = _explicit_abstain(
            messages=abstain_messages,
            model_name=judge_model_name,
            model_kwargs=judge_model_kwargs,
            max_tokens=max_tokens,
            format_correction_rounds=format_correction_rounds,
        )
    explicit_abstain, (scores, errors) = await asyncio.gather(
        explicit_task, score_task
    )
    if errors:
        raise RuntimeError(f"direct rubric judge errors: {errors}")
    node_ids = [str(continuation["node_id"]) for continuation in continuations]
    score_mapping = bank.score_mapping
    rewards, signed_scores, range_mean = _weighted_scores(
        node_ids=node_ids,
        score_records=scores,
        rubrics=rubrics,
        mapping=score_mapping,
    )
    numeric = _numeric_prediction(range_mean, detector) if detector else None
    explicit_prediction = (explicit_abstain or {}).get("parsed") or {}
    predicted_zero_variance = bool(
        detector
        and (
            explicit_prediction.get("should_abstain", False)
            or (numeric or {}).get("predicts_zero_variance", False)
        )
    )
    return {
        "schema_version": "direct_golden_rubric_group_judge.v1",
        "instance_id": instance_id,
        "rubric_mode": "direct",
        "score_mapping": list(score_mapping),
        "view_budget": view_budget,
        "explicit_abstain": explicit_abstain,
        "numeric_detector": numeric,
        "predicted_zero_variance": predicted_zero_variance,
        "rubrics": [
            {
                "rubric_id": rubric.rubric_id,
                "title": rubric.title,
                "direction": rubric.direction,
                "description": rubric.description,
                "scale": rubric.scale,
                "metadata": rubric.metadata,
                "weight": rubric.weight,
            }
            for rubric in rubrics
        ],
        "judge_scores": rewards,
        "signed_mapped_scores": signed_scores,
        "score_records": scores,
    }
