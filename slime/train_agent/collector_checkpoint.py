"""Serialization helpers for restart-safe asynchronous rollout collectors.

Only logical training state belongs in a checkpoint.  Ray object references,
worker handles, live endpoint reservations, API keys, and provider routes are
process-local and are deliberately excluded; pending task descriptors are
rebound to the resumed policy and current routes before redispatch.
"""

from __future__ import annotations

import copy
from typing import Any

from slime.utils.types import Sample


COLLECTOR_CHECKPOINT_SCHEMA_VERSION = 1

_RUNTIME_TASK_KEYS = {
    "_endpoints_picked",
    "_branch_done_dir",
    "_released_branch_indices",
    "_pinned_endpoints",
    "_policy_done_marker",
    "_policy_slot_released",
    "api_key",
    "policy_api_key",
    "rubric_api_key",
    "direct_judge_api_key",
    "policy_base_url",
    "policy_base_urls",
    "rubric_base_url",
    "direct_judge_api_base",
    "usage_ledger",
}

DIRECT_DIAGNOSTIC_KEYS = (
    "groups",
    "predicted_zero_groups",
    "explicit_abstain_groups",
    "rollouts",
    "collapse_rollouts",
    "gt_pairs",
    "gt_correct",
    "variance_tp",
    "variance_fp",
    "variance_tn",
    "variance_fn",
    "stage1_tp",
    "stage1_fp",
    "stage1_tn",
    "stage1_fn",
)


def restore_direct_diagnostics(payload: Any) -> dict[str, int]:
    restored = dict(payload or {})
    return {
        key: int(restored.get(key, 0))
        for key in DIRECT_DIAGNOSTIC_KEYS
    }


def checkpoint_task_source_group_index(task: dict[str, Any]) -> int:
    """Return the explicitly checkpointed source-attempt cursor."""

    if "source_group_index" not in task:
        raise RuntimeError(
            "checkpoint pending task is missing source_group_index: "
            f"instance={task.get('instance_id')!r}"
        )
    source_group_index = int(task["source_group_index"])
    if source_group_index < 0:
        raise ValueError(
            "checkpoint pending task source_group_index must be "
            f"non-negative, got {source_group_index}"
        )
    return source_group_index


def serialize_sample(sample: Sample) -> dict[str, Any]:
    if hasattr(sample, "to_dict"):
        payload = sample.to_dict()
    else:  # Lightweight unit-test samples.
        payload = dict(vars(sample))
        status = payload.get("status")
        if hasattr(status, "value"):
            payload["status"] = status.value
    if not isinstance(payload, dict):
        raise TypeError("Sample.to_dict() must return a mapping")
    return copy.deepcopy(payload)


def deserialize_sample(payload: dict[str, Any]) -> Sample:
    payload = copy.deepcopy(payload)
    from_dict = getattr(Sample, "from_dict", None)
    if callable(from_dict):
        return from_dict(payload)
    # Lightweight unit-test samples do not expose from_dict.  Let their
    # constructor restore the fields it understands.
    status = payload.get("status")
    status_class = getattr(Sample, "Status", None)
    if isinstance(status, str) and status_class is not None:
        payload["status"] = status_class(status)
    return Sample(**payload)


def serialize_buffer(buffer: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in buffer:
        row = {
            "samples": [
                serialize_sample(sample) for sample in group.samples
            ],
            "rollout_id": int(group.rollout_id),
            "usage_group_id": str(
                getattr(group, "usage_group_id", "") or ""
            ),
        }
        if hasattr(group, "group_kind"):
            row["group_kind"] = str(group.group_kind)
        source_task = getattr(group, "source_task", None)
        if source_task is not None:
            row["source_task"] = checkpoint_task_descriptor(source_task)
        rows.append(row)
    return rows


def deserialize_buffer(
    rows: list[dict[str, Any]],
    buffered_group_class,
) -> list[Any]:
    groups: list[Any] = []
    for row in rows:
        kwargs = {
            "samples": [
                deserialize_sample(sample)
                for sample in list(row.get("samples") or [])
            ],
            "rollout_id": int(row.get("rollout_id", 0)),
            "usage_group_id": str(
                row.get("usage_group_id") or ""
            ),
        }
        if "group_kind" in row:
            kwargs["group_kind"] = str(row["group_kind"])
        if "source_task" in row:
            kwargs["source_task"] = copy.deepcopy(row["source_task"])
        groups.append(buffered_group_class(**kwargs))
    return groups


def checkpoint_task_descriptor(task: dict[str, Any]) -> dict[str, Any]:
    """Strip process-local routing state from a retryable source task."""

    return {
        key: copy.deepcopy(value)
        for key, value in task.items()
        if key not in _RUNTIME_TASK_KEYS
    }


def serialize_attempt_error(
    error: Any,
    limit_field: str,
) -> dict[str, int] | None:
    if error is None:
        return None
    attempted = getattr(error, "attempted_instances", None)
    limit = getattr(error, limit_field, None)
    return {
        "attempted_instances": -1 if attempted is None else int(attempted),
        limit_field: -1 if limit is None else int(limit),
    }


def deserialize_attempt_error(payload, error_class, limit_field: str):
    if not payload:
        return None
    attempted = int(payload.get("attempted_instances", -1))
    limit = int(payload.get(limit_field, -1))
    return error_class(
        attempted_instances=None if attempted < 0 else attempted,
        **{limit_field: None if limit < 0 else limit},
    )

def group_outcome_stats(
    *,
    attempted: int,
    accepted: int,
    invalid: int,
    dynamic_filtered: int,
    excess: int,
    carried_in: int = 0,
    carried_out: int = 0,
    prefix: str = "",
) -> dict[str, float | int]:
    """Build per-update group counts and attempted-denominator rates."""

    stem = f"swe_agent/{prefix}"
    denominator = attempted if attempted > 0 else 1
    return {
        f"{stem}groups_attempted": attempted,
        f"{stem}groups_accepted": accepted,
        f"{stem}groups_invalid": invalid,
        f"{stem}groups_dynamic_filtered": dynamic_filtered,
        f"{stem}groups_excess": excess,
        f"{stem}groups_carried_in": carried_in,
        f"{stem}groups_carried_out": carried_out,
        f"{stem}groups_accepted_rate": (
            accepted / denominator if attempted else 0.0
        ),
        f"{stem}groups_invalid_rate": (
            invalid / denominator if attempted else 0.0
        ),
        f"{stem}groups_dynamic_filtered_rate": (
            dynamic_filtered / denominator if attempted else 0.0
        ),
        f"{stem}groups_excess_rate": (
            excess / denominator if attempted else 0.0
        ),
    }


def observe_direct_group(
    diagnostics: dict[str, int],
    samples: list[Sample],
) -> None:
    """Accumulate direct-judge, collapse, and variance diagnostics."""

    metadata = [sample.metadata or {} for sample in samples]
    if not metadata or any(
        item.get("raw_rubric_score") is None for item in metadata
    ):
        return
    diagnostics["groups"] += 1
    diagnostics["rollouts"] += len(metadata)
    diagnostics["collapse_rollouts"] += sum(
        bool(item.get("collapse_reward_applied")) for item in metadata
    )
    predicted_zero = bool(metadata[0].get("predicted_zero_variance"))
    explicit_abstain = bool(metadata[0].get("explicit_abstain"))
    diagnostics["predicted_zero_groups"] += int(predicted_zero)
    diagnostics["explicit_abstain_groups"] += int(explicit_abstain)
    if any(item.get("raw_gt_score") is None for item in metadata):
        return

    gt_values = [float(item["raw_gt_score"]) for item in metadata]
    judge_values = [float(item["raw_rubric_score"]) for item in metadata]
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            gt_delta = gt_values[left] - gt_values[right]
            if abs(gt_delta) <= 1e-12:
                continue
            judge_delta = judge_values[left] - judge_values[right]
            diagnostics["gt_pairs"] += 1
            if (gt_delta > 0.0 and judge_delta > 1e-12) or (
                gt_delta < 0.0 and judge_delta < -1e-12
            ):
                diagnostics["gt_correct"] += 1

    actual_variance = max(gt_values) - min(gt_values) > 1e-12
    for prefix, predicts_variance in (
        ("variance", not predicted_zero),
        ("stage1", not explicit_abstain),
    ):
        outcome = (
            "tp"
            if predicts_variance and actual_variance
            else "fp"
            if predicts_variance
            else "fn"
            if actual_variance
            else "tn"
        )
        diagnostics[f"{prefix}_{outcome}"] += 1
