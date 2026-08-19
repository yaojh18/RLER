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
    "policy_base_url",
    "policy_base_urls",
    "rubric_base_url",
    "usage_ledger",
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
        groups.append(buffered_group_class(**kwargs))
    return groups


def serialize_pending_tasks(
    pending: dict[Any, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in pending.values():
        row = {
            key: copy.deepcopy(value)
            for key, value in task.items()
            if key not in _RUNTIME_TASK_KEYS
        }
        rows.append(row)
    return rows


def _serialize_attempt_error(
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


def _deserialize_attempt_error(payload, error_class, limit_field: str):
    if not payload:
        return None
    attempted = int(payload.get("attempted_instances", -1))
    limit = int(payload.get(limit_field, -1))
    return error_class(
        attempted_instances=None if attempted < 0 else attempted,
        **{limit_field: None if limit < 0 else limit},
    )


def serialize_budget_error(error: Any) -> dict[str, int] | None:
    return _serialize_attempt_error(error, "budget")


def deserialize_budget_error(payload, error_class):
    return _deserialize_attempt_error(payload, error_class, "budget")


def serialize_validation_error(error: Any) -> dict[str, int] | None:
    return _serialize_attempt_error(error, "boundary")


def deserialize_validation_error(payload, error_class):
    return _deserialize_attempt_error(payload, error_class, "boundary")
