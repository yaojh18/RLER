from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Iterable

from .io import load_json, load_jsonl


REQUIRED_CONTEXT_FIELDS = {
    "context_id",
    "instance_id",
    "round_index",
    "node_ids",
    "gt_rewards",
    "view",
}


def strict_reward_variance(context: dict[str, Any], *, epsilon: float = 1e-9) -> bool:
    """Return the final experiment's strict reward==2/non-2 variance gate."""

    node_ids = [str(value) for value in context.get("node_ids") or []]
    rewards = context.get("gt_rewards") or {}
    values = [float(rewards[node_id]) for node_id in node_ids if node_id in rewards]
    return (
        len(values) >= 2
        and any(math.isclose(value, 2.0, abs_tol=epsilon) for value in values)
        and any(not math.isclose(value, 2.0, abs_tol=epsilon) for value in values)
    )


def validate_context(context: dict[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_CONTEXT_FIELDS - set(context)
    if missing:
        raise ValueError(
            f"Context {context.get('context_id', '<unknown>')} is missing {sorted(missing)}"
        )
    if not isinstance(context["view"], dict):
        raise ValueError(f"{context['context_id']}: view must be an object")
    if not isinstance(context["gt_rewards"], dict):
        raise ValueError(f"{context['context_id']}: gt_rewards must be an object")
    node_ids = [str(value) for value in context["node_ids"]]
    if len(node_ids) < 2 or len(set(node_ids)) != len(node_ids):
        raise ValueError(f"{context['context_id']}: node_ids must be unique and contain >=2 nodes")
    if set(node_ids) - set(context["gt_rewards"]):
        raise ValueError(f"{context['context_id']}: gt_rewards do not cover every node")
    continuations = context["view"].get("continuations") or []
    continuation_ids = {str(row.get("node_id")) for row in continuations if isinstance(row, dict)}
    if set(node_ids) - continuation_ids:
        raise ValueError(f"{context['context_id']}: visible continuations do not cover every node")
    normalized = copy.deepcopy(context)
    normalized["node_ids"] = node_ids
    normalized["round_index"] = int(normalized["round_index"])
    normalized["has_variance"] = strict_reward_variance(normalized)
    return normalized


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("Context artifact must be a JSON list or object")
    for key in ("contexts", "rows", "items"):
        if isinstance(payload.get(key), list):
            return payload[key]
    if REQUIRED_CONTEXT_FIELDS <= set(payload):
        return [payload]
    raise ValueError("Could not find contexts/rows/items in context artifact")


def load_contexts(
    paths: str | Path | Iterable[str | Path],
    *,
    require_variance: bool = True,
) -> list[dict[str, Any]]:
    """Load normalized landed contexts produced from trajectory artifacts."""

    if isinstance(paths, (str, Path)):
        paths = [paths]
    contexts: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        payload = load_jsonl(path) if path.suffix == ".jsonl" else load_json(path)
        for raw in _rows(payload):
            context = validate_context(raw)
            if require_variance and not context["has_variance"]:
                continue
            context_id = str(context["context_id"])
            existing = contexts.get(context_id)
            if existing is not None and existing != context:
                raise ValueError(f"Conflicting duplicate context: {context_id}")
            contexts[context_id] = context
    return [contexts[key] for key in sorted(contexts)]


def full_variance_population(
    contexts: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return every landed group with strict reward==2/non-2 variance."""

    normalized = [validate_context(row) for row in contexts]
    selected = [
        row
        for row in normalized
        if strict_reward_variance(row)
    ]
    return sorted(
        selected,
        key=lambda row: (
            str(row["instance_id"]),
            int(row["round_index"]),
            str(row["context_id"]),
        ),
    )


def historical_selection_error(context: dict[str, Any]) -> bool:
    """Return whether the historically selected branch failed terminal GT.

    Calculate-GT artifacts record the actual asynchronous selection.  Older
    normalized contexts may omit ``selection_success`` but still retain the
    selected node and rewards; that case is derived without re-running a
    judge.  Missing selection evidence is rejected rather than silently
    treating every variance group as a generation target.
    """

    normalized = validate_context(context)
    selected_node_id = str(normalized.get("selected_node_id") or "")
    if not selected_node_id and "selection_success" not in normalized:
        raise ValueError(
            f"{normalized['context_id']}: missing historical selection evidence"
        )
    if not selected_node_id:
        return not bool(normalized["selection_success"])
    rewards = normalized["gt_rewards"]
    if selected_node_id not in rewards:
        raise ValueError(
            f"{normalized['context_id']}: selected node lacks terminal GT"
        )
    derived_success = math.isclose(
        float(rewards[selected_node_id]), 2.0, abs_tol=1e-9
    )
    if (
        "selection_success" in normalized
        and bool(normalized["selection_success"]) != derived_success
    ):
        raise ValueError(
            f"{normalized['context_id']}: inconsistent historical selection outcome"
        )
    return not derived_success


def context_scopes(context: dict[str, Any]) -> list[str]:
    """Return only scopes for which the landed round has baseline evidence."""

    metrics = context.get("scope_metrics") or {}
    events = {
        str(row.get("scope"))
        for row in context.get("retrieval_events") or []
        if isinstance(row, dict)
    }
    scopes = []
    if "siblings" in metrics or context.get("generated_rubrics") or context.get("judge_scores"):
        scopes.append("siblings")
    if "pc" in metrics or "pc" in events:
        scopes.append("pc")
    return scopes
