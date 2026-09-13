"""Normalize calculate-GT trajectory-search runs into TTS contexts."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Iterable

from .io import load_json
from .metrics import ranking_metrics


SCOPES = ("siblings", "pc")


def selected_sample(round_payload: dict[str, Any], scope: str) -> dict[str, Any] | None:
    key = "rubric_samples" if scope == "siblings" else "pc_rubric_samples"
    samples = list(round_payload.get(key) or [])
    selected = [row for row in samples if row.get("selected")]
    if len(selected) > 1:
        raise ValueError(f"multiple selected {scope} samples")
    if selected:
        return selected[0]
    if scope == "siblings" and samples:
        selected_index = round_payload.get("selected_sample_index")
        matches = [
            row for row in samples if row.get("sample_index") == selected_index
        ]
        if len(matches) > 1:
            raise ValueError("selected_sample_index matches multiple sibling samples")
        if matches:
            return matches[0]
    return samples[0] if len(samples) == 1 else None


def ground_truth_rewards(sample: dict[str, Any], run_dir: Path) -> dict[str, float]:
    landed = []
    for payload in (sample.get("gt_by_rubric") or {}).values():
        values = payload.get("ground_truth_by_node") or {}
        if values:
            landed.append(
                {str(key): float(value) for key, value in values.items()}
            )
    if landed:
        if any(values != landed[0] for values in landed[1:]):
            raise ValueError("calculate-GT rewards disagree across rubrics")
        return landed[0]
    output = {}
    for node_id in (sample.get("average_rubric_judged_scores") or {}):
        judge_path = run_dir / "nodes" / str(node_id) / "judge.json"
        if judge_path.is_file():
            value = load_json(judge_path).get("ground_truth_reward")
            if value is not None:
                output[str(node_id)] = float(value)
    return output


def _continuation_view(
    run_dir: Path,
    node_id: str,
    parent_index: int | None,
) -> dict[str, Any]:
    node = load_json(run_dir / "nodes" / node_id / "node.json")
    judge = load_json(run_dir / "nodes" / node_id / "judge.json")
    recent = judge.get("recent_segments") or []
    if not recent:
        raise ValueError(f"{run_dir}:{node_id} has no recent trajectory segment")
    workspace = judge.get("workspace_meta") or {}
    cards = recent[-1].get("step_cards") or []
    output = {
        "node_id": node_id,
        "summary": {
            "step_count": len(cards),
            "changed_files": list(workspace.get("changed_files") or [])[:8],
            "untracked_files": list(workspace.get("untracked_files") or [])[:8],
            "diff_stat": workspace.get("diff_stat", ""),
            "current_patch_chars": int(workspace.get("current_patch_chars") or 0),
            "result_status": (
                "finished"
                if node.get("status") == "finished"
                or node.get("exit_status") == "Submitted"
                else "paused"
            ),
            "exit_status": node.get("exit_status", ""),
        },
        "trajectory_continuation": copy.deepcopy(recent[-1]),
    }
    if parent_index is not None:
        output["parent_index"] = parent_index
    return output


def _parent_view(
    run_dir: Path,
    parent_ids: list[str],
    continuations: list[dict[str, Any]],
) -> tuple[dict[str, Any], Any]:
    parents = []
    for index, parent_id in enumerate(parent_ids, start=1):
        judge = load_json(run_dir / "nodes" / parent_id / "judge.json")
        segments = judge.get("recent_segments") or []
        positions = [
            position
            for position, continuation in enumerate(continuations, start=1)
            if continuation.get("parent_index", 1) == index
        ]
        parents.append(
            {
                "parent_index": index,
                "continuations": (
                    f"{positions[0]}-{positions[-1]}" if positions else "<none>"
                ),
                "state": copy.deepcopy(judge.get("persistent_state") or {}),
                "trajectory": copy.deepcopy(segments[-1]) if segments else None,
            }
        )
    if len(parents) == 1:
        return parents[0]["state"], parents[0]["trajectory"]
    return (
        {
            "parent_states": [
                {
                    "parent_index": row["parent_index"],
                    "continuations": row["continuations"],
                    "state": row["state"],
                }
                for row in parents
            ]
        },
        {
            "parent_views": [
                {
                    "parent_index": row["parent_index"],
                    "continuations": row["continuations"],
                    "trajectory": row["trajectory"],
                }
                for row in parents
            ]
        },
    )


def context_from_round(
    instance_id: str,
    run_dir: Path,
    round_path: Path,
) -> dict[str, Any] | None:
    payload = load_json(round_path)
    round_index = int(payload.get("round_index") or round_path.stem.split("_")[-1])
    sibling = selected_sample(payload, "siblings")
    if sibling is None:
        return None
    rewards = ground_truth_rewards(sibling, run_dir)
    node_ids = [str(row["node_id"]) for row in payload.get("node_scores") or []]
    if len(node_ids) < 2 or set(node_ids) != set(rewards):
        raise ValueError(
            f"incomplete calculate-GT node coverage: {instance_id}:R{round_index}"
        )
    if node_ids != list(rewards):
        raise ValueError(
            f"calculate-GT node ordering drift: {instance_id}:R{round_index}"
        )
    parent_ids = list(map(str, payload.get("parent_ids") or [payload["parent_id"]]))
    parent_index = {
        node_id: index for index, node_id in enumerate(parent_ids, start=1)
    }
    continuations = []
    for node_id in node_ids:
        node = load_json(run_dir / "nodes" / node_id / "node.json")
        if str(node["parent_id"]) not in parent_index:
            raise ValueError(f"unknown parent for {instance_id}:R{round_index}/{node_id}")
        continuations.append(
            _continuation_view(
                run_dir,
                node_id,
                parent_index[str(node["parent_id"])] if len(parent_ids) > 1 else None,
            )
        )
    previous_state, parent_trajectory = _parent_view(
        run_dir,
        parent_ids,
        continuations,
    )
    root = load_json(run_dir / "nodes/root/snapshot.json")
    messages = root["agent"]["state"]["messages"]
    question = {
        "system_prompt": messages[0].get("content", ""),
        "user_prompt": root["spec"].get("task")
        or root["spec"].get("raw_user_query", ""),
    }
    sibling_scores = {
        str(key): float(value)
        for key, value in (sibling.get("average_rubric_judged_scores") or {}).items()
    }
    if set(sibling_scores) != set(node_ids):
        raise ValueError(f"incomplete sibling judge scores: {instance_id}:R{round_index}")
    judge_scores = {
        str(key): float(value)
        for key, value in (
            payload.get("average_rubric_judged_scores") or sibling_scores
        ).items()
    }
    if set(judge_scores) != set(node_ids):
        raise ValueError(f"incomplete round judge scores: {instance_id}:R{round_index}")
    selected_node_id = str(
        (payload.get("async_terminal_eval") or {}).get("selected_node_id") or ""
    )
    selection_source = "async_terminal_eval"
    if not selected_node_id:
        selected_node_id = str(
            max(
                payload["node_scores"],
                key=lambda row: (
                    float(row.get("siblings_score", row.get("score", 0.0))),
                    str(row.get("node_id") or ""),
                ),
            )["node_id"]
        )
        selection_source = "max_siblings_score_fallback"
    if selected_node_id not in rewards:
        raise ValueError(f"selected node lacks GT: {instance_id}:R{round_index}")

    scope_metrics = {}
    retrieval_events = []
    for scope in SCOPES:
        sample = selected_sample(payload, scope)
        if sample is None:
            continue
        scores = {
            str(key): float(value)
            for key, value in (
                sample.get("average_rubric_judged_scores") or {}
            ).items()
        }
        scope_rewards = ground_truth_rewards(sample, run_dir)
        if set(scores) != set(node_ids) or any(
            not math.isclose(scope_rewards.get(node_id, math.nan), rewards[node_id])
            for node_id in node_ids
        ):
            raise ValueError(
                f"{scope} sample is not aligned to the sibling group: "
                f"{instance_id}:R{round_index}"
            )
        scope_metrics[scope] = ranking_metrics(scores, rewards, node_ids)
        retrieval_events.append(
            {
                "scope": scope,
                "judge_scores": scores,
                "generated_rubrics": copy.deepcopy(
                    sample.get("generated_rubrics")
                    or sample.get("rubrics")
                    or []
                ),
                "retrieved_experience_ids": [
                    str(row["experience_id"])
                    for row in sample.get("retrieved") or []
                    if row.get("experience_id")
                ],
            }
        )
    context = {
        "context_id": f"{instance_id}:R{round_index}",
        "instance_id": instance_id,
        "round_index": round_index,
        "run_dir": str(run_dir),
        "round_artifact": str(round_path),
        "node_ids": node_ids,
        "parent_ids": parent_ids,
        "judge_scores": judge_scores,
        "generated_rubrics": copy.deepcopy(
            sibling.get("generated_rubrics")
            or sibling.get("rubrics")
            or []
        ),
        "gt_rewards": rewards,
        "selected_node_id": selected_node_id,
        "selection_source": selection_source,
        "selection_success": math.isclose(rewards[selected_node_id], 2.0),
        "scope_metrics": scope_metrics,
        "retrieval_events": retrieval_events,
        "view": {
            "question": question,
            "previous_state": previous_state,
            "parent_trajectory": parent_trajectory,
            "continuations": continuations,
        },
    }
    return context


def _remap_workspace_path(path: Path, workspace_root: Path | None) -> Path:
    if workspace_root is not None and str(path).startswith("/workspace/"):
        return workspace_root / str(path).removeprefix("/workspace/")
    return path


def _instance_id_from_run(run_dir: Path) -> str:
    snapshot = load_json(run_dir / "nodes/root/snapshot.json")
    spec = snapshot.get("spec") or {}
    for key in ("instance_id", "task_id", "id"):
        value = str(spec.get(key) or "").strip()
        if value:
            return value
    raise ValueError(
        f"cannot infer instance ID from {run_dir}; use a canonical plan whose "
        "completed records contain instance_id"
    )


def search_run_records(
    *,
    plan: Path | None = None,
    search_root: Path | None = None,
    workspace_root: Path | None = None,
) -> list[tuple[str, Path]]:
    if (plan is None) == (search_root is None):
        raise ValueError("provide exactly one of plan or search_root")
    if plan is not None:
        completed = load_json(plan).get("completed") or []
        records = [
            (
                str(row["instance_id"]),
                _remap_workspace_path(Path(str(row["run_dir"])), workspace_root),
            )
            for row in completed
        ]
    else:
        assert search_root is not None
        run_dirs = {
            path.parent.parent
            for path in search_root.rglob("rubrics/round_*.json")
        }
        records = [(_instance_id_from_run(path), path) for path in sorted(run_dirs)]
    if not records or len(records) != len(set(records)):
        raise ValueError("search runs are empty or duplicated")
    for _, run_dir in records:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
    return sorted(records)


def contexts_from_search_runs(
    records: Iterable[tuple[str, Path]],
) -> list[dict[str, Any]]:
    output = []
    for instance_id, run_dir in records:
        round_paths = sorted((run_dir / "rubrics").glob("round_*.json"))
        if not round_paths:
            raise ValueError(f"search run has no round artifacts: {run_dir}")
        output.extend(
            context
            for context in (
                context_from_round(instance_id, run_dir, round_path)
                for round_path in round_paths
            )
            if context is not None
        )
    ids = [row["context_id"] for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate normalized context IDs")
    return sorted(
        output,
        key=lambda row: (
            row["instance_id"],
            row["round_index"],
            row["context_id"],
        ),
    )
