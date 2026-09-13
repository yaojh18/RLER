from __future__ import annotations

import json
from pathlib import Path

import pytest

from experience_gen.artifacts import (
    context_from_round,
    contexts_from_search_runs,
    search_run_records,
)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run(tmp_path: Path) -> tuple[str, Path, Path]:
    instance_id = "repo__project-1"
    run_dir = tmp_path / "runs" / "run-001"
    _write(
        run_dir / "nodes/root/snapshot.json",
        {
            "agent": {"state": {"messages": [{"content": "system"}]}},
            "spec": {"instance_id": instance_id, "task": "fix the bug"},
        },
    )
    _write(
        run_dir / "nodes/parent/judge.json",
        {"persistent_state": {"phase": "edit"}, "recent_segments": []},
    )
    for node_id, reward, changed in (
        ("a", 0.0, ["bad.py"]),
        ("b", 2.0, ["good.py"]),
    ):
        _write(
            run_dir / f"nodes/{node_id}/node.json",
            {"parent_id": "parent", "status": "finished", "exit_status": "Submitted"},
        )
        _write(
            run_dir / f"nodes/{node_id}/judge.json",
            {
                "ground_truth_reward": reward,
                "recent_segments": [{"step_cards": [{"node": node_id}]}],
                "workspace_meta": {"changed_files": changed},
            },
        )
    sample = {
        "selected": True,
        "average_rubric_judged_scores": {"a": 0.9, "b": 0.1},
        "gt_by_rubric": {
            "rubric": {"ground_truth_by_node": {"a": 0.0, "b": 2.0}}
        },
        "generated_rubrics": [{"rubric": "visible evidence"}],
        "retrieved": [{"experience_id": "exp-1"}],
    }
    round_path = run_dir / "rubrics/round_001.json"
    _write(
        round_path,
        {
            "round_index": 1,
            "parent_id": "parent",
            "node_scores": [
                {"node_id": "a", "siblings_score": 0.9},
                {"node_id": "b", "siblings_score": 0.1},
            ],
            "average_rubric_judged_scores": {"a": 0.9, "b": 0.1},
            "rubric_samples": [sample],
            "pc_rubric_samples": [{**sample, "selected": True}],
            "async_terminal_eval": {"selected_node_id": "a"},
        },
    )
    return instance_id, run_dir, round_path


def test_calculate_gt_round_becomes_visible_context(tmp_path: Path) -> None:
    instance_id, run_dir, round_path = _run(tmp_path)
    context = context_from_round(instance_id, run_dir, round_path)

    assert context is not None
    assert context["selection_success"] is False
    assert context["selected_node_id"] == "a"
    assert context["node_ids"] == ["a", "b"]
    assert context["view"]["question"]["user_prompt"] == "fix the bug"
    assert context["view"]["continuations"][1]["summary"]["changed_files"] == [
        "good.py"
    ]
    assert context["generated_rubrics"] == [{"rubric": "visible evidence"}]
    assert set(context["scope_metrics"]) == {"siblings", "pc"}


def test_plan_and_search_root_discovery_are_content_equivalent(tmp_path: Path) -> None:
    instance_id, run_dir, _ = _run(tmp_path)
    plan = tmp_path / "plan.json"
    _write(
        plan,
        {"completed": [{"instance_id": instance_id, "run_dir": str(run_dir)}]},
    )
    by_plan = search_run_records(plan=plan)
    by_root = search_run_records(search_root=tmp_path / "runs")
    assert by_plan == by_root == [(instance_id, run_dir)]
    assert len(contexts_from_search_runs(by_plan)) == 1


def test_calculate_gt_adapter_rejects_incomplete_gt(tmp_path: Path) -> None:
    instance_id, run_dir, round_path = _run(tmp_path)
    payload = json.loads(round_path.read_text(encoding="utf-8"))
    payload["rubric_samples"][0]["gt_by_rubric"]["rubric"][
        "ground_truth_by_node"
    ].pop("b")
    payload["pc_rubric_samples"][0]["gt_by_rubric"]["rubric"][
        "ground_truth_by_node"
    ].pop("b")
    _write(round_path, payload)
    with pytest.raises(ValueError, match="incomplete calculate-GT node coverage"):
        context_from_round(instance_id, run_dir, round_path)
