from __future__ import annotations

import json
from pathlib import Path

from slime.swe_agent.data_export import GRPODataExporter, SFTDataExporter


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "run_manifest.json",
        {
            "instance_id": "pallets__click-2380",
            "run_id": "run-1",
            "system_prompt": "system",
            "user_prompt": "task",
            "current_round": 1,
            "policy_model_name": "openai/gpt-5.4",
            "student_policy_model_name": "Qwen/Qwen3.5-9B",
        },
    )
    _write_json(
        run_dir / "nodes" / "root" / "node.json",
        {
            "node_id": "root",
            "parent_id": None,
            "round_index": 0,
            "depth": 0,
            "session_id": "root",
            "status": "frontier",
            "step_start": 0,
            "step_end": -1,
            "submission": "",
            "exit_status": "",
            "policy_source": "teacher",
            "policy_model_name": "openai/gpt-5.4",
        },
    )
    _write_json(run_dir / "nodes" / "root" / "messages.json", {"messages": []})
    _write_json(run_dir / "nodes" / "root" / "judge.json", {"persistent_state": {}, "recent_segments": [], "workspace_meta": {}})

    node_rows = []
    node_scores = []
    for index, (source, score, gt) in enumerate(
        [("teacher", 0.9, 1.0), ("teacher", 0.6, 1.0), ("student", 0.1, 0.0), ("student", -0.2, 0.0)]
    ):
        node_id = f"node-{index}"
        node_payload = {
            "node_id": node_id,
            "parent_id": "root",
            "round_index": 1,
            "depth": 1,
            "session_id": node_id,
            "status": "finished",
            "step_start": 0,
            "step_end": 1,
            "submission": "patch" if gt else "",
            "exit_status": "submitted",
            "policy_source": source,
            "policy_model_name": "openai/gpt-5.4" if source == "teacher" else "Qwen/Qwen3.5-9B",
        }
        node_rows.append(node_payload)
        _write_json(run_dir / "nodes" / node_id / "node.json", node_payload)
        _write_json(
            run_dir / "nodes" / node_id / "messages.json",
            {
                "messages": [
                    {"role": "assistant", "message": f"<think>{source}</think>\nplan {index}"},
                    {"role": "user", "message": f"obs {index}"},
                    {"role": "assistant", "message": f"apply patch {index}"},
                ]
            },
        )
        _write_json(
            run_dir / "nodes" / node_id / "judge.json",
            {
                "overall_reward": score,
                "ground_truth_reward": gt,
                "recent_segments": [],
                "workspace_meta": {},
            },
        )
        node_scores.append({"node_id": node_id, "score": score})

    (run_dir / "node_index.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [{"node_id": "root", "parent_id": None, "policy_source": "teacher"}, *node_rows]),
        encoding="utf-8",
    )

    rubric_list_id = "rubric-r001-s00"
    _write_json(
        run_dir / "rubrics" / rubric_list_id / "messages.json",
        {
            "rubric_list_id": rubric_list_id,
            "messages": [
                {"role": "user", "content": "Generate the first rubric"},
                {"role": "assistant", "content": "<think>r0</think>\n{\"rubric\": {\"title\": \"A\"}}"},
                {"role": "user", "content": "Generate the next best rubric or return an empty object."},
                {"role": "assistant", "content": "<think>r1</think>\n{\"rubric\": {\"title\": \"B\"}}"},
            ],
        },
    )
    _write_json(
        run_dir / "rubrics" / rubric_list_id / "rubric.json",
        {
            "rubric_list_id": rubric_list_id,
            "parent_node_id": "root",
            "generated": [
                {"rubric_id": "r0", "title": "A"},
                {"rubric_id": "r1", "title": "B"},
            ],
            "active_bank_before": [],
            "active_bank_after": [],
            "inactive_bank_after": [],
            "variance_by_rubric": {"r0": 0.3, "r1": 0.2},
            "redundency_by_rubric": {"r0": 0.1, "r1": -0.1},
            "reward_by_rubric": {"r0": 0.4, "r1": 0.1},
            "child_score_by_rubric": {"r0": {"node-0": 1.0}, "r1": {"node-1": 0.5}},
            "parent_score_by_rubric": {"r0": 0.2, "r1": 0.1},
            "child_rewards": {"node-0": 0.9, "node-1": 0.6, "node-2": 0.1, "node-3": -0.2},
            "parent_reward": 0.5,
            "judge_errors": [],
            "generated_titles": ["A", "B"],
            "selected": True,
            "gt_by_rubric": {"rubric-r001-s00": {"node-0": {"score": 0.9, "gt_score": 1.0, "is_parent": False}}},
            "gt_reward_siblings": 0.8,
            "gt_reward_parent": 0.4,
        },
    )
    _write_json(
        run_dir / "rubrics" / "round_001.json",
        {
            "round_index": 1,
            "parent_id": "root",
            "selected_sample_index": 0,
            "selected_rubric_list_id": rubric_list_id,
            "generated": [{"rubric_id": "r0", "title": "A"}, {"rubric_id": "r1", "title": "B"}],
            "active_bank_before": [],
            "active_bank_after": [],
            "inactive_bank_after": [],
            "parent_reward": 0.5,
            "child_rewards": {"node-0": 0.9, "node-1": 0.6, "node-2": 0.1, "node-3": -0.2},
            "parent_score_by_rubric": {"r0": 0.2, "r1": 0.1},
            "child_score_by_rubric": {"r0": {"node-0": 1.0}, "r1": {"node-1": 0.5}},
            "variance_by_rubric": {"r0": 0.3, "r1": 0.2},
            "redundency_by_rubric": {"r0": 0.1, "r1": -0.1},
            "reward_by_rubric": {"r0": 0.4, "r1": 0.1},
            "gt_by_rubric": {},
            "gt_reward_siblings": 0.8,
            "gt_reward_parent": 0.4,
            "baseline_parent_reward": 0.0,
            "node_scores": node_scores,
            "judge_errors": [],
            "policy_generation_errors": [],
            "rubric_generation_errors": [],
            "rubric_samples": [{"rubric_list_id": rubric_list_id, "selected": True}],
        },
    )
    return run_dir


def test_same_harness_run_emits_node_records_and_round_groups(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    sft_bundle = SFTDataExporter(run_dir=run_dir).export_bundle()
    rl_bundle = GRPODataExporter(run_dir=run_dir).export_bundle()

    assert sft_bundle.accepted_group_ids == ["pallets__click-2380:round:1:parent:root"]
    assert len(sft_bundle.policy_samples) == 4
    assert len(sft_bundle.rubric_samples) == 2
    assert sft_bundle.policy_samples[0].prompt[0]["content"] == "system"
    assert sft_bundle.policy_samples[0].turns[0]["content"].startswith("<think>")
    assert sft_bundle.policy_samples[1].prompt[-2]["content"].startswith("<think>")
    assert sft_bundle.policy_samples[1].turns[0]["content"] == "apply patch 0"
    assert len(rl_bundle.policy_groups) == 1
    assert len(rl_bundle.rubric_groups) == 1
    assert len(rl_bundle.policy_groups[0].samples) == 4
    assert len(rl_bundle.rubric_groups[0].samples) == 1
    assert rl_bundle.rubric_groups[0].samples[0].metadata["turn_rewards"] == [0.4, 0.1]
    assert abs(rl_bundle.rubric_groups[0].samples[0].reward - 0.6) < 1e-9
