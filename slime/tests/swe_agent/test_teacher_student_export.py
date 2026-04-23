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
            "instance_id": "psf__requests-1142",
            "run_id": "run-1",
            "system_prompt": "system",
            "user_prompt": "task",
            "current_round": 1,
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
            "checkpoint_image_tag": "base",
            "checkpoint_image_id": "base-id",
            "workspace_fingerprint": None,
            "raw_traj_path": str(run_dir / "nodes" / "root" / "raw_traj.json"),
            "messages_path": str(run_dir / "nodes" / "root" / "messages.json"),
            "judge_path": str(run_dir / "nodes" / "root" / "judge.json"),
            "snapshot_path": None,
            "rubric_ref": None,
            "score": 0.0,
            "step_start": 0,
            "step_end": -1,
            "submission": "",
            "exit_status": "",
            "regressed_vs_parent": False,
            "policy_source": "teacher",
            "policy_model_name": "gemini/gemini-3.1-pro-preview",
        },
    )
    _write_json(run_dir / "nodes" / "root" / "messages.json", {"messages": []})
    _write_json(run_dir / "nodes" / "root" / "judge.json", {"persistent_state": {}, "recent_segments": [], "workspace_meta": {}})

    members = []
    for index, (source, score, gt) in enumerate(
        [("teacher", 1.0, 1.0), ("teacher", 0.8, 1.0), ("student", -0.2, 0.0), ("student", -0.3, 0.0)]
    ):
        node_id = f"node-{index}"
        _write_json(
            run_dir / "nodes" / node_id / "node.json",
            {
                "node_id": node_id,
                "parent_id": "root",
                "round_index": 1,
                "depth": 1,
                "session_id": node_id,
                "status": "frontier",
                "checkpoint_image_tag": None,
                "checkpoint_image_id": None,
                "workspace_fingerprint": None,
                "raw_traj_path": str(run_dir / "nodes" / node_id / "raw_traj.json"),
                "messages_path": str(run_dir / "nodes" / node_id / "messages.json"),
                "judge_path": str(run_dir / "nodes" / node_id / "judge.json"),
                "snapshot_path": None,
                "rubric_ref": str(run_dir / "rubrics" / "round_001.json"),
                "score": score,
                "step_start": 0,
                "step_end": 1,
                "submission": "patch" if gt else "",
                "exit_status": "submitted",
                "regressed_vs_parent": False,
                "policy_source": source,
                "policy_model_name": "gemini/gemini-3.1-pro-preview" if source == "teacher" else "Qwen/Qwen3.5-9B",
            },
        )
        _write_json(
            run_dir / "nodes" / node_id / "messages.json",
            {
                "messages": [
                    {"role": "assistant", "message": f"<think>{source}</think>\nthought {index}", "tool_calls": [], "parser_model": "model", "index": 0},
                    {"role": "user", "message": "obs", "tool_calls": [], "parser_model": "model", "index": 1},
                    {"role": "assistant", "message": f"answer {index}", "tool_calls": [], "parser_model": "model", "index": 2},
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
        members.append({"node_id": node_id, "score": score, "ground_truth_reward": gt})

    rubric_dir = run_dir / "rubrics" / "rubric-r001-s00-main"
    _write_json(
        rubric_dir / "messages.json",
        {
            "sample_index": 0,
            "messages": [
                {"role": "user", "content": "Generate rubric"},
                {"role": "assistant", "content": "<think>r0</think>\n{\"rubric\": {}}", "content_no_thinking": "{\"rubric\": {}}"},
                {"role": "user", "content": "Generate the next best rubric or return an empty object."},
                {"role": "assistant", "content": "<think>r1</think>\n{}", "content_no_thinking": "{}"},
            ],
        },
    )
    _write_json(
        run_dir / "rubrics" / "round_001.json",
        {
            "round_index": 1,
            "parent_id": "root",
            "active_bank_before": [],
            "active_bank_after": [],
            "inactive_bank_after": [],
            "selected_sample_index": 0,
            "node_scores": [{"node_id": member["node_id"], "score": member["score"], "finished": False, "judge_error": None} for member in members],
            "rubric_samples": [
                {
                    "sample_index": 0,
                    "artifact_tag": "rubric-r001-s00-main",
                    "artifact_dir": str(rubric_dir),
                    "generated": [{"rubric_id": "r0", "title": "Rubric 0"}],
                    "generated_titles": ["Rubric 0"],
                    "child_rewards": {member["node_id"]: member["score"] for member in members},
                    "teacher_student_gap": 1.2,
                    "turn_rewards": [0.5, 0.3],
                    "list_reward": 0.7,
                    "selected": True,
                    "error": None,
                }
            ],
        },
    )
    return run_dir


def test_same_harness_run_emits_node_records_and_round_groups(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    sft_bundle = SFTDataExporter(run_dir=run_dir).export_bundle()
    rl_bundle = GRPODataExporter(run_dir=run_dir).export_bundle()

    assert sft_bundle.accepted_group_ids == ["psf__requests-1142:round:1:parent:root"]
    assert len(sft_bundle.policy_samples) == 1
    assert len(sft_bundle.rubric_samples) == 1
    assert len(rl_bundle.policy_groups) == 1
    assert len(rl_bundle.rubric_groups) == 1
    assert len(rl_bundle.policy_groups[0].samples) == 4
    assert len(rl_bundle.rubric_groups[0].samples) == 1
    assert rl_bundle.rubric_groups[0].samples[0].reward > 0.9
