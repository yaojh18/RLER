import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from swe_agent.run.aggregate_swe_agent import (
    AggregateTrajectoryRunner,
    _generate_aggregate_rubrics,
    _score_aggregate_summaries,
    _summarize_aggregate_trajectory,
)
from swe_agent.trajectory_search import (
    JUDGE_RESPONSE_FORMAT,
    PERSISTENT_STATE_RESPONSE_FORMAT,
    RUBRIC_GENERATION_RESPONSE_FORMAT,
    RubricRecord,
    SearchConfig,
)


class _Payload:
    def __init__(self, payload):
        self.payload = copy.deepcopy(payload)

    def model_dump(self, mode="json"):
        return copy.deepcopy(self.payload)


class _FakeSession:
    def __init__(self, snapshot_payload, result_payload, candidate_index):
        self._snapshot_payload = copy.deepcopy(snapshot_payload)
        self._result_payload = copy.deepcopy(result_payload)
        self.agent = SimpleNamespace(env=SimpleNamespace(config=SimpleNamespace(cwd="/repo"), candidate_index=candidate_index))

    def snapshot(self):
        return _Payload(self._snapshot_payload)

    def run_until_pause(self, max_steps=None):
        return _Payload(self._result_payload)


class _FakeBackend:
    def __init__(self, sessions):
        self.agent_config = {"step_limit": 50}
        self._sessions = list(sessions)

    def create_session(self, spec):
        return self._sessions.pop(0)


def _build_snapshot(candidate_id: str, steps: list[tuple[str, str, str]]) -> dict:
    messages = [
        {"role": "system", "content": "You are SWE-agent."},
        {"role": "user", "content": "Fix the failing test."},
    ]
    events = []
    for step_index, (assistant_text, command, observation) in enumerate(steps):
        messages.extend(
            [
                {"role": "assistant", "content": assistant_text, "extra": {"actions": [{"command": command}]}},
                {"role": "user", "content": observation, "extra": {"raw_output": observation}},
            ]
        )
        events.extend(
            [
                {"event_id": f"{candidate_id}:{step_index}:0", "session_id": candidate_id, "step_index": step_index, "kind": "model_response", "payload": {"message": {"content": assistant_text}}},
                {"event_id": f"{candidate_id}:{step_index}:1", "session_id": candidate_id, "step_index": step_index, "kind": "environment_action", "payload": {"actions": [{"command": command}]}},
                {"event_id": f"{candidate_id}:{step_index}:2", "session_id": candidate_id, "step_index": step_index, "kind": "environment_result", "payload": {"messages": [{"content": observation}]}}
            ]
        )
    return {
        "session_id": candidate_id,
        "status": "finished",
        "spec": {"session_id": candidate_id, "task": "Fix the failing test.", "task_id": "demo__demo"},
        "agent": {
            "type_path": "fake.agent",
            "config": {},
            "state": {
                "messages": messages,
                "cost": 0.0,
                "n_calls": len(steps),
            },
        },
        "model": {"type_path": "fake.model", "config": {"model_name": "openai/fake"}, "state": {}},
        "environment": {"type_path": "fake.env", "config": {"image": "base-image", "cwd": "/repo", "executable": "docker"}, "state": {}},
        "metadata": {"events": events, "model_turns": []},
        "last_step_index": len(steps) - 1,
    }


def _build_result(snapshot_payload: dict, submission: str, status: str = "finished") -> dict:
    return {
        "session_id": snapshot_payload["session_id"],
        "status": status,
        "spec": snapshot_payload["spec"],
        "events": [],
        "model_turns": [],
        "final_messages": [],
        "exit_status": "submitted" if submission else "",
        "submission": submission,
        "metadata": {"n_calls": snapshot_payload["agent"]["state"]["n_calls"], "cost": 1.0},
    }


def test_aggregate_runner_selects_highest_scoring_candidate_and_writes_outputs(tmp_path: Path, monkeypatch):
    snapshots = [
        _build_snapshot("candidate-0", [("Inspect target", "sed -n '1,20p' pkg/core.py", "<returncode>0</returncode>\n<output>\ncore\n</output>")]),
        _build_snapshot("candidate-1", [("Run focused test", "pytest tests/test_alpha.py -q", "<returncode>0</returncode>\n<output>\n1 passed\n</output>")]),
        _build_snapshot("candidate-2", [("Inspect README", "sed -n '1,20p' README.md", "<returncode>0</returncode>\n<output>\nreadme\n</output>")]),
        _build_snapshot("candidate-3", [("Run broader test", "pytest tests/test_beta.py -q", "<returncode>0</returncode>\n<output>\n1 passed\n</output>")]),
    ]
    backend = _FakeBackend(
        [
            _FakeSession(snapshots[0], _build_result(snapshots[0], "patch-0"), 0),
            _FakeSession(snapshots[1], _build_result(snapshots[1], "patch-1"), 1),
            _FakeSession(snapshots[2], _build_result(snapshots[2], "patch-2"), 2),
            _FakeSession(snapshots[3], _build_result(snapshots[3], "patch-3"), 3),
        ]
    )

    monkeypatch.setattr(
        "swe_agent.run.aggregate_swe_agent._collect_workspace_meta",
        lambda env: {
            "head_commit": f"commit-{env.candidate_index}",
            "changed_files": [f"pkg/core_{env.candidate_index}.py"],
            "untracked_files": [],
            "diff_stat": f"pkg/core_{env.candidate_index}.py | 1 +",
            "current_patch_chars": 50 + env.candidate_index,
        },
    )

    async def fake_summarize_aggregate_trajectory(**kwargs):
        return {
            "current_state": "Submitted a focused patch after targeted validation.",
            "task_specification": "Fix the failing test for the selected candidate.",
            "files_and_functions": "- `pkg/core.py`: candidate-specific target file",
            "errors_and_corrections": "",
            "codebase_and_system_documentation": "",
            "learnings": "",
            "key_results": kwargs["step_cards"][0]["commands"][0],
            "worklog": f"- steps-{kwargs['trajectory_metadata']['step_count']}",
        }

    async def fake_generate_aggregate_rubrics(**kwargs):
        return [
            RubricRecord(
                rubric_id="validation",
                title="Validation",
                direction="positive",
                description="Runs targeted validation.",
                scale={"1": "none", "2": "weak", "3": "mixed", "4": "good", "5": "strong"},
                weight=1,
                source_round=1,
            )
        ]

    async def fake_score_aggregate_summaries(**kwargs):
        values = [2, 5, 1, 4]
        per_view_scores = []
        for score in values:
            per_view_scores.append(
                [
                    {
                        "rubric_id": "validation",
                        "rubric": {
                            "rubric_id": "validation",
                            "title": "Validation",
                            "direction": "positive",
                            "description": "Runs targeted validation.",
                            "scale": {"1": "none", "2": "weak", "3": "mixed", "4": "good", "5": "strong"},
                            "weight": 1,
                            "source_round": 1,
                        },
                        "score_raw": score,
                        "score_normalized": (score - 1.0) / 4.0,
                        "weighted_score": (score - 1.0) / 4.0,
                        "judge_response": json.dumps({"score": score}),
                    }
                ]
            )
        return per_view_scores, {"validation": 0.25}, {}

    monkeypatch.setattr("swe_agent.run.aggregate_swe_agent._summarize_aggregate_trajectory", fake_summarize_aggregate_trajectory)
    monkeypatch.setattr("swe_agent.run.aggregate_swe_agent._generate_aggregate_rubrics", fake_generate_aggregate_rubrics)
    monkeypatch.setattr("swe_agent.run.aggregate_swe_agent._score_aggregate_summaries", fake_score_aggregate_summaries)
    monkeypatch.setattr("swe_agent.run.aggregate_swe_agent.get_swebench_harness_namespace", lambda instance: "test-namespace")

    runner = AggregateTrajectoryRunner(
        instance={"instance_id": "demo__demo", "problem_statement": "Fix the failing test.", "patch": ""},
        backend=backend,
        run_dir=tmp_path / "aggregate",
        policy_model_name="openai/fake",
        rubric_model_name="openai/fake",
        judge_model_name="openai/fake",
        search_config=SearchConfig(max_active_rubrics=4),
        num_trajectories=4,
        compression_chunk_size=2,
    )
    result = runner.run()

    manifest = json.loads((tmp_path / "aggregate" / "run_manifest.json").read_text())
    patch_payload = json.loads(Path(result.patch_path).read_text())
    candidate_view = json.loads((tmp_path / "aggregate" / "candidates" / "candidate-01" / "view.json").read_text())

    assert manifest["selected_candidate_id"] == "candidate-01"
    assert patch_payload["demo__demo"]["model_patch"] == "patch-1"
    assert candidate_view["trajectory"]["compressed_trajectory"]["key_results"] == "pytest tests/test_alpha.py -q"
    assert Path(result.raw_trajectory_path).exists()
    assert Path(result.slim_trajectory_path).exists()
    assert Path(result.patch_path).exists()
    assert not (tmp_path / "aggregate" / "selected.json").exists()
    assert not (tmp_path / "aggregate" / "run.log").exists()


def test_aggregate_runner_summarizes_step_cards_into_view(tmp_path: Path, monkeypatch):
    steps = [
        ("Step 0", "cmd-0", "<returncode>0</returncode>\n<output>\nout-0\n</output>"),
        ("Step 1", "cmd-1", "<returncode>0</returncode>\n<output>\nout-1\n</output>"),
        ("Step 2", "cmd-2", "<returncode>0</returncode>\n<output>\nout-2\n</output>"),
        ("Step 3", "cmd-3", "<returncode>0</returncode>\n<output>\nout-3\n</output>"),
        ("Step 4", "cmd-4", "<returncode>0</returncode>\n<output>\nout-4\n</output>"),
    ]
    snapshot = _build_snapshot("candidate-0", steps)
    backend = _FakeBackend([_FakeSession(snapshot, _build_result(snapshot, "patch-0"), 0)])
    captured = {}

    monkeypatch.setattr(
        "swe_agent.run.aggregate_swe_agent._collect_workspace_meta",
        lambda env: {
            "head_commit": "commit-0",
            "changed_files": ["pkg/core.py"],
            "untracked_files": [],
            "diff_stat": "pkg/core.py | 1 +",
            "current_patch_chars": 64,
        },
    )

    async def fake_summarize_aggregate_trajectory(**kwargs):
        captured["step_indexes"] = [card["step_index"] for card in kwargs["step_cards"]]
        captured["metadata"] = copy.deepcopy(kwargs["trajectory_metadata"])
        return {
            "current_state": "Summary-ready and awaiting rubric scoring.",
            "task_specification": "",
            "files_and_functions": "",
            "errors_and_corrections": "",
            "codebase_and_system_documentation": "",
            "learnings": "",
            "key_results": "",
            "worklog": "- summary-ready",
        }

    async def fake_generate_aggregate_rubrics(**kwargs):
        return [
            RubricRecord(
                rubric_id="validation",
                title="Validation",
                direction="positive",
                description="Runs targeted validation.",
                scale={"1": "none", "2": "weak", "3": "mixed", "4": "good", "5": "strong"},
                weight=1,
                source_round=1,
            )
        ]

    async def fake_score_aggregate_summaries(**kwargs):
        return (
            [[
                {
                    "rubric_id": "validation",
                    "rubric": {
                        "rubric_id": "validation",
                        "title": "Validation",
                        "direction": "positive",
                        "description": "Runs targeted validation.",
                        "scale": {"1": "none", "2": "weak", "3": "mixed", "4": "good", "5": "strong"},
                        "weight": 1,
                        "source_round": 1,
                    },
                    "score_raw": 5,
                    "score_normalized": 1.0,
                    "weighted_score": 1.0,
                    "judge_response": json.dumps({"score": 5}),
                }
            ]],
            {"validation": 0.0},
            {},
        )

    monkeypatch.setattr("swe_agent.run.aggregate_swe_agent._summarize_aggregate_trajectory", fake_summarize_aggregate_trajectory)
    monkeypatch.setattr("swe_agent.run.aggregate_swe_agent._generate_aggregate_rubrics", fake_generate_aggregate_rubrics)
    monkeypatch.setattr("swe_agent.run.aggregate_swe_agent._score_aggregate_summaries", fake_score_aggregate_summaries)
    monkeypatch.setattr("swe_agent.run.aggregate_swe_agent.get_swebench_harness_namespace", lambda instance: "test-namespace")

    runner = AggregateTrajectoryRunner(
        instance={"instance_id": "demo__demo", "problem_statement": "Fix the failing test.", "patch": ""},
        backend=backend,
        run_dir=tmp_path / "aggregate",
        policy_model_name="openai/fake",
        rubric_model_name="openai/fake",
        judge_model_name="openai/fake",
        search_config=SearchConfig(max_active_rubrics=4),
        num_trajectories=1,
        compression_chunk_size=2,
    )
    runner.run()

    view = json.loads((tmp_path / "aggregate" / "candidates" / "candidate-00" / "view.json").read_text())

    assert captured["step_indexes"] == [0, 1, 2, 3, 4]
    assert captured["metadata"]["step_count"] == 5
    assert "step_cards" not in json.dumps(view)
    assert view["trajectory"]["compressed_trajectory"]["worklog"] == "- summary-ready"


@pytest.mark.asyncio
async def test_summarize_aggregate_trajectory_normalizes_new_state_sections(monkeypatch: pytest.MonkeyPatch):
    async def fake_chat(route_name, model_name, user_prompt=None, system_prompt=None, **kwargs):
        return json.dumps(
            {
                "current_state": "Submitted a patch and stopped after a passing targeted check.",
                "files_and_functions": "- `astropy/utils/misc.py`: contains `InheritDocstrings`.",
                "key_results": "Patch updates the property branch of `InheritDocstrings`.",
                "worklog": "- Reproduced the issue\n- Patched the metaclass\n- Ran focused validation",
            }
        )

    monkeypatch.setattr("swe_agent.run.aggregate_swe_agent.run_chat_with_route_async", fake_chat)

    summary = await _summarize_aggregate_trajectory(
        question={"system_prompt": "You are SWE-agent.", "user_prompt": "Fix the bug."},
        step_cards=[{"step_index": 0, "assistant_message": "inspect", "commands": ["pytest -q"], "observation": "ok"}],
        trajectory_metadata={"candidate_index": 0, "step_count": 1},
        model_name="openai/fake",
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
    )

    assert summary["current_state"] == "Submitted a patch and stopped after a passing targeted check."
    assert summary["files_and_functions"] == "- `astropy/utils/misc.py`: contains `InheritDocstrings`."
    assert summary["key_results"] == "Patch updates the property branch of `InheritDocstrings`."
    assert summary["task_specification"] == ""


@pytest.mark.asyncio
async def test_aggregate_evaluator_calls_use_exact_json_schema(monkeypatch: pytest.MonkeyPatch):
    calls = {"summary": None, "rubric": None, "judge": None}

    async def fake_chat(route_name, model_name, user_prompt=None, system_prompt=None, **kwargs):
        if "## Full Agent Trajectory:" in (user_prompt or ""):
            calls["summary"] = kwargs
            return json.dumps(
                {
                    "current_state": "Submitted a patch and stopped after a focused check.",
                    "task_specification": "Fix the failing test.",
                    "files_and_functions": "- `pkg/core.py`: active target.",
                    "errors_and_corrections": "Avoid unrelated edits.",
                    "codebase_and_system_documentation": "`pkg/core.py` contains the failing logic.",
                    "learnings": "Focused validation is the best signal.",
                    "key_results": "Prepared the final patch.",
                    "worklog": "- Reproduced\n- Patched\n- Validated",
                }
            )
        if "## Trajectory Summaries:" in (user_prompt or ""):
            calls["rubric"] = kwargs
            return json.dumps(
                {
                    "reasoning": "Validation quality differentiates the summaries.",
                    "positive_rubrics": [
                        {
                            "title": "Validation",
                            "description": "Runs targeted validation relevant to the fix.",
                            "scale": {
                                "1": "No validation",
                                "2": "Weak validation",
                                "3": "Some validation",
                                "4": "Targeted validation",
                                "5": "Targeted validation with follow-through",
                            },
                        }
                    ],
                    "negative_rubrics": [],
                }
            )
        calls["judge"] = kwargs
        return json.dumps({"reasoning": "Strong match.", "score": 4})

    monkeypatch.setattr("swe_agent.run.aggregate_swe_agent.run_chat_with_route_async", fake_chat)

    summary = await _summarize_aggregate_trajectory(
        question={"system_prompt": "sys", "user_prompt": "user"},
        step_cards=[{"step_index": 0, "assistant_message": "inspect", "commands": ["pytest -q"], "observation": "ok"}],
        trajectory_metadata={"candidate_index": 0, "step_count": 1},
        model_name="openai/fake",
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
    )
    rubrics = await _generate_aggregate_rubrics(
        question={"system_prompt": "sys", "user_prompt": "user"},
        trajectories_summaries=[summary],
        active_bank=[],
        model_name="openai/fake",
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        round_index=1,
    )
    await _score_aggregate_summaries(
        question={"system_prompt": "sys", "user_prompt": "user"},
        trajectories_summaries=[summary],
        rubrics=rubrics,
        model_name="openai/fake",
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
    )

    assert calls["summary"]["enable_json_schema_validation"] is True
    assert calls["summary"]["response_format"] == PERSISTENT_STATE_RESPONSE_FORMAT
    assert calls["rubric"]["enable_json_schema_validation"] is True
    assert calls["rubric"]["response_format"] == RUBRIC_GENERATION_RESPONSE_FORMAT
    assert calls["judge"]["enable_json_schema_validation"] is True
    assert calls["judge"]["response_format"] == JUDGE_RESPONSE_FORMAT
