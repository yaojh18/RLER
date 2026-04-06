import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from swe_agent.trajectory_search import (
    EMPTY_PERSISTENT_STATE,
    EMPTY_WORKSPACE_META,
    MAX_OBSERVATION_CHARS,
    OBSERVATION_TRUNCATION_MARKER,
    PERSISTENT_STATE_UPDATE_PROMPT,
    SearchNode,
    SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT,
    SWE_TRAJECTORY_RUBRIC_JUDGE_PROMPT,
    SearchConfig,
    TrajectorySearchRunner,
    _build_initial_rubric_bank,
    _build_step_cards,
    _collect_workspace_meta,
    _compute_weighted_reward,
    _generate_round_rubrics,
    _score_parent_round,
    _score_round,
    _truncate_structured_observation,
    _update_persistent_state,
    _update_rubric_bank,
    RubricRecord,
)


class _Payload:
    def __init__(self, payload):
        self.payload = copy.deepcopy(payload)

    def model_dump(self, mode="json"):
        return copy.deepcopy(self.payload)


class _FakeSession:
    def __init__(self, snapshot_payload, result_payload=None, container_id="cid"):
        self._snapshot_payload = copy.deepcopy(snapshot_payload)
        self._result_payload = copy.deepcopy(
            result_payload
            or {
                "session_id": snapshot_payload["session_id"],
                "status": "paused",
                "spec": snapshot_payload["spec"],
                "events": [],
                "model_turns": [],
                "final_messages": [],
                "exit_status": "",
                "submission": "",
                "metadata": {"n_calls": 1, "cost": 1.0},
            }
        )
        self.agent = SimpleNamespace(env=SimpleNamespace(container_id=container_id, config=SimpleNamespace(cwd="/repo")))

    def snapshot(self):
        return _Payload(self._snapshot_payload)

    def run_until_pause(self, max_steps=None):
        return _Payload(self._result_payload)


class _SequentialSession:
    def __init__(self, snapshots, results, container_id="cid"):
        self._snapshots = [copy.deepcopy(snapshot) for snapshot in snapshots]
        self._results = [copy.deepcopy(result) for result in results]
        self._index = 0
        self._current_snapshot = copy.deepcopy(self._snapshots[0])
        self.agent = SimpleNamespace(
            env=SimpleNamespace(container_id=container_id, config=SimpleNamespace(cwd="/repo")),
            model=SimpleNamespace(config=SimpleNamespace(model_kwargs={"temperature": 1.0, "top_p": 0.95})),
        )

    def snapshot(self):
        return _Payload(self._current_snapshot)

    def run_until_pause(self, max_steps=None):
        payload = copy.deepcopy(self._results[self._index])
        self._current_snapshot = copy.deepcopy(self._snapshots[self._index])
        if self._index < len(self._results) - 1:
            self._index += 1
        return _Payload(payload)


class _FakeBackend:
    def __init__(self, root_snapshot, branch_sessions):
        self.environment_config = {"image": "base-image", "executable": "docker"}
        self.agent_config = {"step_limit": 50}
        self._root_session = _FakeSession(root_snapshot, container_id="root-container")
        self._branch_sessions = list(branch_sessions)

    def create_session(self, spec):
        root = copy.deepcopy(self._root_session._snapshot_payload)
        root["session_id"] = spec.session_id
        root["spec"]["session_id"] = spec.session_id
        self._root_session = _FakeSession(root, container_id="root-container")
        return self._root_session

    def resume_session(self, snapshot):
        return self._branch_sessions.pop(0)


def _root_snapshot():
    return {
        "session_id": "root-session",
        "status": "paused",
        "spec": {"session_id": "root-session", "task": "Fix the failing test.", "task_id": "demo__demo"},
        "agent": {
            "type_path": "fake.agent",
            "config": {},
            "state": {
                "messages": [
                    {"role": "system", "content": "You are SWE-agent."},
                    {"role": "user", "content": "Fix the failing test."},
                ],
                "cost": 0.0,
                "n_calls": 0,
            },
        },
        "model": {"type_path": "fake.model", "config": {"model_name": "openai/fake"}, "state": {}},
        "environment": {"type_path": "fake.env", "config": {"image": "base-image", "cwd": "/repo", "executable": "docker"}, "state": {}},
        "metadata": {"events": [], "model_turns": []},
        "last_step_index": -1,
    }


def _branch_snapshot(parent_snapshot, *, session_id, step_index, assistant_text, command, observation):
    snapshot = copy.deepcopy(parent_snapshot)
    snapshot["session_id"] = session_id
    snapshot["spec"]["session_id"] = session_id
    snapshot["agent"]["state"]["messages"] = copy.deepcopy(parent_snapshot["agent"]["state"]["messages"]) + [
        {"role": "assistant", "content": assistant_text, "extra": {"actions": [{"command": command}]}},
        {"role": "user", "content": observation, "extra": {"raw_output": observation}},
    ]
    snapshot["metadata"]["events"] = copy.deepcopy(parent_snapshot["metadata"]["events"]) + [
        {"event_id": f"{session_id}:0", "session_id": session_id, "step_index": step_index, "kind": "model_response", "ts": 0.0, "payload": {"message": {"content": assistant_text}}},
        {"event_id": f"{session_id}:1", "session_id": session_id, "step_index": step_index, "kind": "environment_action", "ts": 0.0, "payload": {"actions": [{"command": command}]}},
        {"event_id": f"{session_id}:2", "session_id": session_id, "step_index": step_index, "kind": "environment_result", "ts": 0.0, "payload": {"messages": [{"content": observation}]}},
    ]
    snapshot["last_step_index"] = step_index
    return snapshot


def _branch_result(snapshot_payload, *, status="paused", submission=""):
    return {
        "session_id": snapshot_payload["session_id"],
        "status": status,
        "spec": snapshot_payload["spec"],
        "events": [],
        "model_turns": [],
        "final_messages": [],
        "exit_status": "Submitted" if status == "finished" else "",
        "submission": submission,
        "metadata": {"n_calls": 1, "cost": 1.0},
    }


def _make_fake_chat():
    async def fake_chat(route_name, model_name, user_prompt=None, system_prompt=None, **kwargs):
        judge_prompt = "\n".join(part for part in [system_prompt, user_prompt] if part)
        if user_prompt and PERSISTENT_STATE_UPDATE_PROMPT.strip() in user_prompt:
            return json.dumps(
                {
                    "current_state": "Continue targeted validation on `pkg/core.py`.",
                    "task_specification": "Fix the failing test without unrelated changes.",
                    "files_and_functions": "- `pkg/core.py`: target file for the failing behavior.",
                    "errors_and_corrections": "Avoid unrelated inspection once targeted pytest evidence exists.",
                    "codebase_and_system_documentation": "`pkg/core.py` contains the logic being debugged.",
                    "learnings": "Targeted pytest runs provide the clearest signal.",
                    "key_results": "Focused on `pytest tests/test_alpha.py -q` evidence.",
                    "worklog": "- Compressed older validation step\n- Preserved target file context",
                }
            )
        if user_prompt and SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT.strip() in user_prompt:
            return json.dumps(
                {
                    "question": "Fix the failing test.",
                    "positive_rubrics": [
                        {
                            "title": "Validation",
                            "description": "The trajectory runs targeted validation relevant to the fix.",
                            "scale": {
                                "1": "No validation",
                                "2": "Incidental validation",
                                "3": "Some relevant validation",
                                "4": "Targeted validation",
                                "5": "Targeted validation plus edge coverage",
                            },
                        }
                    ],
                    "negative_rubrics": [
                        {
                            "title": "Drift",
                            "description": "The trajectory makes unfocused changes without evidence.",
                            "scale": {
                                "1": "No drift",
                                "2": "Minor drift",
                                "3": "Noticeable drift",
                                "4": "Serious drift",
                                "5": "Severe drift",
                            },
                        }
                    ],
                }
            )
        latest_bad = "echo 'noise' >> notes.txt" in judge_prompt or "sed -n '1,20p' README.md" in judge_prompt
        latest_test = "pytest tests/test_alpha.py -q" in judge_prompt
        is_positive = "Type: positive" in judge_prompt
        if is_positive:
            return json.dumps({"score": 1 if latest_bad else (5 if latest_test else 3)})
        return json.dumps({"score": 5 if latest_bad else (1 if latest_test else 2)})

    return fake_chat


def _patch_docker_subprocess(monkeypatch: pytest.MonkeyPatch, *, initial_images: list[str] | None = None):
    state = {"images": set(initial_images or []), "removed_images": [], "removed_containers": []}

    def fake_run(cmd, check=False, capture_output=False, text=False, **kwargs):
        if cmd[:3] == ["docker", "image", "inspect"]:
            image_tag = cmd[3]
            if image_tag == "base-image":
                return SimpleNamespace(returncode=0, stdout="sha256:base\n", stderr="")
            return SimpleNamespace(
                returncode=0 if image_tag in state["images"] else 1,
                stdout=(f"sha256:{image_tag}\n" if image_tag in state["images"] else ""),
                stderr="",
            )
        if len(cmd) >= 4 and cmd[:3] == ["docker", "image", "ls"]:
            repo = cmd[3]
            return SimpleNamespace(
                returncode=0,
                stdout="\n".join(sorted(tag for tag in state["images"] if tag.startswith(f"{repo}:"))),
                stderr="",
            )
        if cmd[:4] == ["docker", "image", "rm", "-f"]:
            state["removed_images"].append(cmd[4])
            state["images"].discard(cmd[4])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["docker", "rm", "-f"]:
            state["removed_containers"].append(cmd[3])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"Unexpected docker command: {cmd}")

    monkeypatch.setattr("swe_agent.trajectory_search.subprocess.run", fake_run)
    monkeypatch.setattr(
        "swe_agent.trajectory_search._docker_commit",
        lambda executable, container_id, image_tag: (state["images"].add(image_tag) or image_tag, f"sha256:{container_id}"),
    )
    return state


def test_initial_rubric_bank_includes_invalid_patch_negative_rubric():
    rubrics = _build_initial_rubric_bank("Fix the failing test.")

    invalid_patch = next((rubric for rubric in rubrics if rubric.title == "Invalid Patch Format"), None)
    assert invalid_patch is not None
    assert invalid_patch.direction == "negative"
    assert "valid" in invalid_patch.description.lower()
    assert "patch" in invalid_patch.description.lower()
    assert invalid_patch.scale["1"] == "Final patch is a valid, directly applicable patch"
    assert invalid_patch.scale["5"] == "Patch is not a legitimate patch at all"


def test_weighted_reward_and_rubric_bank_update():
    positive = RubricRecord(
        rubric_id="pos",
        title="Validation",
        direction="positive",
        description="Runs tests",
        scale={"1": "none", "2": "weak", "3": "okay", "4": "good", "5": "great"},
        weight=1,
        source_round=1,
    )
    negative = RubricRecord(
        rubric_id="neg",
        title="Drift",
        direction="negative",
        description="Unfocused edits",
        scale={"1": "none", "2": "weak", "3": "okay", "4": "bad", "5": "severe"},
        weight=-1,
        source_round=1,
    )
    score = _compute_weighted_reward(
        [
            {"rubric": {"weight": 1}, "score_normalized": 1.0},
            {"rubric": {"weight": -1}, "score_normalized": 0.25},
        ]
    )
    assert score == pytest.approx(0.75)

    active, inactive, combined = _update_rubric_bank(
        active_bank=[],
        inactive_bank=[],
        generated=[positive, negative],
        variances={"pos": 0.2, "neg": 0.0},
        max_active_rubrics=2,
    )
    assert [rubric.rubric_id for rubric in active] == ["pos", "neg"]
    assert inactive == []
    assert {rubric.rubric_id for rubric in combined} == {"pos", "neg"}


def test_update_rubric_bank_deduplicates_by_title():
    original = RubricRecord(
        rubric_id="old",
        title="Validation",
        direction="positive",
        description="Older validation rubric.",
        scale={"1": "bad", "2": "weak", "3": "ok", "4": "good", "5": "great"},
        weight=1,
        source_round=1,
    )
    duplicate = RubricRecord(
        rubric_id="new",
        title="Validation",
        direction="positive",
        description="Newer validation rubric.",
        scale={"1": "bad", "2": "weak", "3": "ok", "4": "good", "5": "great"},
        weight=1,
        source_round=2,
    )
    negative = RubricRecord(
        rubric_id="neg",
        title="Drift",
        direction="negative",
        description="Unfocused edits",
        scale={"1": "none", "2": "weak", "3": "okay", "4": "bad", "5": "severe"},
        weight=-1,
        source_round=1,
    )

    active, inactive, combined = _update_rubric_bank(
        active_bank=[original],
        inactive_bank=[],
        generated=[duplicate, negative],
        variances={"old": 0.1, "new": 0.3, "neg": 0.2},
        max_active_rubrics=2,
    )

    assert [rubric.title for rubric in combined] == ["Validation", "Drift"]
    assert [rubric.rubric_id for rubric in active] == ["new", "neg"]
    assert inactive == []


def test_build_step_cards_keeps_assistant_message_and_truncates_large_sections():
    assistant_text = "<think>diagnose issue</think>\nTHOUGHT: run reproduction first"
    long_exception = "E" * 6000
    long_output = "O" * 1200
    cards = _build_step_cards(
        [
            {
                "step_index": 3,
                "kind": "model_response",
                "payload": {"message": {"content": assistant_text}},
            },
            {
                "step_index": 3,
                "kind": "environment_action",
                "payload": {"actions": [{"command": "python repro.py"}]},
            },
            {
                "step_index": 3,
                "kind": "environment_result",
                "payload": {
                    "messages": [
                        {
                            "content": (
                                f"<exception>{long_exception}</exception>\n"
                                "<returncode>1</returncode>\n"
                                f"<output>\n{long_output}</output>"
                            )
                        }
                    ]
                },
            },
        ]
    )

    assert len(cards) == 1
    assert cards[0]["assistant_message"] == assistant_text
    assert cards[0]["commands"] == ["python repro.py"]
    assert "signal" not in cards[0]
    assert OBSERVATION_TRUNCATION_MARKER in cards[0]["observation"]
    assert "<exception>" in cards[0]["observation"]
    assert "<returncode>1</returncode>" in cards[0]["observation"]
    assert "<output>\n" in cards[0]["observation"]
    exception_body = re.search(r"<exception>(.*?)</exception>", cards[0]["observation"], re.DOTALL).group(1)
    output_body = re.search(r"<output>\n(.*?)</output>", cards[0]["observation"], re.DOTALL).group(1)
    assert OBSERVATION_TRUNCATION_MARKER in exception_body
    assert OBSERVATION_TRUNCATION_MARKER in output_body
    assert len(exception_body) < len(long_exception)
    assert len(output_body) < len(long_output)
    assert len(exception_body) > len(output_body)


def test_truncate_structured_observation_keeps_small_section_untruncated():
    long_exception = "E" * 6000
    short_output = "O" * (MAX_OBSERVATION_CHARS // 10)
    observation = (
        f"<exception>{long_exception}</exception>\n"
        "<returncode>1</returncode>\n"
        f"<output>\n{short_output}</output>"
    )

    truncated = _truncate_structured_observation(observation)

    assert len(truncated) <= MAX_OBSERVATION_CHARS
    exception_body = re.search(r"<exception>(.*?)</exception>", truncated, re.DOTALL).group(1)
    output_body = re.search(r"<output>\n(.*?)</output>", truncated, re.DOTALL).group(1)
    assert OBSERVATION_TRUNCATION_MARKER in exception_body
    assert OBSERVATION_TRUNCATION_MARKER not in output_body
    assert output_body == short_output


@pytest.mark.asyncio
async def test_prepare_round_judging_updates_shared_context_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _root_snapshot()
    backend = _FakeBackend(root, [])
    runner = TrajectorySearchRunner(
        instance={"instance_id": "demo__demo-shared", "problem_statement": "Fix the failing test."},
        backend=backend,
        run_dir=tmp_path / "run",
        policy_model_name="openai/fake",
        search_config=SearchConfig(
            m=2,
            k=1,
            p=1,
            max_rounds=1,
            rubric_temperature=0.0,
            rubric_top_p=1.0,
            rubric_max_tokens=111,
            judge_temperature=0.0,
            judge_top_p=1.0,
            judge_max_tokens=37,
        ),
    )
    runner.system_prompt = "You are SWE-agent."
    runner.user_prompt = "Fix the failing test."
    runner.active_bank = []
    runner.inactive_bank = []

    grandparent_dir = tmp_path / "run" / "nodes" / "grandparent"
    grandparent_dir.mkdir(parents=True, exist_ok=True)
    grandparent_judge_path = grandparent_dir / "judge.json"
    grandparent_judge_path.write_text(
        json.dumps(
            {
                "workspace_meta": {
                    "changed_files": ["pkg/from-grandparent.py"],
                    "untracked_files": [],
                    "diff_stat": " pkg/from-grandparent.py | 2 +-",
                    "current_patch_chars": 12,
                    "workspace_fingerprint": "grandparent",
                }
            }
        ),
        encoding="utf-8",
    )
    runner.nodes = {
        "grandparent": SearchNode(
            node_id="grandparent",
            parent_id="root",
            round_index=1,
            depth=1,
            session_id="grandparent-session",
            status="frontier",
            checkpoint_image_tag=None,
            checkpoint_image_id=None,
            workspace_fingerprint="grandparent",
            raw_traj_path=str(grandparent_dir / "raw_traj.json"),
            messages_path=str(grandparent_dir / "messages.json"),
            judge_path=str(grandparent_judge_path),
            snapshot_path=None,
            rubric_ref=None,
        )
    }
    parent_node = SearchNode(
        node_id="parent",
        parent_id="grandparent",
        round_index=2,
        depth=2,
        session_id="parent-session",
        status="frontier",
        checkpoint_image_tag=None,
        checkpoint_image_id=None,
        workspace_fingerprint="parent",
        raw_traj_path=str(tmp_path / "run" / "nodes" / "parent" / "raw_traj.json"),
        messages_path=str(tmp_path / "run" / "nodes" / "parent" / "messages.json"),
        judge_path=str(tmp_path / "run" / "nodes" / "parent" / "judge.json"),
        snapshot_path=None,
        rubric_ref=None,
    )
    runner.nodes["parent"] = parent_node

    parent_judge = {
        "persistent_state": {
            **copy.deepcopy(EMPTY_PERSISTENT_STATE),
            "current_state": "Investigating the failure from the parent branch.",
            "key_results": "- old-shared-state",
        },
        "recent_segments": [
            {
                "step_cards": [
                    {
                        "step_index": 0,
                        "assistant_message": "older reasoning",
                        "commands": ["pytest tests/test_alpha.py -q"],
                        "observation": "<returncode>0</returncode>\n<output>\n1 passed\n</output>",
                    }
                ],
                "segment_step_range": [0, 0],
            },
            {
                "step_cards": [
                    {
                        "step_index": 1,
                        "assistant_message": "shared reasoning",
                        "commands": ["sed -n '1,40p' pkg/core.py"],
                        "observation": "<returncode>0</returncode>\n<output>\nclass Core: ...\n</output>",
                    }
                ],
                "segment_step_range": [1, 1],
            },
        ],
        "workspace_meta": {
            "changed_files": ["pkg/from-parent.py"],
            "untracked_files": [],
            "diff_stat": " pkg/from-parent.py | 1 +",
            "current_patch_chars": 7,
            "workspace_fingerprint": "parent",
        },
    }
    branch_records = [
        {
            "recent_segments": [
                copy.deepcopy(parent_judge["recent_segments"][1]),
                {
                    "step_cards": [
                        {
                            "step_index": 2,
                            "assistant_message": "branch one",
                            "commands": ["pytest tests/test_alpha.py::test_one -q"],
                            "observation": "<returncode>0</returncode>\n<output>\n1 passed\n</output>",
                        }
                    ],
                    "segment_step_range": [2, 2],
                },
            ],
            "workspace_meta": {
                "changed_files": ["pkg/core.py"],
                "untracked_files": [],
                "diff_stat": " pkg/core.py | 2 +-",
                "current_patch_chars": 20,
                "workspace_fingerprint": "branch-1",
            },
            "result": {"status": "paused", "exit_status": "", "submission": ""},
        },
        {
            "recent_segments": [
                copy.deepcopy(parent_judge["recent_segments"][1]),
                {
                    "step_cards": [
                        {
                            "step_index": 2,
                            "assistant_message": "branch two",
                            "commands": ["pytest tests/test_alpha.py::test_two -q"],
                            "observation": "<returncode>1</returncode>\n<output>\n1 failed\n</output>",
                        }
                    ],
                    "segment_step_range": [2, 2],
                },
            ],
            "workspace_meta": {
                "changed_files": ["pkg/core.py"],
                "untracked_files": [],
                "diff_stat": " pkg/core.py | 3 ++-",
                "current_patch_chars": 30,
                "workspace_fingerprint": "branch-2",
            },
            "result": {"status": "paused", "exit_status": "", "submission": ""},
        },
    ]

    calls = {"persistent": [], "rubric": [], "judge": []}

    async def fake_chat(route_name, model_name, user_prompt=None, system_prompt=None, **kwargs):
        if user_prompt and PERSISTENT_STATE_UPDATE_PROMPT.strip() in user_prompt:
            calls["persistent"].append({"prompt": user_prompt, "kwargs": kwargs})
            return json.dumps(
                {
                    "current_state": "Continue from the shared pytest context and compare branch-specific validation.",
                    "task_specification": "Fix the failing test while keeping the branch focused on relevant files.",
                    "files_and_functions": "- `pkg/from-grandparent.py`: older shared target\n- `pkg/core.py`: active branch file",
                    "errors_and_corrections": "Do not revisit the parent's broad scan now that focused evidence exists.",
                    "codebase_and_system_documentation": "`pkg/core.py` remains the active implementation hotspot.",
                    "learnings": "Targeted branch-level pytest commands are the most informative differentiator.",
                    "key_results": "- compressed-once",
                    "worklog": "- Moved older segment into persistent memory\n- Preserved shared branch context",
                }
            )
        if route_name == "rubric_generation":
            calls["rubric"].append({"prompt": user_prompt, "kwargs": kwargs})
            return json.dumps(
                {
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
        calls["judge"].append({"prompt": user_prompt, "kwargs": kwargs, "system_prompt": system_prompt})
        return json.dumps({"score": 4})

    monkeypatch.setattr("swe_agent.trajectory_search.run_chat_with_route_async", fake_chat)

    judged = await runner._prepare_round_judging(
        parent_node=parent_node,
        parent_judge=parent_judge,
        branch_records=branch_records,
        round_index=3,
        compare_parent=False,
    )

    assert len(calls["persistent"]) == 1
    assert '"segment_step_range"' not in calls["persistent"][0]["prompt"]
    assert "pkg/from-grandparent.py" in calls["persistent"][0]["prompt"]
    assert "pkg/from-parent.py" not in calls["persistent"][0]["prompt"]
    assert "compressed-once" in calls["rubric"][0]["prompt"]
    assert calls["rubric"][0]["kwargs"]["temperature"] == 0.0
    assert calls["rubric"][0]["kwargs"]["top_p"] == 1.0
    assert calls["rubric"][0]["kwargs"]["max_tokens"] == 111
    assert all("compressed-once" in call["prompt"] for call in calls["judge"])
    assert all("old-shared-state" not in call["prompt"] for call in calls["judge"])
    assert all(call["kwargs"]["temperature"] == 0.0 for call in calls["judge"])
    assert all(call["kwargs"]["top_p"] == 1.0 for call in calls["judge"])
    assert all(call["kwargs"]["max_tokens"] == 37 for call in calls["judge"])
    assert all(branch["persistent_state"]["current_state"] == "Continue from the shared pytest context and compare branch-specific validation." for branch in branch_records)
    assert all(branch["persistent_state"]["key_results"] == "- compressed-once" for branch in branch_records)
    assert judged["generated"]


@pytest.mark.asyncio
async def test_update_persistent_state_preserves_existing_sections_when_model_skips_them(monkeypatch: pytest.MonkeyPatch):
    async def fake_chat(route_name, model_name, user_prompt=None, system_prompt=None, **kwargs):
        return json.dumps(
            {
                "current_state": "Editing `astropy/utils/misc.py` and rerunning the focused property case next.",
                "worklog": "- Re-read the metaclass logic\n- Prepared the next targeted validation",
            }
        )

    monkeypatch.setattr("swe_agent.trajectory_search.run_chat_with_route_async", fake_chat)

    previous_state = {
        "current_state": "Investigating the failure.",
        "task_specification": "Fix property docstring inheritance without touching tests.",
        "files_and_functions": "- `astropy/utils/misc.py`: contains `InheritDocstrings`.",
        "errors_and_corrections": "Avoid broad repo scans once the target metaclass is identified.",
        "codebase_and_system_documentation": "`InheritDocstrings` copies docstrings from base classes.",
        "learnings": "Targeted reproduction is more useful than speculative edits.",
        "key_results": "Confirmed `inspect.isfunction(property(...))` is false.",
        "worklog": "- Read the target file",
    }
    state = await _update_persistent_state(
        system_prompt="You are SWE-agent.",
        user_prompt="Fix the bug.",
        previous_state=previous_state,
        evicted_step_cards=[{"step_index": 0, "assistant_message": "Inspect file"}],
        workspace_meta=copy.deepcopy(EMPTY_WORKSPACE_META),
        model_name="openai/fake",
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
    )

    assert state["current_state"] == "Editing `astropy/utils/misc.py` and rerunning the focused property case next."
    assert state["worklog"] == "- Re-read the metaclass logic\n- Prepared the next targeted validation"
    assert state["task_specification"] == previous_state["task_specification"]
    assert state["files_and_functions"] == previous_state["files_and_functions"]


@pytest.mark.asyncio
async def test_rubric_and_judge_retry_up_to_four_times(monkeypatch: pytest.MonkeyPatch):
    rubric_attempts = {"count": 0}
    judge_attempts = {"count": 0}

    async def fake_chat(route_name, model_name, user_prompt=None, system_prompt=None, **kwargs):
        if route_name == "rubric_generation":
            rubric_attempts["count"] += 1
            if rubric_attempts["count"] < 4:
                return "not valid json"
            return json.dumps(
                {
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
        judge_attempts["count"] += 1
        if judge_attempts["count"] < 4:
            return '{"score": "bad"}'
        return json.dumps({"score": 5})

    monkeypatch.setattr("swe_agent.trajectory_search.run_chat_with_route_async", fake_chat)

    rubrics = await _generate_round_rubrics(
        question={"system_prompt": "sys", "user_prompt": "user"},
        previous_state={},
        latest_shared_segment=None,
        continuations=[
            {
                "summary": {"step_count": 1},
                "trajectory_continuation": {"step_cards": [{"commands": ["pytest -q"]}]},
            }
        ],
        active_bank=[],
        model_name="openai/fake",
        temperature=0.0,
        top_p=1.0,
        max_tokens=123,
        round_index=1,
    )
    scores, _, errors = await _score_round(
        question={"system_prompt": "sys", "user_prompt": "user"},
        shared_context={"previous_persistent_state": {}, "latest_agent_trajectory": None},
        continuations=[
            {
                "summary": {"step_count": 1},
                "trajectory_continuation": {"step_cards": [{"commands": ["pytest -q"]}]},
            }
        ],
        rubrics=rubrics,
        model_name="openai/fake",
        temperature=0.0,
        top_p=1.0,
        max_tokens=33,
    )

    assert rubric_attempts["count"] == 4
    assert judge_attempts["count"] == 4
    assert len(rubrics) == 1
    assert errors == {}
    assert scores[0][0]["score_raw"] == 5


@pytest.mark.asyncio
async def test_score_parent_round_scores_latest_trajectory_without_continuation(monkeypatch: pytest.MonkeyPatch):
    prompts = []

    async def fake_chat(route_name, model_name, user_prompt=None, system_prompt=None, **kwargs):
        prompts.append(user_prompt)
        return json.dumps({"score": 3})

    monkeypatch.setattr("swe_agent.trajectory_search.run_chat_with_route_async", fake_chat)

    rubric = RubricRecord(
        rubric_id="pos",
        title="Validation",
        direction="positive",
        description="Runs targeted validation.",
        scale={"1": "none", "2": "weak", "3": "some", "4": "good", "5": "great"},
        weight=1,
        source_round=1,
    )
    scores, variances, errors = await _score_parent_round(
        question={"system_prompt": "sys", "user_prompt": "user"},
        shared_context={
            "previous_persistent_state": {"current_state": "Investigating root cause."},
            "latest_agent_trajectory": {"step_cards": [{"commands": ["pytest -q"]}]},
        },
        rubrics=[rubric],
        model_name="openai/fake",
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
    )

    assert errors == {}
    assert variances == {"pos": 0.0}
    assert len(scores) == 1
    assert scores[0][0]["score_raw"] == 3
    assert "## Agent Trajectory:" in prompts[0]
    assert "## Continuation Trajectory:" not in prompts[0]


@pytest.mark.asyncio
async def test_prepare_round_judging_keeps_only_active_rubric_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _root_snapshot()
    backend = _FakeBackend(root, [])
    runner = TrajectorySearchRunner(
        instance={"instance_id": "demo__demo-active", "problem_statement": "Fix the failing test."},
        backend=backend,
        run_dir=tmp_path / "run",
        policy_model_name="openai/fake",
        search_config=SearchConfig(m=2, k=1, p=1, max_rounds=1, max_active_rubrics=1),
    )
    runner.system_prompt = "You are SWE-agent."
    runner.user_prompt = "Fix the failing test."
    runner.active_bank = []
    runner.inactive_bank = []

    parent_node = SearchNode(
        node_id="parent",
        parent_id=None,
        round_index=1,
        depth=1,
        session_id="parent-session",
        status="frontier",
        checkpoint_image_tag=None,
        checkpoint_image_id=None,
        workspace_fingerprint="parent",
        raw_traj_path=str(tmp_path / "run" / "nodes" / "parent" / "raw_traj.json"),
        messages_path=str(tmp_path / "run" / "nodes" / "parent" / "messages.json"),
        judge_path=str(tmp_path / "run" / "nodes" / "parent" / "judge.json"),
        snapshot_path=None,
        rubric_ref=None,
    )
    runner.nodes["parent"] = parent_node
    parent_judge = {
        "persistent_state": copy.deepcopy(EMPTY_PERSISTENT_STATE),
        "recent_segments": [],
        "workspace_meta": copy.deepcopy(EMPTY_WORKSPACE_META),
    }
    branch_records = [
        {
            "recent_segments": [
                {
                    "step_cards": [{"step_index": 0, "assistant_message": "a", "commands": ["pytest -q"], "observation": "ok"}],
                    "segment_step_range": [0, 0],
                }
            ],
            "workspace_meta": copy.deepcopy(EMPTY_WORKSPACE_META),
            "result": {"status": "paused", "exit_status": "", "submission": ""},
        },
        {
            "recent_segments": [
                {
                    "step_cards": [{"step_index": 0, "assistant_message": "b", "commands": ["sed -n '1,20p' x"], "observation": "ok"}],
                    "segment_step_range": [0, 0],
                }
            ],
            "workspace_meta": copy.deepcopy(EMPTY_WORKSPACE_META),
            "result": {"status": "paused", "exit_status": "", "submission": ""},
        },
    ]

    async def fake_chat(route_name, model_name, user_prompt=None, system_prompt=None, **kwargs):
        if route_name == "rubric_generation":
            return json.dumps(
                {
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
                        },
                        {
                            "title": "Grounding",
                            "description": "Stays grounded in the relevant files.",
                            "scale": {
                                "1": "Ungrounded",
                                "2": "Weak grounding",
                                "3": "Some grounding",
                                "4": "Good grounding",
                                "5": "Excellent grounding",
                            },
                        },
                    ],
                    "negative_rubrics": [],
                }
            )
        if "Title: Validation" in user_prompt:
            return json.dumps({"score": 5 if "pytest -q" in user_prompt else 1})
        return json.dumps({"score": 3})

    monkeypatch.setattr("swe_agent.trajectory_search.run_chat_with_route_async", fake_chat)

    judged = await runner._prepare_round_judging(
        parent_node=parent_node,
        parent_judge=parent_judge,
        branch_records=branch_records,
        round_index=2,
        compare_parent=False,
    )

    assert len(judged["active_after"]) == 1
    active_ids = {rubric.rubric_id for rubric in judged["active_after"]}
    assert all({record["rubric_id"] for record in score_records} == active_ids for score_records in judged["child_scores"])


@pytest.mark.asyncio
async def test_score_round_uses_error_handling_after_retry_exhaustion(monkeypatch: pytest.MonkeyPatch):
    async def fake_chat(route_name, model_name, user_prompt=None, system_prompt=None, **kwargs):
        return "not valid json"

    monkeypatch.setattr("swe_agent.trajectory_search.run_chat_with_route_async", fake_chat)

    rubric = RubricRecord(
        rubric_id="validation",
        title="Validation",
        direction="positive",
        description="Runs targeted validation.",
        scale={"1": "No validation", "2": "Weak", "3": "Some", "4": "Targeted", "5": "Strong"},
        weight=1,
        source_round=1,
    )
    scores, variances, errors = await _score_round(
        question={"system_prompt": "sys", "user_prompt": "user"},
        shared_context={"previous_persistent_state": {}, "latest_agent_trajectory": None},
        continuations=[
            {
                "summary": {"step_count": 1},
                "trajectory_continuation": {"step_cards": [{"commands": ["pytest -q"]}]},
            }
        ],
        rubrics=[rubric],
        model_name="openai/fake",
        temperature=0.0,
        top_p=1.0,
        max_tokens=33,
    )

    assert errors == {}
    assert variances == {"validation": 0.0}
    assert len(scores[0]) == 1
    assert scores[0][0]["score_raw"] == 1
    assert json.loads(scores[0][0]["judge_response"])["score"] == 1


@pytest.mark.asyncio
async def test_score_round_preserves_other_scores_when_one_rubric_fails(monkeypatch: pytest.MonkeyPatch):
    async def fake_chat(route_name, model_name, user_prompt=None, system_prompt=None, **kwargs):
        if "Title: Validation" in user_prompt:
            return json.dumps({"reasoning": "targeted validation", "score": 5})
        return "not valid json"

    monkeypatch.setattr("swe_agent.trajectory_search.run_chat_with_route_async", fake_chat)

    rubrics = [
        RubricRecord(
            rubric_id="validation",
            title="Validation",
            direction="positive",
            description="Runs targeted validation.",
            scale={"1": "No validation", "2": "Weak", "3": "Some", "4": "Targeted", "5": "Strong"},
            weight=1,
            source_round=1,
        ),
        RubricRecord(
            rubric_id="precision",
            title="Precision",
            direction="positive",
            description="Makes precise changes.",
            scale={"1": "Poor", "2": "Weak", "3": "Mixed", "4": "Good", "5": "Excellent"},
            weight=1,
            source_round=1,
        ),
    ]
    scores, variances, errors = await _score_round(
        question={"system_prompt": "sys", "user_prompt": "user"},
        shared_context={"previous_persistent_state": {}, "latest_agent_trajectory": None},
        continuations=[
            {
                "summary": {"step_count": 1},
                "trajectory_continuation": {"step_cards": [{"commands": ["pytest -q"]}]},
            }
        ],
        rubrics=rubrics,
        model_name="openai/fake",
        temperature=0.0,
        top_p=1.0,
        max_tokens=33,
    )

    assert errors == {}
    assert {record["rubric_id"] for record in scores[0]} == {"validation", "precision"}
    assert {record["rubric_id"]: record["score_raw"] for record in scores[0]} == {"validation": 5, "precision": 1}
    assert variances == {"validation": 0.0, "precision": 0.0}


def test_collect_workspace_meta_builds_python_literal_payload():
    class _FakeEnv:
        config = SimpleNamespace(cwd="/repo")

        def execute(self, action):
            assert '"git_repo": False' in action["command"]
            assert '"workspace_fingerprint": None' in action["command"]
            return {
                "returncode": 0,
                "output": json.dumps(
                    {
                        "cwd": "/repo",
                        "git_repo": True,
                        "head_commit": "abc123",
                        "changed_files": ["pkg/core.py"],
                        "untracked_files": [],
                        "status": [" M pkg/core.py"],
                        "diff_stat": " pkg/core.py | 2 +-",
                        "current_patch_chars": 120,
                        "workspace_fingerprint": "fp",
                    }
                ),
            }

    meta = _collect_workspace_meta(_FakeEnv())

    assert meta["git_repo"] is True
    assert meta["head_commit"] == "abc123"
    assert meta["changed_files"] == ["pkg/core.py"]
    assert meta["workspace_fingerprint"] == "fp"


def test_new_frontier_children_are_prepended_before_existing_frontier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _root_snapshot()
    child_a = _branch_snapshot(
        root,
        session_id="child-a",
        step_index=0,
        assistant_text="Run the failing test.",
        command="pytest tests/test_alpha.py -q",
        observation="<returncode>0</returncode>\n<output>\n1 passed\n</output>",
    )
    child_b = _branch_snapshot(
        root,
        session_id="child-b",
        step_index=0,
        assistant_text="Read the code path.",
        command="sed -n '1,80p' pkg/core.py",
        observation="<returncode>0</returncode>\n<output>\nclass Core: ...\n</output>",
    )
    grandchild_a = _branch_snapshot(
        child_a,
        session_id="grandchild-a",
        step_index=1,
        assistant_text="Run a better targeted test.",
        command="pytest tests/test_beta.py -q",
        observation="<returncode>0</returncode>\n<output>\n1 passed\n</output>",
    )
    grandchild_b = _branch_snapshot(
        child_a,
        session_id="grandchild-b",
        step_index=1,
        assistant_text="Inspect unrelated file.",
        command="sed -n '1,20p' README.md",
        observation="<returncode>0</returncode>\n<output>\nREADME\n</output>",
    )
    backend = _FakeBackend(
        root,
        [
            _FakeSession(child_a, _branch_result(child_a), container_id="child-a-container"),
            _FakeSession(child_b, _branch_result(child_b), container_id="child-b-container"),
            _FakeSession(grandchild_a, _branch_result(grandchild_a), container_id="grandchild-a-container"),
            _FakeSession(grandchild_b, _branch_result(grandchild_b), container_id="grandchild-b-container"),
        ],
    )
    workspace_meta = {
        "root-container": {"changed_files": [], "untracked_files": [], "diff_stat": "", "current_patch_chars": 0, "workspace_fingerprint": "root"},
        "child-a-container": {"changed_files": ["pkg/core.py"], "untracked_files": [], "diff_stat": " pkg/core.py | 2 +-", "current_patch_chars": 120, "workspace_fingerprint": "a"},
        "child-b-container": {"changed_files": [], "untracked_files": [], "diff_stat": "", "current_patch_chars": 0, "workspace_fingerprint": "b"},
        "grandchild-a-container": {"changed_files": ["pkg/core.py"], "untracked_files": [], "diff_stat": " pkg/core.py | 3 ++-", "current_patch_chars": 180, "workspace_fingerprint": "ga"},
        "grandchild-b-container": {"changed_files": [], "untracked_files": [], "diff_stat": "", "current_patch_chars": 0, "workspace_fingerprint": "gb"},
    }

    def _make_prepend_chat():
        async def fake_chat(route_name, model_name, user_prompt=None, system_prompt=None, **kwargs):
            judge_prompt = "\n".join(part for part in [system_prompt, user_prompt] if part)
            if user_prompt and PERSISTENT_STATE_UPDATE_PROMPT.strip() in user_prompt:
                return json.dumps(
                    {
                        "current_state": "Continue focused debugging on `pkg/core.py`.",
                        "task_specification": "Fix the failing test without unrelated edits.",
                        "files_and_functions": "- `pkg/core.py`: active target file",
                        "errors_and_corrections": "Discard unrelated file inspection.",
                        "codebase_and_system_documentation": "`pkg/core.py` contains the failing logic.",
                        "learnings": "Targeted validation is the best signal.",
                        "key_results": "Retained focused pytest evidence.",
                        "worklog": "- Compressed the earlier shared segment",
                    }
                )
            if user_prompt and SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT.strip() in user_prompt:
                return json.dumps(
                    {
                        "question": "Fix the failing test.",
                        "positive_rubrics": [
                            {
                                "title": "Validation",
                                "description": "The trajectory runs targeted validation relevant to the fix.",
                                "scale": {
                                    "1": "No validation",
                                    "2": "Incidental validation",
                                    "3": "Some relevant validation",
                                    "4": "Targeted validation",
                                    "5": "Targeted validation plus edge coverage",
                                },
                            }
                        ],
                        "negative_rubrics": [],
                    }
                )
            is_positive = "Type: positive" in judge_prompt
            is_baseline = "No continuation beyond the shared trajectory." in judge_prompt
            if not is_positive:
                return json.dumps({"score": 1})
            if "pytest tests/test_beta.py -q" in judge_prompt:
                return json.dumps({"score": 5})
            if "pytest tests/test_alpha.py -q" in judge_prompt:
                return json.dumps({"score": 4 if is_baseline else 5})
            if "sed -n '1,80p' pkg/core.py" in judge_prompt:
                return json.dumps({"score": 3})
            if "sed -n '1,20p' README.md" in judge_prompt:
                return json.dumps({"score": 1})
            return json.dumps({"score": 3})

        return fake_chat

    monkeypatch.setattr("swe_agent.trajectory_search.run_chat_with_route_async", _make_prepend_chat())
    monkeypatch.setattr("swe_agent.trajectory_search._collect_workspace_meta", lambda env: workspace_meta[env.container_id])
    _patch_docker_subprocess(monkeypatch)

    runner = TrajectorySearchRunner(
        instance={"instance_id": "demo__prepend-order", "problem_statement": "Fix the failing test."},
        backend=backend,
        run_dir=tmp_path / "run",
        policy_model_name="openai/fake",
        search_config=SearchConfig(m=1, k=1, p=2, max_rounds=2, max_active_rubrics=2),
    )
    runner.run()

    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert len(manifest["frontier_ids"]) == 2
    assert manifest["frontier_ids"][0].startswith("node-r002-s00")
    assert manifest["frontier_ids"][1].startswith("node-r001-s")


def test_trajectory_search_runner_writes_expected_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _root_snapshot()
    child_a = _branch_snapshot(
        root,
        session_id="child-a",
        step_index=0,
        assistant_text="Run the failing test.",
        command="pytest tests/test_alpha.py -q",
        observation="<returncode>0</returncode>\n<output>\n1 passed\n</output>",
    )
    child_b = _branch_snapshot(
        root,
        session_id="child-b",
        step_index=0,
        assistant_text="Inspect the file only.",
        command="sed -n '1,40p' pkg/core.py",
        observation="<returncode>0</returncode>\n<output>\nclass Core: ...\n</output>",
    )
    backend = _FakeBackend(
        root,
        [
            _FakeSession(child_a, _branch_result(child_a), container_id="child-a-container"),
            _FakeSession(child_b, _branch_result(child_b), container_id="child-b-container"),
        ],
    )
    workspace_meta = {
        "root-container": {"changed_files": [], "untracked_files": [], "diff_stat": "", "current_patch_chars": 0, "workspace_fingerprint": "root"},
        "child-a-container": {"changed_files": ["pkg/core.py"], "untracked_files": [], "diff_stat": " pkg/core.py | 2 +-", "current_patch_chars": 120, "workspace_fingerprint": "a"},
        "child-b-container": {"changed_files": [], "untracked_files": [], "diff_stat": "", "current_patch_chars": 0, "workspace_fingerprint": "b"},
    }

    monkeypatch.setattr("swe_agent.trajectory_search.run_chat_with_route_async", _make_fake_chat())
    monkeypatch.setattr("swe_agent.trajectory_search._collect_workspace_meta", lambda env: workspace_meta[env.container_id])
    docker_state = _patch_docker_subprocess(monkeypatch)

    runner = TrajectorySearchRunner(
        instance={"instance_id": "demo__demo-1", "problem_statement": "Fix the failing test."},
        backend=backend,
        run_dir=tmp_path / "run",
        policy_model_name="openai/fake",
        search_config=SearchConfig(m=1, k=1, p=1, max_rounds=1, max_active_rubrics=2),
    )
    result = runner.run()

    assert Path(result.raw_trajectory_path).exists()
    assert Path(result.slim_trajectory_path).exists()
    assert Path(result.patch_path).exists()
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["active_bank"]
    assert manifest["frontier_ids"]
    round_payload = json.loads((tmp_path / "run" / "rubrics" / "round_001.json").read_text())
    assert round_payload["generated"]
    assert any(value > 0 for value in round_payload["variance_by_rubric"].values())

    node_ids = [node_id for node_id in manifest["node_ids"] if node_id != "root"]
    assert len(node_ids) >= 2
    kept_nodes = []
    dropped_nodes = []
    for node_id in node_ids:
        node = json.loads((tmp_path / "run" / "nodes" / node_id / "node.json").read_text())
        judge = json.loads((tmp_path / "run" / "nodes" / node_id / "judge.json").read_text())
        assert isinstance(judge["scores"], list)
        assert judge["rubric_round"]["active_bank_after"]
        assert "persistent_state" in judge
        assert "recent_segments" in judge
        if node["status"] == "frontier":
            kept_nodes.append(node_id)
            assert (tmp_path / "run" / "nodes" / node_id / "snapshot.json").exists()
        else:
            dropped_nodes.append(node_id)
            assert not (tmp_path / "run" / "nodes" / node_id / "snapshot.json").exists()
    assert len(kept_nodes) == 1
    assert len(dropped_nodes) >= 1
    assert len(docker_state["images"]) == 1


def test_trajectory_search_runner_calculates_gt_reward_and_writes_terminal_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _root_snapshot()
    child_a = _branch_snapshot(
        root,
        session_id="child-a",
        step_index=0,
        assistant_text="Run the failing test.",
        command="pytest tests/test_alpha.py -q",
        observation="<returncode>0</returncode>\n<output>\n1 passed\n</output>",
    )
    child_b = _branch_snapshot(
        root,
        session_id="child-b",
        step_index=0,
        assistant_text="Inspect the file only.",
        command="sed -n '1,40p' pkg/core.py",
        observation="<returncode>0</returncode>\n<output>\nclass Core: ...\n</output>",
    )
    child_a_final = _branch_snapshot(
        child_a,
        session_id="child-a-final",
        step_index=1,
        assistant_text="Submit the patch.",
        command="printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\npatch-a\\n'",
        observation="<returncode>0</returncode>\n<output>\nsubmitted\n</output>",
    )
    child_b_final = _branch_snapshot(
        child_b,
        session_id="child-b-final",
        step_index=1,
        assistant_text="Submit the weaker patch.",
        command="printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\npatch-b\\n'",
        observation="<returncode>0</returncode>\n<output>\nsubmitted\n</output>",
    )
    backend = _FakeBackend(
        root,
        [
            _SequentialSession(
                [child_a, child_a_final],
                [_branch_result(child_a), _branch_result(child_a_final, status="finished", submission="patch-a")],
                container_id="child-a-container",
            ),
            _SequentialSession(
                [child_b, child_b_final],
                [_branch_result(child_b), _branch_result(child_b_final, status="finished", submission="patch-b")],
                container_id="child-b-container",
            ),
        ],
    )
    workspace_meta = {
        "root-container": {"changed_files": [], "untracked_files": [], "diff_stat": "", "current_patch_chars": 0, "workspace_fingerprint": "root"},
        "child-a-container": {"changed_files": ["pkg/core.py"], "untracked_files": [], "diff_stat": " pkg/core.py | 2 +-", "current_patch_chars": 120, "workspace_fingerprint": "a"},
        "child-b-container": {"changed_files": [], "untracked_files": [], "diff_stat": "", "current_patch_chars": 0, "workspace_fingerprint": "b"},
    }

    monkeypatch.setattr("swe_agent.trajectory_search.run_chat_with_route_async", _make_fake_chat())
    monkeypatch.setattr("swe_agent.trajectory_search._collect_workspace_meta", lambda env: workspace_meta[env.container_id])
    monkeypatch.setattr(
        "swe_agent.trajectory_search.evaluate_swebench_instance_patches",
        lambda **kwargs: {
            node_id: (1.0 if patch == "patch-a" else 0.0)
            for node_id, patch in kwargs["patches_by_key"].items()
        },
    )
    _patch_docker_subprocess(monkeypatch)

    runner = TrajectorySearchRunner(
        instance={"instance_id": "demo__gt-reward", "problem_statement": "Fix the failing test."},
        backend=backend,
        run_dir=tmp_path / "run",
        policy_model_name="openai/fake",
        harness_namespace="test-namespace",
        search_config=SearchConfig(m=1, k=1, p=1, max_rounds=1, max_active_rubrics=2, calculate_gt_reward=True),
    )
    runner.run()

    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    node_ids = [node_id for node_id in manifest["node_ids"] if node_id != "root"]
    assert len(node_ids) >= 2
    for node_id in node_ids:
        node_dir = tmp_path / "run" / "nodes" / node_id
        if (node_dir / "terminal_patch.json").exists():
            assert (node_dir / "terminal_raw_traj.json").exists()
            assert (node_dir / "terminal_messages.json").exists()
            assert (node_dir / "terminal_patch.json").exists()
    rewards = {
        node_id: json.loads((tmp_path / "run" / "nodes" / node_id / "judge.json").read_text())["ground_truth_reward"]
        for node_id in node_ids
    }
    assert 0.0 in rewards.values()
    assert 1.0 in rewards.values()


def test_trajectory_search_runner_root_expansion_keeps_top_valid_children(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _root_snapshot()
    child_a = _branch_snapshot(
        root,
        session_id="child-a",
        step_index=0,
        assistant_text="Run the failing test.",
        command="pytest tests/test_alpha.py -q",
        observation="<returncode>0</returncode>\n<output>\n1 passed\n</output>",
    )
    child_b = _branch_snapshot(
        root,
        session_id="child-b",
        step_index=0,
        assistant_text="Read the code path.",
        command="sed -n '1,80p' pkg/core.py",
        observation="<returncode>0</returncode>\n<output>\nclass Core: ...\n</output>",
    )
    backend = _FakeBackend(
        root,
        [
            _FakeSession(child_a, _branch_result(child_a), container_id="child-a-container"),
            _FakeSession(child_b, _branch_result(child_b), container_id="child-b-container"),
        ],
    )
    workspace_meta = {
        "root-container": {"changed_files": [], "untracked_files": [], "diff_stat": "", "current_patch_chars": 0, "workspace_fingerprint": "root"},
        "child-a-container": {"changed_files": ["pkg/core.py"], "untracked_files": [], "diff_stat": " pkg/core.py | 2 +-", "current_patch_chars": 120, "workspace_fingerprint": "a"},
        "child-b-container": {"changed_files": [], "untracked_files": [], "diff_stat": "", "current_patch_chars": 0, "workspace_fingerprint": "b"},
    }

    monkeypatch.setattr("swe_agent.trajectory_search.run_chat_with_route_async", _make_fake_chat())
    monkeypatch.setattr("swe_agent.trajectory_search._collect_workspace_meta", lambda env: workspace_meta[env.container_id])
    _patch_docker_subprocess(monkeypatch)

    runner = TrajectorySearchRunner(
        instance={"instance_id": "demo__demo-threshold", "problem_statement": "Fix the failing test."},
        backend=backend,
        run_dir=tmp_path / "run",
        policy_model_name="openai/fake",
        search_config=SearchConfig(m=2, k=1, p=2, max_rounds=1, max_active_rubrics=2),
    )
    runner.run()

    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert len(manifest["frontier_ids"]) == 2
    assert all(node_id.startswith("node-r001") for node_id in manifest["frontier_ids"])


def test_trajectory_search_runner_failed_parent_expansion_drops_parent_from_frontier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _root_snapshot()
    child_a = _branch_snapshot(
        root,
        session_id="child-a",
        step_index=0,
        assistant_text="Run the failing test.",
        command="pytest tests/test_alpha.py -q",
        observation="<returncode>0</returncode>\n<output>\n1 passed\n</output>",
    )
    child_b = _branch_snapshot(
        root,
        session_id="child-b",
        step_index=0,
        assistant_text="Read the code path.",
        command="sed -n '1,80p' pkg/core.py",
        observation="<returncode>0</returncode>\n<output>\nclass Core: ...\n</output>",
    )
    grandchild_a = _branch_snapshot(
        child_a,
        session_id="grandchild-a",
        step_index=1,
        assistant_text="Make an unfocused edit.",
        command="echo 'noise' >> notes.txt",
        observation="<returncode>0</returncode>\n<output>\n</output>",
    )
    grandchild_b = _branch_snapshot(
        child_a,
        session_id="grandchild-b",
        step_index=1,
        assistant_text="Inspect unrelated file.",
        command="sed -n '1,20p' README.md",
        observation="<returncode>0</returncode>\n<output>\nREADME\n</output>",
    )
    backend = _FakeBackend(
        root,
        [
            _FakeSession(child_a, _branch_result(child_a), container_id="child-a-container"),
            _FakeSession(child_b, _branch_result(child_b), container_id="child-b-container"),
            _FakeSession(grandchild_a, _branch_result(grandchild_a), container_id="grandchild-a-container"),
            _FakeSession(grandchild_b, _branch_result(grandchild_b), container_id="grandchild-b-container"),
        ],
    )
    workspace_meta = {
        "root-container": {"changed_files": [], "untracked_files": [], "diff_stat": "", "current_patch_chars": 0, "workspace_fingerprint": "root"},
        "child-a-container": {"changed_files": ["pkg/core.py"], "untracked_files": [], "diff_stat": " pkg/core.py | 2 +-", "current_patch_chars": 120, "workspace_fingerprint": "a"},
        "child-b-container": {"changed_files": [], "untracked_files": [], "diff_stat": "", "current_patch_chars": 0, "workspace_fingerprint": "b"},
        "grandchild-a-container": {"changed_files": ["notes.txt"], "untracked_files": ["notes.txt"], "diff_stat": " notes.txt | 1 +", "current_patch_chars": 20, "workspace_fingerprint": "ga"},
        "grandchild-b-container": {"changed_files": [], "untracked_files": [], "diff_stat": "", "current_patch_chars": 0, "workspace_fingerprint": "gb"},
    }

    def _make_regression_chat():
        async def fake_chat(route_name, model_name, user_prompt=None, system_prompt=None, **kwargs):
            judge_prompt = "\n".join(part for part in [system_prompt, user_prompt] if part)
            if user_prompt and PERSISTENT_STATE_UPDATE_PROMPT.strip() in user_prompt:
                return json.dumps(
                    {
                        "current_state": "Continue focused debugging on `pkg/core.py`.",
                        "task_specification": "Fix the failing test without unrelated edits.",
                        "files_and_functions": "- `pkg/core.py`: active target file",
                        "errors_and_corrections": "Discard unrelated file inspection.",
                        "codebase_and_system_documentation": "`pkg/core.py` contains the failing logic.",
                        "learnings": "Targeted validation is the best signal.",
                        "key_results": "Retained focused pytest evidence.",
                        "worklog": "- Compressed the earlier shared segment",
                    }
                )
            if user_prompt and SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT.strip() in user_prompt:
                return json.dumps(
                    {
                        "question": "Fix the failing test.",
                        "positive_rubrics": [
                            {
                                "title": "Validation",
                                "description": "The trajectory runs targeted validation relevant to the fix.",
                                "scale": {
                                    "1": "No validation",
                                    "2": "Incidental validation",
                                    "3": "Some relevant validation",
                                    "4": "Targeted validation",
                                    "5": "Targeted validation plus edge coverage",
                                },
                            }
                        ],
                        "negative_rubrics": [
                            {
                                "title": "Drift",
                                "description": "The trajectory makes unfocused changes without evidence.",
                                "scale": {
                                    "1": "No drift",
                                    "2": "Minor drift",
                                    "3": "Noticeable drift",
                                    "4": "Serious drift",
                                    "5": "Severe drift",
                                },
                            }
                        ],
                    }
                )
            latest_bad = "echo 'noise' >> notes.txt" in judge_prompt or "sed -n '1,20p' README.md" in judge_prompt
            latest_test = "pytest tests/test_alpha.py -q" in judge_prompt
            latest_read = "sed -n '1,80p' pkg/core.py" in judge_prompt
            is_baseline = "No continuation beyond the shared trajectory." in judge_prompt
            is_positive = "Type: positive" in judge_prompt
            if is_positive:
                if latest_bad:
                    return json.dumps({"score": 1})
                if latest_test:
                    return json.dumps({"score": 5})
                if latest_read:
                    return json.dumps({"score": 4})
                return json.dumps({"score": 1 if is_baseline else 3})
            if latest_bad:
                return json.dumps({"score": 5})
            return json.dumps({"score": 1})

        return fake_chat

    monkeypatch.setattr("swe_agent.trajectory_search.run_chat_with_route_async", _make_regression_chat())
    monkeypatch.setattr("swe_agent.trajectory_search._collect_workspace_meta", lambda env: workspace_meta[env.container_id])
    _patch_docker_subprocess(monkeypatch)

    runner = TrajectorySearchRunner(
        instance={"instance_id": "demo__demo-2", "problem_statement": "Fix the failing test."},
        backend=backend,
        run_dir=tmp_path / "run",
        policy_model_name="openai/fake",
        search_config=SearchConfig(m=1, k=1, p=2, max_rounds=2, max_active_rubrics=2),
    )
    runner.run()

    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert len(manifest["frontier_ids"]) == 1
    assert manifest["frontier_ids"][0].startswith("node-r001-s01")
    round_two = json.loads((tmp_path / "run" / "rubrics" / "round_002.json").read_text())
    assert round_two["regressed"] is True


def test_trajectory_search_final_cleanup_keeps_only_best_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _root_snapshot()
    child_a = _branch_snapshot(
        root,
        session_id="child-a",
        step_index=0,
        assistant_text="Run the failing test.",
        command="pytest tests/test_alpha.py -q",
        observation="<returncode>0</returncode>\n<output>\n1 passed\n</output>",
    )
    child_b = _branch_snapshot(
        root,
        session_id="child-b",
        step_index=0,
        assistant_text="Inspect unrelated file.",
        command="sed -n '1,20p' README.md",
        observation="<returncode>0</returncode>\n<output>\nREADME\n</output>",
    )
    backend = _FakeBackend(
        root,
        [
            _FakeSession(child_a, _branch_result(child_a, status="finished", submission="patch-a"), container_id="child-a-container"),
            _FakeSession(child_b, _branch_result(child_b, status="finished", submission="patch-b"), container_id="child-b-container"),
        ],
    )
    workspace_meta = {
        "root-container": {"changed_files": [], "untracked_files": [], "diff_stat": "", "current_patch_chars": 0, "workspace_fingerprint": "root"},
        "child-a-container": {"changed_files": ["pkg/core.py"], "untracked_files": [], "diff_stat": " pkg/core.py | 2 +-", "current_patch_chars": 120, "workspace_fingerprint": "a"},
        "child-b-container": {"changed_files": [], "untracked_files": [], "diff_stat": "", "current_patch_chars": 0, "workspace_fingerprint": "b"},
    }

    monkeypatch.setattr("swe_agent.trajectory_search.run_chat_with_route_async", _make_fake_chat())
    monkeypatch.setattr("swe_agent.trajectory_search._collect_workspace_meta", lambda env: workspace_meta[env.container_id])
    docker_state = _patch_docker_subprocess(monkeypatch)

    runner = TrajectorySearchRunner(
        instance={"instance_id": "demo__demo-3", "problem_statement": "Fix the failing test."},
        backend=backend,
        run_dir=tmp_path / "run",
        policy_model_name="openai/fake",
        search_config=SearchConfig(m=1, k=1, p=2, max_rounds=1, max_active_rubrics=2),
    )
    result = runner.run()

    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["best_node_id"] == result.best_node_id
    assert manifest["frontier_ids"] == [result.best_node_id]
    loser_nodes = [node_id for node_id in manifest["node_ids"] if node_id not in {"root", result.best_node_id}]
    assert loser_nodes
    for node_id in loser_nodes:
        loser_node = json.loads((tmp_path / "run" / "nodes" / node_id / "node.json").read_text())
        assert loser_node["snapshot_path"] is None
        assert loser_node["checkpoint_image_tag"] is None
        assert not (tmp_path / "run" / "nodes" / node_id / "snapshot.json").exists()
    assert len(docker_state["images"]) == 1
