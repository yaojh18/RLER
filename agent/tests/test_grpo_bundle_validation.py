import asyncio
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from swe_agent.lane_to_grpo_bundle import fork_group_to_export_group
from swe_agent.exceptions import PolicyVersionMismatch
from swe_agent.naive_search import (
    NaiveRecord,
    NaiveRollout,
    NaiveSearchConfig,
    NaiveSearchRunner,
)
from swe_agent.naive_to_grpo_bundle import naive_record_to_bundle
from swe_agent.parallel_utils import (
    exact_rollout_token_error,
    extract_terminal_patch_from_session,
    normalize_terminal_patch_text,
)
from swe_agent.trajectory_search_parallel import (
    ForkGroup,
    LaneBBranch,
    MidCp,
    ParallelSearchConfig,
    TrajectorySearchParallelRunner,
    _unexpected_rollout_exit,
)


def _messages(*, valid: bool = True) -> list[dict]:
    assistant = {
        "role": "assistant",
        "content": "work",
        "prompt_token_ids": [1, 2],
        "token_ids": [3],
        "logprobs": [-0.1],
    }
    if not valid:
        assistant.pop("logprobs")
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "problem"},
        assistant,
    ]


def test_naive_dump_preserves_exhausted_rollout_error(tmp_path):
    runner = object.__new__(NaiveSearchRunner)
    runner.run_dir = tmp_path
    runner.task_id = "django__django-1"
    runner.policy_model_name = "Qwen/Qwen3.5-9B"
    runner.config = NaiveSearchConfig(evaluate_gt=True)
    runner._rollout_run_dir = lambda rollout_index: tmp_path / f"rollout_{rollout_index:04d}"
    rollout_error = "rollout_retry_exhausted after 8 attempts: connection error"
    record = NaiveRecord(
        instance_id="django__django-1",
        run_dir=str(tmp_path),
        task_id="django__django-1",
        config={},
        rollouts=[
            NaiveRollout(
                rollout_index=0,
                node_id="naive-django__django-1-r00-test",
                status="error",
                error=rollout_error,
            )
        ],
    )

    runner._dump_record(record)

    payload = json.loads(
        (tmp_path / "rollout_0000" / "evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "error"
    assert payload["metainfo"]["infrastructure_error"] is True
    assert rollout_error in payload["metainfo"]["output"]


def test_terminal_patch_normalization_preserves_diff_and_strips_only_cwd_warning():
    warning = (
        "WARNING: Error changing the container working directory. Using '/root' "
        "instead: chdir /workspace: no such file or directory"
    )
    patch = "diff --git a/a b/a\n@@ -1 +1 @@\n-a\n+b\n "

    assert normalize_terminal_patch_text(patch) == patch + "\n"
    assert normalize_terminal_patch_text(warning) == ""
    assert normalize_terminal_patch_text(f"{warning}\n{patch}\n") == patch + "\n"

    env = SimpleNamespace(
        execute=lambda *_args, **_kwargs: {"output": warning, "returncode": 0}
    )
    extracted, is_fallback = extract_terminal_patch_from_session(
        {"submission": ""}, SimpleNamespace(agent=SimpleNamespace(env=env))
    )
    assert extracted == ""
    assert is_fallback is True


def test_zero_token_abort_is_a_policy_branch_error():
    messages = _messages()
    messages[-1] = {
        "role": "assistant",
        "content": "",
        "finish_reason": "abort",
        "prompt_token_ids": [1, 2],
        "token_ids": [],
        "logprobs": [],
    }

    error = exact_rollout_token_error(messages)

    assert error is not None
    assert "invalid_policy_completion" in error
    assert "finish_reason=abort" in error


def _load_smoke_verifier():
    script = (
        Path(__file__).resolve().parents[3]
        / "exp"
        / "scripts"
        / "verify_rler_earlypred_smoke.py"
    )
    spec = importlib.util.spec_from_file_location(
        "verify_rler_earlypred_smoke_for_test",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_invalid_naive_rollout_drops_complete_group():
    record = NaiveRecord(
        instance_id="instance",
        run_dir="run",
        task_id="instance",
        config={},
        rollouts=[
            NaiveRollout(
                rollout_index=0,
                node_id="valid",
                messages=_messages(),
                gt_score=0.5,
            ),
            NaiveRollout(
                rollout_index=1,
                node_id="invalid",
                messages=_messages(valid=False),
                gt_score=0.0,
            ),
        ],
    )

    bundle = naive_record_to_bundle(record)

    assert bundle.policy_groups == []
    assert bundle.metadata["invalid_rollouts"] == ["invalid"]


def test_naive_all_pass_reward_defaults_to_unit_scale():
    assert NaiveSearchConfig().all_pass_reward == 1.0
    assert NaiveSearchConfig().gt_eval_timeout == 600


class _PolicyVersionCaptureBackend:
    def __init__(self):
        self.agent_config = {"step_limit": 250}
        self.spec = None

    def create_session(self, spec):
        self.spec = spec
        return SimpleNamespace(
            agent=SimpleNamespace(
                model=SimpleNamespace(
                    config=SimpleNamespace(
                        model_kwargs={},
                        model_name="Qwen/Qwen3.5-9B",
                    )
                )
            )
        )


@pytest.mark.parametrize(
    ("enforce_policy_version", "expected_session_version"),
    [(False, None), (True, "checkpoint-0000007")],
)
def test_naive_request_version_guard_is_validation_only(
    enforce_policy_version,
    expected_session_version,
):
    runner = object.__new__(NaiveSearchRunner)
    runner.task = "fix"
    runner.task_id = "django__django-1"
    runner.instance = {"patch": "gold"}
    runner.policy_model_name = "Qwen/Qwen3.5-9B"
    runner.policy_version = "checkpoint-0000007"
    runner.enforce_policy_version = enforce_policy_version
    runner.backend = _PolicyVersionCaptureBackend()
    runner.policy_base_urls = ["http://127.0.0.1:30000"]
    runner.api_key = "EMPTY"
    runner.config = NaiveSearchConfig()
    runner._live_sessions = []

    runner._make_session(0)

    assert runner.backend.spec.policy_version == expected_session_version
    # The logical dispatch version remains available for exported sample
    # metadata even when training does not enforce it request-by-request.
    assert runner.policy_version == "checkpoint-0000007"


def test_policy_mismatch_uses_normal_session_cleanup():
    runner = object.__new__(NaiveSearchRunner)
    runner.task_id = "django__django-1"
    runner.config = NaiveSearchConfig(step_limit=250)
    cleanup_calls = []

    class Snapshot:
        def model_dump(self, mode="json"):
            return {
                "metadata": {
                    "model_turns": [],
                    "events": [],
                }
            }

    session = SimpleNamespace(snapshot=lambda: Snapshot())
    runner._make_session = lambda rollout_index: session

    def stale_step(session, max_steps):
        raise PolicyVersionMismatch("request_complete: stale validation")

    runner._step_session = stale_step
    runner._cleanup_session = lambda session: cleanup_calls.append(session)

    rollout = runner._run_one_rollout(0)

    assert rollout.status == "error"
    assert "PolicyVersionMismatch" in rollout.error
    assert cleanup_calls == [session]


@pytest.mark.parametrize(
    ("exit_status", "expected_patch"),
    [
        ("CompletionLengthExceeded", "fallback"),
        ("ContextWindowExceeded", "fallback"),
        ("", "fallback"),
        ("Submitted", "fallback"),
    ],
)
def test_naive_extracts_terminal_or_workspace_patch(
    monkeypatch, exit_status, expected_patch
):
    runner = object.__new__(NaiveSearchRunner)
    runner.task_id = "django__django-1"
    runner.config = NaiveSearchConfig(step_limit=250)
    snapshot = {
        "metadata": {"model_turns": [], "events": []},
        "agent": {"state": {"messages": _messages()}},
    }
    session = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(model_dump=lambda mode="json": snapshot)
    )
    runner._make_session = lambda rollout_index: session
    runner._step_session = lambda session, max_steps: {
        "status": "finished",
        "exit_status": exit_status,
        "submission": "",
    }
    runner._cleanup_session = lambda session: None
    extraction_calls = []
    monkeypatch.setattr(
        "swe_agent.naive_search.extract_terminal_patch_from_session",
        lambda result, session: extraction_calls.append((result, session)) or ("fallback", True),
    )

    rollout = runner._run_one_rollout(0)

    assert len(extraction_calls) == 1
    assert rollout.terminated_early is (exit_status == "Submitted")
    assert rollout.terminal_patch == expected_patch
    assert rollout.terminal_patch_from_fallback is bool(expected_patch)


def test_naive_gt_creates_rollout_work_dir_before_evaluation(
    monkeypatch, tmp_path
):
    runner = object.__new__(NaiveSearchRunner)
    runner.run_dir = tmp_path / "not-created-yet"
    runner.run_timestamp = "20260728-120000"
    runner.task_id = "django__django-1"
    runner.instance = {
        "instance_id": runner.task_id,
        "problem_statement": "test",
    }
    runner.policy_model_name = "Qwen/Qwen3.6-27B"
    runner.harness_namespace = "swebench"
    runner.config = NaiveSearchConfig(gt_eval_timeout=1800)
    rollout = NaiveRollout(
        rollout_index=0,
        node_id="node-0",
        terminal_patch="diff --git a/a b/a\n",
        n_action_steps=1,
    )

    def fake_evaluate(**kwargs):
        assert kwargs["timeout"] == 1800
        work_dir = kwargs["work_dir"]
        assert work_dir.is_dir()
        key = next(iter(kwargs["patches_by_key"]))
        return {
            key: {
                "status": "unresolved",
                "reward": 0.0,
                "metainfo": {"infrastructure_error": False},
            }
        }

    monkeypatch.setattr(
        "swe_agent.naive_search.evaluate_swebench_instance_patches",
        fake_evaluate,
    )

    runner._evaluate_gt(rollout)

    assert rollout.gt_score == 0.0
    assert rollout.evaluation_payload["status"] == "unresolved"


def test_naive_retries_only_failed_rollout_before_completing_group(tmp_path):
    runner = object.__new__(NaiveSearchRunner)
    runner.task_id = "django__django-1"
    runner.run_dir = tmp_path
    runner.policy_base_urls = ["http://127.0.0.1:30000"]
    runner.config = NaiveSearchConfig(
        m=1,
        rollout_pool_size=1,
        gt_eval_workers=1,
        rollout_max_attempts=3,
    )
    runner._live_sessions = []
    attempts = []
    evaluations = []
    events = []

    def run_one(rollout_index):
        attempts.append(rollout_index)
        return NaiveRollout(
            rollout_index=rollout_index,
            node_id=f"node-{len(attempts)}",
            error="transient sglang error" if len(attempts) == 1 else None,
        )

    runner._run_one_rollout = run_one

    def evaluate(rollout):
        evaluations.append(rollout.rollout_index)
        events.append("evaluate")
        rollout.gt_score = 0.5 if len(evaluations) == 2 else None

    runner._evaluate_gt = evaluate
    runner._dump_record = lambda record: None
    runner._cleanup_all = lambda: None

    def release(rollout_index):
        events.append(f"release-{rollout_index}")

    try:
        record = asyncio.run(
            asyncio.wait_for(
                runner.run(on_rollout_done=release),
                timeout=10,
            )
        )
    except asyncio.TimeoutError:
        pytest.fail(
            f"naive retry deadlocked: attempts={attempts} "
            f"evaluations={evaluations}"
        )

    assert attempts == [0, 0]
    assert evaluations == [0, 0]
    assert events == ["release-0", "evaluate", "evaluate"]
    assert record.completed is True
    assert record.rollouts[0].gt_score == 0.5


def test_invalid_lane_branch_drops_complete_group():
    mid_cp = MidCp(
        idx=0,
        asst_step=1,
        image_tag="image",
        snapshot={"agent": {"state": {"messages": _messages()[:2]}}},
    )
    group = ForkGroup(
        group_index=0,
        mid_cp=mid_cp,
        branches=[
            LaneBBranch(
                group_index=0,
                branch_index=0,
                node_id="valid",
                parent_image_tag="image",
                messages=_messages()[2:],
                gt_score=0.5,
            ),
            LaneBBranch(
                group_index=0,
                branch_index=1,
                node_id="invalid",
                parent_image_tag="image",
                messages=_messages()[2:] + _messages(valid=False)[2:],
                gt_score=0.0,
            ),
        ],
    )

    assert fork_group_to_export_group(
        instance_id="instance",
        group=group,
        steps_per_round=1,
        gt_only_reward=True,
    ) is None


def test_judge_reward_uses_real_node_id():
    mid_cp = MidCp(
        idx=0,
        asst_step=0,
        image_tag="root-image",
        snapshot={"agent": {"state": {"messages": _messages()[:2]}}},
    )
    branch = LaneBBranch(
        group_index=0,
        branch_index=0,
        node_id="runtime-node-a1b2c3",
        parent_image_tag="root-image",
        messages=_messages()[2:],
    )
    group = ForkGroup(
        group_index=0,
        mid_cp=mid_cp,
        branches=[branch],
        judge_score_by_node={"runtime-node-a1b2c3": 0.75},
    )

    exported = fork_group_to_export_group(
        instance_id="instance",
        group=group,
        steps_per_round=1,
    )

    assert exported is not None
    assert exported.samples[0].reward == 0.75
    assert exported.samples[0].metadata["parent_assistant_spans"] == []
    assert exported.samples[0].metadata["child_assistant_spans"] == [[0, 1]]


def test_explicit_gt_on_submit_reward_overrides_direct_judge_reward():
    mid_cp = MidCp(
        idx=0,
        asst_step=0,
        image_tag="root-image",
        snapshot={"agent": {"state": {"messages": _messages()[:2]}}},
    )
    branch = LaneBBranch(
        group_index=0,
        branch_index=0,
        node_id="submitted-node",
        parent_image_tag="root-image",
        messages=_messages()[2:],
        terminated_early=True,
        gt_score=1.0,
    )
    group = ForkGroup(
        group_index=0,
        mid_cp=mid_cp,
        branches=[branch],
        judge_score_by_node={"submitted-node": 0.15},
        training_reward_by_node={"submitted-node": 1.0},
        reward_source_by_node={
            "submitted-node": "terminal_swebench_binary"
        },
    )

    exported = fork_group_to_export_group(
        instance_id="instance",
        group=group,
        steps_per_round=1,
    )

    assert exported is not None
    assert exported.samples[0].reward == 1.0
    assert (
        exported.samples[0].metadata["reward_source"]
        == "terminal_swebench_binary"
    )
    assert exported.samples[0].metadata["raw_rubric_score"] == 0.15


def test_direct_reward_remains_default_for_submitted_branch():
    mid_cp = MidCp(
        idx=0,
        asst_step=0,
        image_tag="root-image",
        snapshot={"agent": {"state": {"messages": _messages()[:2]}}},
    )
    branch = LaneBBranch(
        group_index=0,
        branch_index=0,
        node_id="submitted-node",
        parent_image_tag="root-image",
        messages=_messages()[2:],
        terminated_early=True,
        gt_score=1.0,
    )
    group = ForkGroup(
        group_index=0,
        mid_cp=mid_cp,
        branches=[branch],
        judge_score_by_node={"submitted-node": 0.15},
        training_reward_by_node={"submitted-node": 0.15},
        reward_source_by_node={
            "submitted-node": "direct_golden_rubric_judge"
        },
    )

    exported = fork_group_to_export_group(
        instance_id="instance",
        group=group,
        steps_per_round=1,
    )

    assert exported is not None
    assert exported.samples[0].reward == 0.15
    assert (
        exported.samples[0].metadata["reward_source"]
        == "direct_golden_rubric_judge"
    )


def test_empty_formal_submission_has_binary_zero_reward():
    runner = object.__new__(TrajectorySearchParallelRunner)
    runner.config = ParallelSearchConfig()
    runner.task_id = "instance"
    branch = LaneBBranch(
        group_index=0,
        branch_index=0,
        node_id="empty-submission",
        parent_image_tag="root-image",
        terminated_early=True,
        terminal_patch="",
    )

    runner._evaluate_gt(branch)

    assert branch.gt_score == 0.0
    assert branch.gt_payload["status"] == "unresolved"


def test_thinking_output_tokens_remain_trainable_in_exact_lane_bundle():
    mid_cp = MidCp(
        idx=0,
        asst_step=0,
        image_tag="root-image",
        snapshot={"agent": {"state": {"messages": _messages()[:2]}}},
    )
    branch = LaneBBranch(
        group_index=0,
        branch_index=0,
        node_id="thinking-node",
        parent_image_tag="root-image",
        messages=[
            {
                "role": "assistant",
                "content": "<think>reasoning</think>action",
                "prompt_token_ids": [1, 2],
                "token_ids": [10, 11, 12, 13],
                "logprobs": [-0.1, -0.2, -0.3, -0.4],
            }
        ],
    )
    group = ForkGroup(
        group_index=0,
        mid_cp=mid_cp,
        branches=[branch],
        judge_score_by_node={"thinking-node": 0.5},
    )

    exported = fork_group_to_export_group(
        instance_id="instance",
        group=group,
        steps_per_round=1,
    )

    assert exported is not None
    sample = exported.samples[0]
    assert sample.token_ids == [1, 2, 10, 11, 12, 13]
    assert sample.loss_mask[-sample.response_length :] == [1, 1, 1, 1]
    assert sample.rollout_logprobs == [-0.1, -0.2, -0.3, -0.4]


def test_policy_overlength_uses_unscaled_hosted_judge_reward():
    mid_cp = MidCp(
        idx=0,
        asst_step=0,
        image_tag="root-image",
        snapshot={"agent": {"state": {"messages": _messages()[:2]}}},
    )
    branch = LaneBBranch(
        group_index=0,
        branch_index=0,
        node_id="overflow-node",
        parent_image_tag="root-image",
        messages=_messages()[2:],
        overlength_reason="ContextWindowExceeded",
    )
    group = ForkGroup(
        group_index=0,
        mid_cp=mid_cp,
        branches=[branch],
        judge_score_by_node={"overflow-node": 0.8},
    )

    exported = fork_group_to_export_group(
        instance_id="instance",
        group=group,
        steps_per_round=1,
    )

    assert exported is not None
    sample = exported.samples[0]
    assert sample.reward == pytest.approx(0.8)
    assert sample.metadata["raw_rubric_score"] == pytest.approx(0.8)
    assert "reward_penalty" not in sample.metadata
    assert (
        sample.metadata["policy_overlength_reason"]
        == "ContextWindowExceeded"
    )


def test_negative_collapse_reward_is_preserved():
    mid_cp = MidCp(
        idx=0,
        asst_step=0,
        image_tag="root-image",
        snapshot={"agent": {"state": {"messages": _messages()[:2]}}},
    )
    branch = LaneBBranch(
        group_index=0,
        branch_index=0,
        node_id="negative-judge-node",
        parent_image_tag="root-image",
        messages=_messages()[2:],
    )
    group = ForkGroup(
        group_index=0,
        mid_cp=mid_cp,
        branches=[branch],
        judge_score_by_node={"negative-judge-node": -0.35},
        training_reward_by_node={"negative-judge-node": -0.35},
        reward_source_by_node={"negative-judge-node": "code_mode_collapse"},
    )

    exported = fork_group_to_export_group(
        instance_id="instance",
        group=group,
        steps_per_round=1,
    )

    assert exported is not None
    sample = exported.samples[0]
    assert sample.reward == pytest.approx(-0.35)
    assert sample.metadata["raw_rubric_score"] == pytest.approx(-0.35)
    assert sample.metadata["reward_source"] == "code_mode_collapse"


def test_beam_parent_assistant_tokens_are_trainable_from_root_prompt():
    root_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "problem"},
    ]
    parent_messages = root_messages + [
        {
            "role": "assistant",
            "content": "parent work",
            "prompt_token_ids": [1, 2],
            "token_ids": [3, 30],
            "logprobs": [-0.1, -0.11],
        },
        {"role": "tool", "content": "observation"},
    ]
    root = MidCp(
        idx=0,
        asst_step=0,
        image_tag="root-image",
        snapshot={"agent": {"state": {"messages": root_messages}}},
    )
    parent = MidCp(
        idx=1,
        asst_step=1,
        image_tag="parent-image",
        node_id="parent-real-id",
        snapshot={"agent": {"state": {"messages": parent_messages}}},
    )
    child = LaneBBranch(
        group_index=1,
        branch_index=0,
        node_id="child-real-id",
        parent_image_tag="parent-image",
        parent_asst_step=1,
        beam_parent_index=0,
        beam_parent_node_id="parent-real-id",
        messages=[
            {
                "role": "assistant",
                "content": "child work",
                "prompt_token_ids": [1, 2, 3, 30, 4],
                "token_ids": [5, 50],
                "logprobs": [-0.2, -0.21],
            }
        ],
    )
    group = ForkGroup(
        group_index=1,
        mid_cp=root,
        group_kind="beam",
        parent_mid_cps=[parent],
        branches=[child],
        judge_score_by_node={"child-real-id": 0.6},
    )

    exported = fork_group_to_export_group(
        instance_id="instance",
        group=group,
        steps_per_round=1,
    )

    assert exported is not None
    sample = exported.samples[0]
    assert sample.prompt == root_messages
    assert sample.turns == [
        {"role": "assistant", "content": "parent work"},
        {"role": "tool", "content": "observation"},
        {"role": "assistant", "content": "child work"},
    ]
    assert sample.token_ids == [1, 2, 3, 30, 4, 5, 50]
    assert sample.response_length == 5
    assert sample.loss_mask[-sample.response_length:] == [1, 1, 0, 1, 1]
    assert sample.metadata["group_kind"] == "beam"
    assert sample.metadata["parent_assistant_spans"] == [[0, 2]]
    assert sample.metadata["child_assistant_spans"] == [[3, 5]]

    verifier = _load_smoke_verifier()
    serialized = {
        "tokens": list(sample.token_ids),
        "response_length": sample.response_length,
        "loss_mask": list(sample.loss_mask[-sample.response_length:]),
        "metadata": copy.deepcopy(sample.metadata),
    }
    assert verifier._validate_response_assistant_spans(
        serialized,
        expected_parent_turns=1,
        expected_child_turns=1,
        label="beam sample",
    ) == {
        "parent_turns": 1,
        "child_turns": 1,
        "masked_tokens": 4,
    }

    parent_start, parent_end = sample.metadata["parent_assistant_spans"][0]
    for response_index in range(parent_start, parent_end):
        corrupted = copy.deepcopy(serialized)
        corrupted["loss_mask"][response_index] = 0
        with pytest.raises(RuntimeError, match="unmasked token"):
            verifier._validate_response_assistant_spans(
                corrupted,
                expected_parent_turns=1,
                expected_child_turns=1,
                label="beam sample",
            )


def test_beam_overlength_before_first_child_turn_trains_parent_only():
    root_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "problem"},
    ]
    parent_messages = root_messages + [
        {
            "role": "assistant",
            "content": "parent work",
            "prompt_token_ids": [1, 2],
            "token_ids": [3, 30],
            "logprobs": [-0.1, -0.11],
        },
        {"role": "tool", "content": "observation"},
    ]
    root = MidCp(
        idx=0,
        asst_step=0,
        image_tag="root-image",
        snapshot={"agent": {"state": {"messages": root_messages}}},
    )
    parent = MidCp(
        idx=1,
        asst_step=1,
        image_tag="parent-image",
        node_id="parent-real-id",
        snapshot={"agent": {"state": {"messages": parent_messages}}},
    )
    child = LaneBBranch(
        group_index=1,
        branch_index=0,
        node_id="child-overlength-id",
        parent_image_tag="parent-image",
        parent_asst_step=1,
        beam_parent_index=0,
        beam_parent_node_id="parent-real-id",
        messages=[],
        status="policy_overlength",
        overlength_reason="ContextWindowExceeded",
    )
    group = ForkGroup(
        group_index=1,
        mid_cp=root,
        group_kind="beam",
        parent_mid_cps=[parent],
        branches=[child],
        judge_score_by_node={"child-overlength-id": 0.4},
    )

    exported = fork_group_to_export_group(
        instance_id="instance",
        group=group,
        steps_per_round=1,
    )

    assert exported is not None
    sample = exported.samples[0]
    assert sample.turns == [
        {"role": "assistant", "content": "parent work"},
        {"role": "tool", "content": "observation"},
    ]
    assert sample.reward == pytest.approx(0.4)
    assert sample.metadata["parent_assistant_spans"] == [[0, 2]]
    assert sample.metadata["child_assistant_spans"] == []
    assert sample.metadata["policy_overlength_reason"] == (
        "ContextWindowExceeded"
    )


def test_assistant_output_overflow_is_fail_closed():
    root_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "problem"},
    ]
    root = MidCp(
        idx=0,
        asst_step=0,
        image_tag="root-image",
        snapshot={"agent": {"state": {"messages": root_messages}}},
    )
    branch = LaneBBranch(
        group_index=0,
        branch_index=0,
        node_id="overflow-node",
        parent_image_tag="root-image",
        messages=[
            {
                "role": "assistant",
                "content": "earlier output",
                "prompt_token_ids": [1, 2],
                "token_ids": [3, 4, 5],
                "logprobs": [-0.1, -0.2, -0.3],
            },
            {
                "role": "assistant",
                "content": "later output",
                # Malformed stored prefix retains only one of the earlier
                # output tokens, so the earlier response would overflow the
                # assembled full sequence if it were clamped.
                "prompt_token_ids": [1, 2, 3],
                "token_ids": [6],
                "logprobs": [-0.4],
            },
        ],
    )
    group = ForkGroup(
        group_index=0,
        mid_cp=root,
        branches=[branch],
        judge_score_by_node={"overflow-node": 0.5},
    )

    with pytest.raises(AssertionError, match="invalid assistant response span"):
        fork_group_to_export_group(
            instance_id="instance",
            group=group,
            steps_per_round=2,
        )


def test_beam_parent_output_token_misalignment_is_fail_closed():
    root_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "problem"},
    ]
    parent_messages = root_messages + [
        {
            "role": "assistant",
            "content": "parent work",
            "prompt_token_ids": [1, 2],
            "token_ids": [3],
            "logprobs": [-0.1],
        }
    ]
    root = MidCp(
        idx=0,
        asst_step=0,
        image_tag="root-image",
        snapshot={"agent": {"state": {"messages": root_messages}}},
    )
    parent = MidCp(
        idx=1,
        asst_step=1,
        image_tag="parent-image",
        node_id="parent-real-id",
        snapshot={"agent": {"state": {"messages": parent_messages}}},
    )
    child = LaneBBranch(
        group_index=1,
        branch_index=0,
        node_id="child-misaligned",
        parent_image_tag="parent-image",
        parent_asst_step=1,
        beam_parent_index=0,
        beam_parent_node_id="parent-real-id",
        messages=[
            {
                "role": "assistant",
                "content": "child work",
                # The prompt has the right length and parent prompt prefix,
                # but does not contain the parent's stored output token 3.
                "prompt_token_ids": [1, 2, 99],
                "token_ids": [4],
                "logprobs": [-0.2],
            }
        ],
    )
    group = ForkGroup(
        group_index=1,
        mid_cp=root,
        group_kind="beam",
        parent_mid_cps=[parent],
        branches=[child],
        judge_score_by_node={"child-misaligned": 0.5},
    )

    with pytest.raises(AssertionError, match="output tokens are misaligned"):
        fork_group_to_export_group(
            instance_id="instance",
            group=group,
            steps_per_round=1,
        )


def test_depth2_config_is_two_parents_and_two_size_eight_groups():
    cfg = ParallelSearchConfig(topology="depth2")

    assert cfg.m == 8
    assert cfg.p == 2
    assert cfg.m // cfg.p == 4
    assert cfg.terminal_rollout is False
    assert cfg.all_pass_reward == 1.0
    assert cfg.judge_max_tokens == 20480
    assert cfg.judge_context_length == 256000
    assert cfg.collapse_reward_margin is None


def test_depth2_uses_only_nonterminal_sampled_parents_without_resampling():
    first = MidCp(
        idx=1,
        asst_step=20,
        image_tag="first",
        terminated_early=False,
    )
    second = MidCp(
        idx=2,
        asst_step=8,
        image_tag="",
        terminated_early=True,
    )

    assert first.can_continue
    assert not second.can_continue
    first.terminated_early = True
    assert not first.can_continue
    first.terminated_early = False
    second.terminated_early = False
    assert first.can_continue
    assert second.can_continue

    first.continuation_unavailable_reason = (
        "policy_overlength:CompletionLengthExceeded"
    )
    assert not first.can_continue
    assert second.can_continue


@pytest.mark.parametrize(
    ("enforce_policy_version", "expected_session_version"),
    [(False, None), (True, "checkpoint-0000007")],
)
def test_lanes_request_version_guard_is_opt_in(
    enforce_policy_version,
    expected_session_version,
):
    runner = object.__new__(TrajectorySearchParallelRunner)
    runner.task = "fix"
    runner.task_id = "django__django-1"
    runner.instance = {"patch": "gold"}
    runner.policy_model_name = "Qwen/Qwen3.5-9B"
    runner.policy_version = "checkpoint-0000007"
    runner.enforce_policy_version = enforce_policy_version
    runner.backend = _PolicyVersionCaptureBackend()
    runner.policy_base_url = "http://127.0.0.1:30000"
    runner.policy_api_key = "EMPTY"
    runner.config = ParallelSearchConfig()
    runner._live_sessions = []

    runner._make_initial_session()

    assert runner.backend.spec.policy_version == expected_session_version
    assert runner.policy_version == "checkpoint-0000007"


def test_depth2_rejects_non_divisible_parent_fanout():
    with pytest.raises(ValueError, match="m divisible by p"):
        ParallelSearchConfig(topology="depth2", m=8, p=3)


def test_usage_group_ids_separate_root_and_beam_cost():
    runner = object.__new__(TrajectorySearchParallelRunner)
    runner.usage_group_prefix = "train:r17:instance"

    assert runner._usage_group_id(0) == "train:r17:instance:g0"
    assert runner._usage_group_id(1) == "train:r17:instance:g1"


def test_policy_overlength_clean_returns_are_expected_terminals():
    for exit_status in (
        "ContextWindowExceeded",
        "CompletionLengthExceeded",
    ):
        assert (
            _unexpected_rollout_exit(
                {"status": "finished", "exit_status": exit_status},
                lane="lane_b",
            )
            is None
        )


def test_other_clean_limit_or_runtime_returns_are_errors():
    for exit_status in (
        "LimitsExceeded",
        "RuntimeError",
    ):
        error = _unexpected_rollout_exit(
            {"status": "finished", "exit_status": exit_status},
            lane="lane_b",
        )
        assert error == f"lane_b_exit_status:{exit_status}"


def test_lane_a_parent_clean_return_with_overflow_is_expected_terminal():
    assert (
        _unexpected_rollout_exit(
            {
                "status": "finished",
                "exit_status": "ContextWindowExceeded",
            },
            lane="lane_a_parent_1",
        )
        is None
    )


def test_strict_k_pause_and_submitted_are_not_exit_errors():
    assert (
        _unexpected_rollout_exit(
            {"status": "paused", "exit_status": ""}, lane="lane_b"
        )
        is None
    )
    assert (
        _unexpected_rollout_exit(
            {"status": "finished", "exit_status": "Submitted"},
            lane="lane_a_parent_0",
        )
        is None
    )
