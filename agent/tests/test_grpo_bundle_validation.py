import asyncio
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import swe_agent.trajectory_search_parallel as trajectory_search_parallel
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
from swe_agent.rubric_bank import RubricGenerationSample, RubricRecord

from swe_agent.trajectory_search_parallel import (
    ForkGroup,
    LaneBBranch,
    MidCp,
    ParallelSearchConfig,
    TrajectorySearchParallelRunner,
    _score_map_errors,
    _tie_break_errors,
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


def test_formal_submission_binary_gt_overrides_hosted_judge_reward():
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


def test_negative_hosted_judge_reward_is_clipped_to_zero():
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
    )

    exported = fork_group_to_export_group(
        instance_id="instance",
        group=group,
        steps_per_round=1,
    )

    assert exported is not None
    sample = exported.samples[0]
    assert sample.reward == 0.0
    assert sample.metadata["raw_rubric_score"] == pytest.approx(-0.35)
    assert sample.metadata["reward_source"] == "hosted_judge"


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
    assert cfg.rubric_max_tokens == 20480
    assert cfg.judge_max_tokens == 20480
    assert cfg.psu_max_tokens == 20480


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


class _FakeExperienceBank:
    async def build_generation_context(self, **_kwargs):
        return SimpleNamespace(
            extra_prompt_sections=[],
            retrieved=[],
            retrieve_messages=[],
        )


class _FakeScoreBank:
    def build_generation_context(self):
        return SimpleNamespace(
            extra_prompt_sections=[],
            existing_rubrics=[],
        )

    def update_from_model(self, *, generated):
        return SimpleNamespace(
            active_after=list(generated),
            inactive_after=[],
        )

    def set_state(self, *, active_bank, inactive_bank):
        self.active_bank = active_bank
        self.inactive_bank = inactive_bank


def _run_fake_lane_c_scope(
    monkeypatch,
    *,
    initial_format_errors=None,
    initial_terminal_error=None,
    initial_judge_errors=None,
    initial_scores=(0.25, 0.75),
    tie_break_result=None,
):
    node_ids = ["node-a", "node-b"]
    rubric = RubricRecord(
        rubric_id="rubric-1",
        title="Criterion",
        direction="positive",
        description="Distinguishes the branches.",
        scale={str(index): str(index) for index in range(1, 6)},
        weight=1.0,
        source_round=1,
    )
    generated_sample = RubricGenerationSample(
        sample_index=0,
        rubric_list_id="rubric-r001-s00",
        generated=[rubric],
        messages=[{"role": "assistant", "content": "corrected rubric"}],
        format_errors=copy.deepcopy(initial_format_errors or []),
        terminal_error=initial_terminal_error,
    )
    judge_errors = copy.deepcopy(initial_judge_errors or [])
    score_batch = {
        "generated_samples": [generated_sample],
        "sample_generated_rubrics": [[rubric]],
        "generated_score_results": {
            0: (
                [
                    [
                        {
                            "rubric_id": rubric.rubric_id,
                            "score_normalized": initial_scores[0],
                            "judge_message": [],
                        }
                    ],
                    [
                        {
                            "rubric_id": rubric.rubric_id,
                            "score_normalized": initial_scores[1],
                            "judge_message": [],
                        }
                    ],
                ],
                judge_errors,
            )
        },
    }

    async def fake_generate_and_score(**_kwargs):
        return copy.deepcopy(score_batch)

    async def fake_tie_break(**_kwargs):
        return copy.deepcopy(tie_break_result)

    monkeypatch.setattr(
        trajectory_search_parallel,
        "_generate_and_score_rubric_batch",
        fake_generate_and_score,
    )
    monkeypatch.setattr(
        trajectory_search_parallel,
        "_run_score_tie_break",
        fake_tie_break,
    )

    runner = object.__new__(TrajectorySearchParallelRunner)
    runner.config = ParallelSearchConfig(n=1)
    runner.rubric_model_name = "glm-rubric"
    runner.judge_model_name = "glm-judge"
    runner.rubric_model_kwargs = {}
    runner.judge_model_kwargs = {}
    runner.task_id = "instance"
    group = ForkGroup(
        group_index=0,
        mid_cp=MidCp(
            idx=0,
            asst_step=0,
            image_tag="root-image",
            snapshot={},
        ),
        branches=[
            LaneBBranch(
                group_index=0,
                branch_index=index,
                node_id=node_id,
                parent_image_tag="root-image",
            )
            for index, node_id in enumerate(node_ids)
        ],
    )
    result = asyncio.run(
        runner._run_lane_c_scope(
            group=group,
            scope="siblings",
            question={"system_prompt": "", "user_prompt": ""},
            shared_context={
                "previous_persistent_state": {},
                "latest_agent_trajectory": None,
            },
            previous_state={},
            latest_shared_segment=None,
            continuations=[{"node_id": node_id} for node_id in node_ids],
            score_bank=_FakeScoreBank(),
            experience_bank=_FakeExperienceBank(),
            generation_prompt="generate",
            rubric_list_prefix="rubric",
            judge_prompt="judge",
        )
    )
    return result


@pytest.mark.parametrize("status", ["success", "fallback"])
def test_tie_break_judge_error_is_fail_closed_for_every_status(status):
    errors = _tie_break_errors(
        {
            "status": status,
            "judge_errors": [
                {
                    "node_id": "node-a",
                    "rubric_id": "tie-rubric",
                    "error": "provider timeout",
                }
            ],
            "adjusted_scores": {"node-a": 0.1, "node-b": 0.2},
        },
        node_ids=["node-a", "node-b"],
    )

    assert errors
    assert any("provider timeout" in error for error in errors)


def test_tie_break_recovered_format_error_is_not_fail_closed():
    errors = _tie_break_errors(
        {
            "status": "success",
            "format_errors": ["invalid rubric json"],
            "terminal_error": "Reached max rubrics=6.",
            "judge_errors": [],
            "adjusted_scores": {"node-a": 0.1, "node-b": 0.2},
        },
        node_ids=["node-a", "node-b"],
    )

    assert errors == []


def test_tie_break_fallback_diagnostics_are_not_fail_closed():
    errors = _tie_break_errors(
        {
            "status": "fallback",
            "reason": "no_valid_tie_break_rubric",
            "format_errors": ["invalid rubric json"],
            "terminal_error": "Reached rubric generation max rounds=6.",
        },
        node_ids=["node-a", "node-b"],
    )

    assert errors == []


def test_lane_c_accepts_recovered_initial_and_tie_break_format_errors(
    monkeypatch,
):
    initial_format_errors = [
        {"turn_index": 1, "error": "invalid initial rubric json"}
    ]
    tie_format_errors = [
        {"turn_index": 1, "error": "returned {} before a rubric"}
    ]
    adjusted_scores = {"node-a": 0.500001, "node-b": 0.5}
    result = _run_fake_lane_c_scope(
        monkeypatch,
        initial_format_errors=initial_format_errors,
        initial_terminal_error="Reached max rubrics=6.",
        initial_scores=(0.5, 0.5),
        tie_break_result={
            "status": "success",
            "format_errors": tie_format_errors,
            "terminal_error": "Reached max rubrics=6.",
            "judge_errors": [],
            "judge_messages": [],
            "messages": [{"role": "assistant", "content": "corrected tie rubric"}],
            "adjusted_scores": adjusted_scores,
        },
    )

    assert result["errors"] == []
    assert result["judge_score_by_node"] == adjusted_scores
    assert result["model_response"]["format_errors"] == initial_format_errors
    assert result["model_response"]["terminal_errors"] == [
        "Reached max rubrics=6."
    ]
    sample = result["samples"][0]
    assert sample["format_errors"] == initial_format_errors
    assert sample["terminal_error"] == "Reached max rubrics=6."
    assert sample["tie_break"]["format_errors"] == tie_format_errors
    assert sample["tie_break"]["terminal_error"] == "Reached max rubrics=6."
    assert sample["tie_break_messages"] == [
        {"role": "assistant", "content": "corrected tie rubric"}
    ]


def test_lane_c_tie_break_fallback_preserves_initial_scores_and_diagnostics(
    monkeypatch,
):
    tie_format_errors = [{"turn_index": 1, "error": "invalid tie rubric"}]
    result = _run_fake_lane_c_scope(
        monkeypatch,
        initial_scores=(0.5, 0.5),
        tie_break_result={
            "status": "fallback",
            "reason": "no_valid_tie_break_rubric",
            "format_errors": tie_format_errors,
            "terminal_error": "Reached rubric generation max rounds=6.",
            "messages": [{"role": "assistant", "content": "unusable rubric"}],
        },
    )

    assert result["errors"] == []
    assert result["judge_score_by_node"] == {
        "node-a": 0.5,
        "node-b": 0.5,
    }
    sample = result["samples"][0]
    assert sample["tie_break"]["status"] == "fallback"
    assert sample["tie_break"]["format_errors"] == tie_format_errors
    assert (
        sample["tie_break"]["terminal_error"]
        == "Reached rubric generation max rounds=6."
    )


def test_lane_c_initial_judge_error_remains_fail_closed(monkeypatch):
    result = _run_fake_lane_c_scope(
        monkeypatch,
        initial_judge_errors=[
            {
                "node_id": "node-a",
                "rubric_id": "rubric-1",
                "error": "InvalidJudgeResponse",
            }
        ],
    )

    assert result["judge_score_by_node"] == {}
    assert any("InvalidJudgeResponse" in error for error in result["errors"])


def test_lane_c_nonfinite_initial_score_remains_fail_closed(monkeypatch):
    # Keep the fake judge records finite so the oracle metric helper can
    # construct its diagnostics, then inject the malformed aggregate at the
    # parallel runner's score-map boundary.  This exercises the fail-closed
    # check without asking statistics.pvariance() to accept NaN input.
    monkeypatch.setattr(
        trajectory_search_parallel,
        "_avg_scores_from_rubrics",
        lambda **_kwargs: {"node-a": float("nan"), "node-b": 0.75},
    )
    result = _run_fake_lane_c_scope(
        monkeypatch,
        initial_scores=(0.25, 0.75),
    )

    assert result["judge_score_by_node"] == {}
    assert any("non_finite:node-a" in error for error in result["errors"])


def test_tie_break_unknown_status_is_fail_closed():
    errors = _tie_break_errors(
        {"status": "partial"},
        node_ids=["node-a", "node-b"],
    )

    assert errors == ["tie_break_status:partial"]


@pytest.mark.parametrize(
    "adjusted_scores",
    [
        None,
        {"node-a": 0.1},
        {"node-a": 0.1, "node-b": 0.2, "node-c": 0.3},
        {"node-a": float("nan"), "node-b": 0.2},
        {"node-a": 0.1, "node-b": float("inf")},
    ],
)
def test_tie_break_success_with_invalid_adjusted_scores_is_fail_closed(
    adjusted_scores,
):
    errors = _tie_break_errors(
        {
            "status": "success",
            "adjusted_scores": adjusted_scores,
        },
        node_ids=["node-a", "node-b"],
    )

    assert errors


@pytest.mark.parametrize(
    ("scores", "expected_fragment"),
    [
        (None, "not_a_mapping"),
        ({"node-a": 0.1}, "coverage"),
        ({"node-a": 0.1, "node-b": 0.2, "node-c": 0.3}, "coverage"),
        ({"node-a": float("nan"), "node-b": 0.2}, "non_finite"),
        ({"node-a": 0.1, "node-b": float("inf")}, "non_finite"),
        ({"node-a": "0.1", "node-b": 0.2}, "non_numeric"),
        ({"node-a": "not-a-score", "node-b": 0.2}, "non_numeric"),
    ],
)
def test_score_map_validation_rejects_inexact_or_nonfinite_scores(
    scores, expected_fragment
):
    errors = _score_map_errors(
        scores,
        node_ids=["node-a", "node-b"],
        label="scores",
    )

    assert any(expected_fragment in error for error in errors)


def test_score_map_validation_rejects_duplicate_expected_node_ids():
    errors = _score_map_errors(
        {"node-a": 0.1},
        node_ids=["node-a", "node-a"],
        label="scores",
    )

    assert "scores:duplicate_expected_node_ids" in errors


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
