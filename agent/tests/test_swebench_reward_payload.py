import os

import pytest

from swe_agent.run.benchmarks.container_runtime import raise_for_container_error
from swe_agent.run.run_swe_agent import (
    EvaluationRewardConfig,
    _benchmark_result_payload,
    _evaluator_exception_payload,
    _r2egym_result_payload,
    make_evaluation_payload,
)


def test_payload_reward_uses_configured_soft_kind(monkeypatch):
    monkeypatch.setenv("RLER_REWARD_KIND", "soft")

    payload = make_evaluation_payload(
        "unresolved",
        passed_tests=["test_a"],
        failed_tests=["test_b"],
        fail_to_pass_expected=["test_a", "test_b"],
    )

    assert payload["reward"] == 0.5


def test_r2egym_result_does_not_override_configured_reward(monkeypatch):
    monkeypatch.setenv("RLER_REWARD_KIND", "joint")

    payload = _r2egym_result_payload(
        {
            "resolved": False,
            "reward": 1.0,
            "passed_actual": ["base_test"],
            "failed_actual": ["new_test"],
            "pass_to_pass_expected": ["base_test"],
            "fail_to_pass_expected": ["new_test"],
        }
    )

    assert payload["reward"] == 0.0
    assert set(payload) == {"status", "passed_tests", "failed_tests", "reward", "metainfo"}


def test_aggregate_benchmark_without_test_sets_uses_status(monkeypatch):
    monkeypatch.setenv("RLER_REWARD_KIND", "f2p_only")

    payload = _benchmark_result_payload(
        {
            "resolved": False,
            "reward": 1.0,
            "passed_actual": [],
            "failed_actual": [],
            "pass_to_pass_expected": [],
            "fail_to_pass_expected": [],
        }
    )

    assert payload["reward"] == 0.0


def test_reward_kind_env_is_not_modified(monkeypatch):
    monkeypatch.setenv("RLER_REWARD_KIND", "f2p_only")
    make_evaluation_payload(
        "resolved",
        passed_tests=["new_test"],
        fail_to_pass_expected=["new_test"],
    )
    assert os.environ["RLER_REWARD_KIND"] == "f2p_only"


def test_joint_reward_penalizes_scaled_p2p_failure_rate():
    payload = make_evaluation_payload(
        "unresolved",
        passed_tests=["new_a", "old_a", "old_b", "old_c"],
        failed_tests=["new_b", "old_d"],
        fail_to_pass_expected=["new_a", "new_b"],
        pass_to_pass_expected=["old_a", "old_b", "old_c", "old_d"],
        reward_config=EvaluationRewardConfig(kind="joint", joint_alpha=0.5),
    )

    assert payload["reward"] == 0.375


def test_delta_reward_kind_is_not_supported():
    with pytest.raises(ValueError, match="Unsupported reward kind"):
        EvaluationRewardConfig(kind="delta")


def test_all_pass_multiplier_and_no_action_adjustment():
    all_pass = make_evaluation_payload(
        "resolved",
        passed_tests=["new", "old"],
        fail_to_pass_expected=["new"],
        pass_to_pass_expected=["old"],
        reward_config=EvaluationRewardConfig(kind="joint", all_pass_reward=2.0),
    )
    no_action = make_evaluation_payload(
        "unresolved",
        passed_tests=["new", "old"],
        fail_to_pass_expected=["new"],
        pass_to_pass_expected=["old"],
        reward_config=EvaluationRewardConfig(
            kind="joint", no_action_patch_penalty=-0.1
        ),
    )
    fallback = make_evaluation_payload(
        "unresolved",
        passed_tests=["new", "old"],
        fail_to_pass_expected=["new"],
        pass_to_pass_expected=["old"],
        reward_config=EvaluationRewardConfig(
            kind="joint", fallback_patch_penalty=0.5
        ),
    )

    assert all_pass["reward"] == 2.0
    assert no_action["reward"] == 0.9
    assert fallback["reward"] == 0.5


def test_empty_and_error_rewards_are_fixed():
    reward_config = EvaluationRewardConfig(
        kind="joint",
        fallback_patch_penalty=0.5,
        no_action_patch_penalty=-0.1,
    )

    assert make_evaluation_payload("empty", reward_config=reward_config)["reward"] == -0.2
    assert make_evaluation_payload("error", reward_config=reward_config)["reward"] == -0.5


def test_infrastructure_error_is_explicit():
    payload = make_evaluation_payload(
        "error",
        error="container failed",
        infrastructure_error=True,
    )

    assert payload["reward"] == -0.5
    assert payload["metainfo"]["infrastructure_error"] is True


def test_evaluator_reported_error_has_fixed_reward_not_infrastructure_error():
    payload = _benchmark_result_payload(
        {"error": "patch broke test script", "evaluation_output": "failed"}
    )

    assert payload["status"] == "error"
    assert payload["reward"] == -0.5
    assert payload["metainfo"]["infrastructure_error"] is False


def test_evaluator_timeout_has_fixed_reward_not_infrastructure_error():
    payload = _evaluator_exception_payload(
        RuntimeError("evaluation command timed out after 600 seconds")
    )

    assert payload["status"] == "error"
    assert payload["reward"] == -0.5
    assert payload["metainfo"]["infrastructure_error"] is False


def test_non_timeout_evaluator_exception_is_infrastructure_error():
    payload = _evaluator_exception_payload(RuntimeError("container image is missing"))

    assert payload["status"] == "error"
    assert payload["reward"] == -0.5
    assert payload["metainfo"]["infrastructure_error"] is True


def test_container_runtime_error_is_distinct_from_command_failure():
    raise_for_container_error({"returncode": 1, "exception_info": ""})

    with pytest.raises(RuntimeError, match="container failed"):
        raise_for_container_error(
            {"returncode": -1, "exception_info": "container failed"}
        )
