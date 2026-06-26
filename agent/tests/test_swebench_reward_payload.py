import os

from swe_agent.run.run_swe_agent import (
    _benchmark_result_payload,
    _r2egym_result_payload,
    make_evaluation_payload,
)


def test_payload_reward_uses_configured_soft_scheme(monkeypatch):
    monkeypatch.setenv("RLER_REWARD_SCHEME", "soft")

    payload = make_evaluation_payload(
        "unresolved",
        passed_tests=["test_a"],
        failed_tests=["test_b"],
        fail_to_pass_expected=["test_a", "test_b"],
    )

    assert payload["reward"] == 0.5


def test_r2egym_result_does_not_override_configured_reward(monkeypatch):
    monkeypatch.setenv("RLER_REWARD_SCHEME", "joint")

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
    assert payload["metainfo"]["benchmark_reward"] == 1.0


def test_aggregate_benchmark_without_test_sets_falls_back_to_status(monkeypatch):
    monkeypatch.setenv("RLER_REWARD_SCHEME", "f2p_only")

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
    assert payload["metainfo"]["benchmark_reward"] == 1.0


def test_reward_scheme_env_is_not_modified(monkeypatch):
    monkeypatch.setenv("RLER_REWARD_SCHEME", "f2p_only")
    make_evaluation_payload(
        "resolved",
        passed_tests=["new_test"],
        fail_to_pass_expected=["new_test"],
    )
    assert os.environ["RLER_REWARD_SCHEME"] == "f2p_only"
