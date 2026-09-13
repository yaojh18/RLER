from __future__ import annotations

import json

import pytest

from swe_agent.exceptions import PolicyVersionMismatch
from swe_agent.policy_version import (
    POLICY_VERSION_STATE_ENV,
    assert_policy_staleness,
    begin_policy_weight_update,
    checkpoint_policy_stale_lag,
    commit_policy_weight_update,
    committed_policy_version,
    observed_policy_version,
    policy_version_for_checkpoint,
)


def test_checkpoint_policy_stale_lag_tracks_true_weight_age():
    assert (
        checkpoint_policy_stale_lag(
            "checkpoint-base",
            consumer_rollout_id=0,
        )
        == 0
    )
    assert (
        checkpoint_policy_stale_lag(
            "checkpoint-base",
            consumer_rollout_id=1,
        )
        == 1
    )
    assert (
        checkpoint_policy_stale_lag(
            "checkpoint-base",
            consumer_rollout_id=2,
        )
        == 2
    )
    assert (
        checkpoint_policy_stale_lag(
            "checkpoint-0000003",
            consumer_rollout_id=5,
        )
        == 1
    )
    with pytest.raises(PolicyVersionMismatch, match="invalid coordinated"):
        checkpoint_policy_stale_lag(
            "legacy-rollout-2",
            consumer_rollout_id=3,
        )


def test_policy_version_transition_is_fail_closed(monkeypatch, tmp_path):
    state_path = tmp_path / "policy-version.json"
    monkeypatch.setenv(POLICY_VERSION_STATE_ENV, str(state_path))

    base = policy_version_for_checkpoint(-1)
    transition = begin_policy_weight_update(base)
    with pytest.raises(PolicyVersionMismatch, match="transitioning"):
        committed_policy_version()
    with pytest.raises(PolicyVersionMismatch, match="request_start"):
        assert_policy_staleness(
            base, max_lag=0, stage="request_start", reject_transition=True
        )

    commit_policy_weight_update(base, transition)
    assert committed_policy_version() == base
    assert observed_policy_version() == base
    assert_policy_staleness(
        base, max_lag=0, stage="request_complete", reject_transition=True
    )

    next_version = policy_version_for_checkpoint(0)
    transition = begin_policy_weight_update(next_version)
    assert observed_policy_version() == base
    with pytest.raises(PolicyVersionMismatch, match="request_complete"):
        assert_policy_staleness(
            base, max_lag=0, stage="request_complete", reject_transition=True
        )
    commit_policy_weight_update(next_version, transition)

    with pytest.raises(PolicyVersionMismatch, match="expected"):
        assert_policy_staleness(
            base, max_lag=0, stage="request_start", reject_transition=True
        )
    assert_policy_staleness(
        next_version, max_lag=0, stage="request_start", reject_transition=True
    )

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["committed_version"] == next_version
    assert payload["updating"] is False


def test_policy_stale_one_accepts_one_checkpoint_only(monkeypatch, tmp_path):
    state_path = tmp_path / "policy-version.json"
    monkeypatch.setenv(POLICY_VERSION_STATE_ENV, str(state_path))

    base = policy_version_for_checkpoint(-1)
    transition = begin_policy_weight_update(base)
    commit_policy_weight_update(base, transition)
    assert assert_policy_staleness(
        base, max_lag=1, stage="rollout_start"
    ) == 0

    next_version = policy_version_for_checkpoint(0)
    transition = begin_policy_weight_update(next_version)
    assert assert_policy_staleness(
        base, max_lag=1, stage="request_start"
    ) == 0
    commit_policy_weight_update(next_version, transition)
    assert assert_policy_staleness(
        base, max_lag=1, stage="request_start"
    ) == 1

    second_version = policy_version_for_checkpoint(1)
    transition = begin_policy_weight_update(second_version)
    assert assert_policy_staleness(
        base, max_lag=1, stage="request_start"
    ) == 1
    commit_policy_weight_update(second_version, transition)
    with pytest.raises(PolicyVersionMismatch, match="max_lag=1"):
        assert_policy_staleness(
            base, max_lag=1, stage="request_start"
        )


def test_policy_version_coordination_is_opt_in(monkeypatch):
    monkeypatch.delenv(POLICY_VERSION_STATE_ENV, raising=False)
    assert committed_policy_version() is None
    assert begin_policy_weight_update("checkpoint-base") is None
    commit_policy_weight_update("checkpoint-base", None)
    assert_policy_staleness(
        "any-legacy-label",
        max_lag=0,
        stage="request_start",
        reject_transition=True,
    )
