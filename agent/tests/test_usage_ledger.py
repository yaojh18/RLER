import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from agent_rl import run_utils
from swe_agent.usage import (
    UsageMetricsTracker,
    build_usage_group_prefix,
    configure_usage_ledger,
    configure_usage_resume_window,
    normalize_usage,
    record_group_disposition,
    record_model_usage,
    usage_ledger_offset,
    usage_context,
)


@pytest.fixture(autouse=True)
def _reset_usage_ledger(monkeypatch):
    configure_usage_ledger(None)
    configure_usage_resume_window(None)
    monkeypatch.delenv("RLER_USAGE_LEDGER_PATH", raising=False)
    monkeypatch.delenv("RLER_USAGE_RESUME", raising=False)
    monkeypatch.delenv("RLER_USAGE_ATTEMPT_NONCE", raising=False)
    monkeypatch.delenv(
        "RLER_LITELLM_RATE_LIMIT_FALLBACK_MODEL",
        raising=False,
    )
    yield
    configure_usage_ledger(None)
    configure_usage_resume_window(None)


def _events(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_normalize_usage_does_not_double_count_cached_input():
    normalized = normalize_usage(
        {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 5},
        }
    )

    assert normalized["input_tokens"] == 11
    assert normalized["cached_input_tokens"] == 5
    assert normalized["output_tokens"] == 7
    assert normalized["total_tokens"] == 18
    assert normalized["exact_usage"] is True


def test_request_ledger_contains_counts_and_context_but_not_payload(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)

    with usage_context(
        phase="train",
        group_id="group-7",
        model_family="qwen",
        model_role="policy",
        logical_call_id="logical-9",
        attempt_index=1,
        correction_index=2,
    ):
        record_model_usage(
            usage={"prompt_tokens": 10, "completion_tokens": 3},
            status="success",
            latency_ms=12.5,
            request_started_at=1234.5,
        )

    [event] = _events(ledger_path)
    assert isinstance(event["hostname"], str)
    assert event["hostname"]
    assert event["phase"] == "train"
    assert event["group_id"] == "group-7"
    assert event["model_family"] == "qwen"
    assert event["model_role"] == "policy"
    assert event["logical_call_id"] == "logical-9"
    assert event["attempt_index"] == 1
    assert event["correction_index"] == 2
    assert event["total_tokens"] == 13
    assert event["request_started_at"] == 1234.5
    serialized = json.dumps(event)
    assert "prompt" not in event
    assert "messages" not in event
    assert "api_key" not in serialized
    assert "api_base" not in serialized


def test_group_disposition_has_nonempty_host_identity(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)

    record_group_disposition(
        "train-group",
        disposition="accepted",
        phase="train",
    )

    [event] = _events(ledger_path)
    assert isinstance(event["hostname"], str)
    assert event["hostname"]


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -0.1, "bad", True],
)
def test_request_started_at_must_be_finite_and_nonnegative(
    tmp_path,
    value,
):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)

    with pytest.raises(ValueError, match="request_started_at"):
        record_model_usage(
            usage={"prompt_tokens": 1, "completion_tokens": 1},
            status="success",
            request_started_at=value,
        )
    assert _events(ledger_path) == []


def test_metrics_deduplicate_events_and_include_filtered_and_validation_tokens(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)

    with usage_context(phase="train", group_id="train-group"):
        for _ in range(2):
            record_model_usage(
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 6},
                },
                status="success",
                model_family="qwen",
                model_role="policy",
                event_id="duplicate-event",
            )
        record_model_usage(
            usage={
                "input_tokens": 4,
                "output_tokens": 1,
                "input_tokens_details": {"cached_tokens": 2},
            },
            status="success",
            model_family="glm",
            model_role="rubric_judge",
        )
        record_group_disposition(
            "train-group",
            disposition="filtered",
            reason="zero_variance",
        )

    with usage_context(phase="validation", group_id="val-group"):
        record_model_usage(
            usage={"prompt_tokens": 2, "completion_tokens": 1},
            status="success",
            model_family="qwen",
            model_role="policy",
        )

    tracker = UsageMetricsTracker(ledger_path)
    metrics = tracker.wandb_metrics()
    assert metrics["usage/total_tokens_cumulative"] == 20
    assert metrics["usage/train_tokens_cumulative"] == 17
    assert metrics["usage/validation_tokens_cumulative"] == 3
    assert metrics["usage/qwen_tokens_cumulative"] == 15
    assert metrics["usage/qwen_input_tokens_cumulative"] == 12
    assert metrics["usage/qwen_cached_input_tokens_cumulative"] == 6
    assert metrics["usage/qwen_output_tokens_cumulative"] == 3
    assert metrics["usage/glm_tokens_cumulative"] == 5
    assert metrics["usage/glm_input_tokens_cumulative"] == 4
    assert metrics["usage/glm_cached_input_tokens_cumulative"] == 2
    assert metrics["usage/glm_output_tokens_cumulative"] == 1
    assert metrics["usage/qwen_requests_cumulative"] == 2
    assert metrics["usage/qwen_failed_requests_cumulative"] == 0
    assert metrics["usage/qwen_missing_usage_attempts_cumulative"] == 0
    assert metrics["usage/glm_requests_cumulative"] == 1
    assert metrics["usage/glm_failed_requests_cumulative"] == 0
    assert metrics["usage/glm_missing_usage_attempts_cumulative"] == 0
    assert metrics["usage/qwen_exact_usage_fraction"] == 1.0
    assert metrics["usage/qwen_failed_request_fraction"] == 0.0
    assert metrics["usage/glm_exact_usage_fraction"] == 1.0
    assert metrics["usage/glm_failed_request_fraction"] == 0.0
    assert metrics["usage/dropped_tokens_cumulative"] == 17
    assert metrics["usage/zero_variance_dropped_tokens_cumulative"] == 17
    assert metrics["usage/qwen_input_tokens_delta"] == 12
    assert metrics["usage/qwen_cached_input_tokens_delta"] == 6
    assert metrics["usage/glm_cached_input_tokens_delta"] == 2
    assert metrics["usage/glm_output_tokens_delta"] == 1
    assert metrics["usage/exact_usage_fraction"] == 1.0

    unchanged = tracker.wandb_metrics()
    assert unchanged["usage/qwen_input_tokens_delta"] == 0
    assert unchanged["usage/qwen_cached_input_tokens_delta"] == 0
    assert unchanged["usage/glm_cached_input_tokens_delta"] == 0
    assert unchanged["usage/glm_output_tokens_delta"] == 0


def test_peek_and_snapshot_do_not_consume_optimizer_update_delta(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)
    tracker = UsageMetricsTracker(ledger_path)

    with usage_context(phase="train", group_id="accepted-group"):
        record_model_usage(
            usage={"prompt_tokens": 10, "completion_tokens": 2},
            status="success",
            model_family="qwen",
            model_role="policy",
        )

    heartbeat = tracker.peek()
    snapshot = tracker.snapshot()
    for telemetry in (heartbeat, snapshot):
        assert telemetry["usage/total_tokens_cumulative"] == 12
        assert telemetry["usage/qwen_tokens_cumulative"] == 12
        assert not any(key.endswith("_delta") for key in telemetry)

    record_group_disposition(
        "accepted-group",
        disposition="accepted",
        phase="train",
    )
    update = tracker.commit_update()
    assert update["usage/total_tokens_cumulative"] == 12
    assert update["usage/dropped_tokens_cumulative"] == 0
    assert update["usage/qwen_input_tokens_delta"] == 10
    assert update["usage/qwen_output_tokens_delta"] == 2

    repeated_commit = tracker.commit_update()
    assert repeated_commit["usage/total_tokens_cumulative"] == 12
    assert repeated_commit["usage/qwen_input_tokens_delta"] == 0
    assert repeated_commit["usage/qwen_output_tokens_delta"] == 0

    with usage_context(phase="train", group_id="new-group"):
        record_model_usage(
            usage={"prompt_tokens": 4, "completion_tokens": 1},
            status="success",
            model_family="glm",
            model_role="rubric_judge",
        )
    after_new_event = tracker.peek()
    assert after_new_event["usage/total_tokens_cumulative"] == 17
    assert after_new_event["usage/glm_tokens_cumulative"] == 5
    assert not any(key.endswith("_delta") for key in after_new_event)

    next_update = tracker.commit_update()
    assert next_update["usage/total_tokens_cumulative"] == 17
    assert next_update["usage/qwen_input_tokens_delta"] == 0
    assert next_update["usage/qwen_output_tokens_delta"] == 0
    assert next_update["usage/glm_input_tokens_delta"] == 4
    assert next_update["usage/glm_output_tokens_delta"] == 1


def test_metrics_apply_late_and_changed_group_dispositions_incrementally(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)
    tracker = UsageMetricsTracker(ledger_path)

    with usage_context(phase="train", group_id="late-group"):
        record_model_usage(
            usage={"prompt_tokens": 10, "completion_tokens": 2},
            status="success",
            model_family="qwen",
            model_role="policy",
        )

    before_disposition = tracker.wandb_metrics()
    assert before_disposition["usage/total_tokens_cumulative"] == 12
    assert before_disposition["usage/dropped_tokens_cumulative"] == 0

    record_group_disposition(
        "late-group",
        disposition="filtered",
        reason="zero_variance",
        phase="train",
    )
    filtered = tracker.wandb_metrics()
    assert filtered["usage/dropped_tokens_cumulative"] == 12
    assert filtered["usage/zero_variance_dropped_tokens_cumulative"] == 12
    assert filtered["usage/qwen_input_tokens_delta"] == 0

    # A request can be appended after the disposition due to cross-process
    # ledger interleaving. It must inherit the current group disposition
    # without requiring a history rescan.
    with usage_context(
        phase="train",
        group_id="late-group",
        attempt_index=1,
    ):
        record_model_usage(
            usage={"prompt_tokens": 4, "completion_tokens": 1},
            status="success",
            model_family="glm",
            model_role="rubric_judge",
        )
    retried = tracker.wandb_metrics()
    assert retried["usage/total_tokens_cumulative"] == 17
    assert retried["usage/dropped_tokens_cumulative"] == 17
    assert retried["usage/zero_variance_dropped_tokens_cumulative"] == 17

    # A later final disposition must correct the historical classification in
    # O(1) using the per-group token total.
    record_group_disposition(
        "late-group",
        disposition="dropped",
        reason="excess_group",
        phase="train",
    )
    excess = tracker.wandb_metrics()
    assert excess["usage/dropped_tokens_cumulative"] == 17
    assert excess["usage/zero_variance_dropped_tokens_cumulative"] == 0

    record_group_disposition(
        "late-group",
        disposition="accepted",
        phase="train",
    )
    accepted = tracker.wandb_metrics()
    assert accepted["usage/dropped_tokens_cumulative"] == 0
    assert accepted["usage/zero_variance_dropped_tokens_cumulative"] == 0


def test_restart_tracker_keeps_history_cumulative_but_bootstraps_deltas(
    tmp_path,
    monkeypatch,
):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)
    with usage_context(phase="train", group_id="old-attempt"):
        record_model_usage(
            usage={"prompt_tokens": 10, "completion_tokens": 2},
            status="success",
            model_family="qwen",
            model_role="policy",
        )

    monkeypatch.setenv("RLER_USAGE_RESUME", "1")
    restarted = UsageMetricsTracker(ledger_path)
    first = restarted.peek()
    assert first["usage/total_tokens_cumulative"] == 12
    assert not any(key.endswith("_delta") for key in first)
    first_commit = restarted.commit_update()
    assert first_commit["usage/qwen_input_tokens_delta"] == 0
    assert first_commit["usage/qwen_output_tokens_delta"] == 0

    with usage_context(phase="train", group_id="new-attempt"):
        record_model_usage(
            usage={"prompt_tokens": 4, "completion_tokens": 1},
            status="success",
            model_family="qwen",
            model_role="policy",
        )
    resumed_heartbeat = restarted.peek()
    assert resumed_heartbeat["usage/total_tokens_cumulative"] == 17
    assert not any(key.endswith("_delta") for key in resumed_heartbeat)
    next_refresh = restarted.commit_update()
    assert next_refresh["usage/total_tokens_cumulative"] == 17
    assert next_refresh["usage/qwen_input_tokens_delta"] == 4
    assert next_refresh["usage/qwen_output_tokens_delta"] == 1

    # A fresh run intentionally reports requests written before its first
    # refresh. Only the explicit resume path suppresses historical deltas.
    monkeypatch.delenv("RLER_USAGE_RESUME")
    fresh = UsageMetricsTracker(ledger_path)
    fresh_first = fresh.wandb_metrics()
    assert fresh_first["usage/qwen_input_tokens_delta"] == 14
    assert fresh_first["usage/qwen_output_tokens_delta"] == 3


def test_checkpoint_resume_skips_orphan_ledger_suffix(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)
    with usage_context(phase="train", group_id="durable"):
        record_model_usage(
            usage={"prompt_tokens": 10, "completion_tokens": 2},
            status="success",
            model_family="qwen",
            model_role="policy",
        )
    durable_offset = usage_ledger_offset()
    assert durable_offset == ledger_path.stat().st_size

    with usage_context(phase="train", group_id="orphan"):
        record_model_usage(
            usage={"prompt_tokens": 20, "completion_tokens": 3},
            status="success",
            model_family="qwen",
            model_role="policy",
        )

    configure_usage_resume_window(durable_offset)
    tracker = UsageMetricsTracker(ledger_path)
    resumed = tracker.peek()
    assert resumed["usage/total_tokens_cumulative"] == 12

    with usage_context(phase="train", group_id="new"):
        record_model_usage(
            usage={"prompt_tokens": 4, "completion_tokens": 1},
            status="success",
            model_family="qwen",
            model_role="policy",
        )
    committed = tracker.commit_update()
    assert committed["usage/total_tokens_cumulative"] == 17
    assert committed["usage/qwen_input_tokens_delta"] == 4
    assert committed["usage/qwen_output_tokens_delta"] == 1


def test_usage_context_can_suppress_checkpoint_replay_cost(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)

    with usage_context(
        phase="train",
        group_id="checkpoint-replay",
        suppress_accounting=True,
    ):
        assert (
            record_model_usage(
                usage={"prompt_tokens": 10, "completion_tokens": 2},
                status="success",
                model_family="qwen",
                model_role="policy",
            )
            is None
        )

    assert _events(ledger_path) == []


def test_excess_and_partial_drop_dispositions_keep_all_cost_in_total(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)
    cases = (
        ("excess-group", 4, 1, "excess", "excess_after_quota"),
        (
            "partial-group",
            3,
            2,
            "dropped",
            "validation_boundary_partial_update",
        ),
        (
            "partial-alias-group",
            2,
            1,
            "partial_dropped",
            "source_budget_partial_update",
        ),
        ("accepted-group", 4, 2, "accepted", ""),
    )
    for group_id, input_tokens, output_tokens, disposition, reason in cases:
        with usage_context(phase="train", group_id=group_id):
            record_model_usage(
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                },
                status="success",
                model_family="qwen",
                model_role="policy",
            )
            record_group_disposition(
                group_id,
                disposition=disposition,
                reason=reason,
            )

    metrics = UsageMetricsTracker(ledger_path).wandb_metrics()
    assert metrics["usage/total_tokens_cumulative"] == 19
    assert metrics["usage/train_tokens_cumulative"] == 19
    assert metrics["usage/qwen_tokens_cumulative"] == 19
    assert metrics["usage/dropped_tokens_cumulative"] == 13
    assert metrics["usage/zero_variance_dropped_tokens_cumulative"] == 0


def test_usage_group_prefix_prevents_replay_from_reclassifying_history(
    tmp_path,
    monkeypatch,
):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)
    monkeypatch.setenv("RLER_USAGE_ATTEMPT_NONCE", "job 17/restart 2")
    first = build_usage_group_prefix(
        phase="train",
        rollout_id=3,
        dataset_name="verified",
        instance_id="django__django-1",
        task_index=12,
    )
    assert first == (
        "train/attempt-job_17_restart_2/r0003/verified/"
        "django__django-1/t000012"
    )
    with usage_context(phase="train", group_id=f"{first}:g0"):
        record_model_usage(
            usage={"prompt_tokens": 10, "completion_tokens": 2},
            status="success",
            model_family="qwen",
            model_role="policy",
        )
        record_group_disposition(
            f"{first}:g0",
            disposition="filtered",
            reason="zero_variance",
        )

    monkeypatch.setenv("RLER_USAGE_ATTEMPT_NONCE", "job-17-restart-3")
    replay = build_usage_group_prefix(
        phase="train",
        rollout_id=3,
        dataset_name="verified",
        instance_id="django__django-1",
        task_index=12,
    )
    assert replay != first
    assert replay.startswith("train/attempt-job-17-restart-3/")
    with usage_context(phase="train", group_id=f"{replay}:g0"):
        record_model_usage(
            usage={"prompt_tokens": 4, "completion_tokens": 1},
            status="success",
            model_family="qwen",
            model_role="policy",
        )
        record_group_disposition(
            f"{replay}:g0",
            disposition="accepted",
        )

    metrics = UsageMetricsTracker(ledger_path).wandb_metrics()
    assert metrics["usage/total_tokens_cumulative"] == 17
    assert metrics["usage/dropped_tokens_cumulative"] == 12
    assert metrics["usage/zero_variance_dropped_tokens_cumulative"] == 12


def test_metrics_process_each_request_only_once_across_refreshes(
    tmp_path,
    monkeypatch,
):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)
    tracker = UsageMetricsTracker(ledger_path)
    processed_event_ids = []
    apply_request = tracker._apply_request_event

    def counted_apply(event):
        processed_event_ids.append(event["event_id"])
        apply_request(event)

    monkeypatch.setattr(tracker, "_apply_request_event", counted_apply)

    for event_id in ("request-1", "request-2"):
        record_model_usage(
            usage={"prompt_tokens": 3, "completion_tokens": 1},
            status="success",
            model_family="qwen",
            model_role="policy",
            event_id=event_id,
        )
    tracker.wandb_metrics()
    assert processed_event_ids == ["request-1", "request-2"]

    tracker.wandb_metrics()
    assert processed_event_ids == ["request-1", "request-2"]

    record_group_disposition(
        "group-without-requests",
        disposition="invalid",
        reason="missing_group",
    )
    tracker.wandb_metrics()
    assert processed_event_ids == ["request-1", "request-2"]

    record_model_usage(
        usage={"prompt_tokens": 5, "completion_tokens": 2},
        status="success",
        model_family="glm",
        model_role="rubric_generation",
        event_id="request-3",
    )
    final = tracker.wandb_metrics()
    assert processed_event_ids == ["request-1", "request-2", "request-3"]
    assert final["usage/requests_cumulative"] == 3
    assert not hasattr(tracker, "_requests")


def test_missing_failure_usage_is_visible_in_coverage(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)

    record_model_usage(
        usage=None,
        status="error",
        model_family="glm",
        model_role="rubric_judge",
        error=RuntimeError("secret-bearing provider response"),
    )

    [event] = _events(ledger_path)
    assert event["error_type"] == "RuntimeError"
    assert "secret-bearing" not in json.dumps(event)
    metrics = UsageMetricsTracker(ledger_path).wandb_metrics()
    assert metrics["usage/missing_usage_attempts_cumulative"] == 1
    assert metrics["usage/failed_requests_cumulative"] == 1
    assert metrics["usage/exact_usage_fraction"] == 0.0
    assert metrics["usage/glm_requests_cumulative"] == 1
    assert metrics["usage/glm_missing_usage_attempts_cumulative"] == 1
    assert metrics["usage/glm_failed_requests_cumulative"] == 1
    assert metrics["usage/glm_exact_usage_fraction"] == 0.0
    assert metrics["usage/glm_failed_request_fraction"] == 1.0


def test_litellm_preserves_provider_retry_configuration_without_fallback(
    tmp_path,
    monkeypatch,
):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)
    monkeypatch.setenv("RLER_LITELLM_RETRY_BASE_SECONDS", "0")
    monkeypatch.setattr(
        run_utils.litellm,
        "token_counter",
        lambda **_kwargs: 3,
    )

    async def inline_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(run_utils.asyncio, "to_thread", inline_to_thread)
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"score": 1}'),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                model_dump=lambda: {
                    "prompt_tokens": 9,
                    "completion_tokens": 4,
                }
            ),
        )

    monkeypatch.setattr(run_utils.litellm, "completion", completion)

    with usage_context(phase="train", group_id="group-1"):
        result = asyncio.run(
            run_utils.run_litellm_completion_async(
                model_name="nvidia/zai-org/glm-5.2",
                messages=[{"role": "user", "content": "not persisted"}],
                usage_model_role="rubric_judge",
                num_retries=1,
            )
        )

    assert result.content == '{"score": 1}'
    assert len(calls) == 1
    assert calls[0]["num_retries"] == 1
    events = _events(ledger_path)
    assert [event["status"] for event in events] == ["success"]
    assert [event["attempt_index"] for event in events] == [0]
    assert len({event["logical_call_id"] for event in events}) == 1
    assert all(event["model_family"] == "glm" for event in events)
    assert all(event["model_role"] == "rubric_judge" for event in events)
    assert "not persisted" not in ledger_path.read_text(encoding="utf-8")


def test_litellm_clips_completion_to_remaining_hosted_context(
    tmp_path,
    monkeypatch,
):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)
    monkeypatch.setenv("RLER_HOSTED_MODEL_CONTEXT_LENGTH", "65536")
    monkeypatch.setenv("RLER_HOSTED_MAX_COMPLETION_TOKENS", "20480")
    monkeypatch.setattr(
        run_utils.litellm,
        "token_counter",
        lambda **_kwargs: 49656,
    )
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"score": 1}'),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                model_dump=lambda: {
                    "prompt_tokens": 49656,
                    "completion_tokens": 4,
                }
            ),
        )

    async def inline_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(run_utils.litellm, "completion", completion)
    monkeypatch.setattr(run_utils.asyncio, "to_thread", inline_to_thread)

    result = asyncio.run(
        run_utils.run_litellm_completion_async(
            model_name="openai/azure/zai-org/glm-5.2",
            messages=[{"role": "user", "content": "not persisted"}],
            usage_model_role="rubric_generation",
            max_tokens=20480,
            num_retries=0,
        )
    )

    assert result.content == '{"score": 1}'
    assert calls[0]["max_tokens"] == 65536 - 49656
    [event] = _events(ledger_path)
    assert event["status"] == "success"


def test_litellm_rejects_prompt_that_already_fills_hosted_context(
    tmp_path,
    monkeypatch,
):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)
    monkeypatch.setenv("RLER_HOSTED_MODEL_CONTEXT_LENGTH", "65536")
    monkeypatch.setattr(
        run_utils.litellm,
        "token_counter",
        lambda **_kwargs: 65536,
    )
    monkeypatch.setattr(
        run_utils.litellm,
        "completion",
        lambda **_kwargs: pytest.fail("provider call must not start"),
    )

    with pytest.raises(
        run_utils.litellm.exceptions.ContextWindowExceededError,
        match="prompt_tokens=65536",
    ):
        asyncio.run(
            run_utils.run_litellm_completion_async(
                model_name="openai/azure/zai-org/glm-5.2",
                messages=[{"role": "user", "content": "not persisted"}],
                usage_model_role="rubric_generation",
                num_retries=0,
            )
        )

    assert _events(ledger_path) == []


def test_litellm_rate_limit_immediately_uses_configured_ultra_fallback(
    tmp_path,
    monkeypatch,
):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)
    primary_model = "nvidia/zai-org/glm-5.2"
    fallback_model = "nvidia/nvidia/nemotron-3-ultra"
    monkeypatch.setenv(
        "RLER_LITELLM_RATE_LIMIT_FALLBACK_MODEL",
        fallback_model,
    )
    monkeypatch.setattr(
        run_utils.litellm,
        "token_counter",
        lambda **_kwargs: 7,
    )

    class FakeRateLimitError(Exception):
        pass

    monkeypatch.setattr(
        run_utils.litellm,
        "RateLimitError",
        FakeRateLimitError,
    )
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == primary_model:
            raise FakeRateLimitError("primary capacity")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"score": 1}'),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                model_dump=lambda: {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                }
            ),
        )

    async def inline_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(run_utils.litellm, "completion", completion)
    monkeypatch.setattr(run_utils.asyncio, "to_thread", inline_to_thread)

    result = asyncio.run(
        run_utils.run_litellm_completion_async(
            model_name=primary_model,
            messages=[{"role": "user", "content": "not persisted"}],
            usage_model_role="rubric_judge",
            num_retries=4,
        )
    )

    assert [call["model"] for call in calls] == [
        primary_model,
        fallback_model,
    ]
    assert result.content == '{"score": 1}'
    assert result.model_name == fallback_model
    assert result.metadata["primary_model"] == primary_model
    assert (
        result.metadata["rate_limit_fallback_model"]
        == fallback_model
    )
    events = _events(ledger_path)
    assert [event["status"] for event in events] == ["success"]
    assert [event["model_family"] for event in events] == ["other"]
    assert UsageMetricsTracker(ledger_path).wandb_metrics()[
        "usage/total_tokens_cumulative"
    ] == 10


def test_litellm_nvidia_529_immediately_uses_configured_ultra_fallback(
    tmp_path,
    monkeypatch,
):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)
    primary_model = "openai/azure/zai-org/glm-5.2"
    fallback_model = "nvidia/nvidia/nemotron-3-ultra"
    monkeypatch.setenv(
        "RLER_LITELLM_RATE_LIMIT_FALLBACK_MODEL",
        fallback_model,
    )
    monkeypatch.setattr(
        run_utils.litellm,
        "token_counter",
        lambda **_kwargs: 7,
    )

    class FakeOverloadedError(Exception):
        status_code = 529

    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == primary_model:
            raise FakeOverloadedError("temporarily overloaded")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"score": 1}'),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                model_dump=lambda: {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                }
            ),
        )

    async def inline_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(run_utils.litellm, "completion", completion)
    monkeypatch.setattr(run_utils.asyncio, "to_thread", inline_to_thread)

    result = asyncio.run(
        run_utils.run_litellm_completion_async(
            model_name=primary_model,
            messages=[{"role": "user", "content": "not persisted"}],
            usage_model_role="rubric_judge",
            num_retries=4,
        )
    )

    assert [call["model"] for call in calls] == [
        primary_model,
        fallback_model,
    ]
    assert result.model_name == fallback_model
    assert len(_events(ledger_path)) == 1


def test_sglang_transport_returns_exact_usage_without_recording_retry_cost(
    tmp_path,
    monkeypatch,
):
    ledger_path = tmp_path / "usage.jsonl"
    configure_usage_ledger(ledger_path)

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return {
                "text": "done",
                "output_ids": [31, 32],
                "meta_info": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "cached_tokens": 2,
                    "finish_reason": {"type": "stop"},
                },
            }

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return Response()

    fake_aiohttp = SimpleNamespace(
        ClientTimeout=lambda **_kwargs: object(),
        ClientSession=lambda **_kwargs: Session(),
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)

    with usage_context(phase="validation", group_id="val-1"):
        completion = asyncio.run(
            run_utils.run_generate_with_route_async(
                route_name="policy",
                input_ids=[1, 2, 3],
                api_base="http://policy/v1",
            )
        )

    assert completion.output_token_ids == [31, 32]
    assert completion.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "cached_tokens": 2,
        "total_tokens": 5,
    }
    assert _events(ledger_path) == []
