import pytest

from experience_gen.contexts import (
    full_variance_population,
    historical_selection_error,
    strict_reward_variance,
    validate_context,
)


def context(rewards):
    node_ids = list(rewards)
    return {
        "context_id": "repo__repo-1:R1",
        "instance_id": "repo__repo-1",
        "round_index": 1,
        "node_ids": node_ids,
        "gt_rewards": rewards,
        "view": {
            "question": {"system_prompt": "", "user_prompt": "task"},
            "previous_state": {},
            "parent_trajectory": None,
            "continuations": [
                {"node_id": node_id, "summary": {}, "raw_continuation": {}}
                for node_id in node_ids
            ],
        },
    }


def test_strict_variance_gate():
    assert strict_reward_variance(context({"a": 0.0, "b": 2.0}))
    assert not strict_reward_variance(context({"a": 0.0, "b": 1.0}))
    assert not strict_reward_variance(context({"a": 1.0, "b": 1.0}))


def test_context_validation_derives_variance():
    value = validate_context(context({"a": -1.0, "b": 2.0}))
    assert value["has_variance"] is True
    assert value["round_index"] == 1


def test_full_population_keeps_every_strict_variance_group():
    first = context({"a": 0.0, "b": 2.0})
    second = {
        **context({"c": 1.0, "d": 2.0}),
        "context_id": "repo__repo-1:R2",
        "round_index": 2,
    }
    orphan = {
        **context({"e": 0.0, "f": 2.0}),
        "context_id": "repo__repo-2:R2",
        "instance_id": "repo__repo-2",
        "round_index": 2,
    }
    assert [row["context_id"] for row in full_variance_population([orphan, second, first])] == [
        "repo__repo-1:R1",
        "repo__repo-1:R2",
        "repo__repo-2:R2",
    ]


def test_historical_error_uses_actual_selected_branch() -> None:
    failed = {
        **context({"a": 0.0, "b": 2.0}),
        "selected_node_id": "a",
        "selection_success": False,
    }
    passed = {**failed, "selected_node_id": "b", "selection_success": True}
    assert historical_selection_error(failed)
    assert not historical_selection_error(passed)


def test_historical_error_rejects_missing_selection_evidence() -> None:
    with pytest.raises(ValueError, match="missing historical selection evidence"):
        historical_selection_error(context({"a": 0.0, "b": 2.0}))


def test_historical_error_rejects_inconsistent_landed_outcome() -> None:
    value = {
        **context({"a": 0.0, "b": 2.0}),
        "selected_node_id": "a",
        "selection_success": True,
    }
    with pytest.raises(ValueError, match="inconsistent historical selection"):
        historical_selection_error(value)
