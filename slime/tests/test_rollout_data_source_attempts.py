from types import SimpleNamespace

import pytest

from slime.rollout.data_source import (
    ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
    RolloutDataSource,
    TrainingInstanceBudgetExhausted,
    TrainingValidationBoundaryReached,
)
from slime.utils.types import Sample


class _Dataset:
    def __init__(self, size: int):
        self.samples = [Sample(metadata={"row": index}) for index in range(size)]
        self.shuffle_calls: list[int] = []

    def __len__(self) -> int:
        return len(self.samples)

    def shuffle(self, epoch_id: int) -> None:
        self.shuffle_calls.append(epoch_id)


def _source(
    *,
    sample_group_index: int = 0,
    sample_offset: int = 0,
    last_validation_attempt: int = 0,
    last_validation_scheduled_attempt: int | None = None,
) -> RolloutDataSource:
    source = RolloutDataSource.__new__(RolloutDataSource)
    source.args = SimpleNamespace(
        n_samples_per_prompt=1,
        eval_instance_interval=100,
        train_instance_budget=1250,
        rollout_shuffle=False,
    )
    source.dataset = _Dataset(250)
    source.epoch_id = sample_group_index // 250
    source.sample_group_index = sample_group_index
    source.sample_index = sample_group_index
    source.sample_offset = sample_offset
    source.last_validation_scheduled_attempt = (
        last_validation_attempt
        if last_validation_scheduled_attempt is None
        else last_validation_scheduled_attempt
    )
    source.last_validation_attempt = last_validation_attempt
    source.validation_rollout_ids = {}
    source.validation_policy_versions = {}
    source.metadata = {}
    return source


def test_validation_boundary_blocks_attempt_101_until_scheduled():
    source = _source()

    groups = source.get_samples(100)
    assert len(groups) == 100
    assert source.training_progress() == {
        "attempted_instances": 100,
        "instance_budget": 1250,
        "dataset_size": 250,
        "epoch": 0.4,
        "last_validation_attempt": 0,
        "last_validation_scheduled_attempt": 0,
        "validation_rollout_ids": {},
        "validation_policy_versions": {},
        "eval_instance_interval": 100,
    }

    with pytest.raises(TrainingValidationBoundaryReached) as exc_info:
        source.get_samples(1)
    assert exc_info.value.attempted_instances == 100
    assert exc_info.value.boundary == 100
    assert exc_info.value.preserved_partial is False
    assert exc_info.value.preserved_group_count == 0
    assert exc_info.value.preserved_pending_count == 0
    assert source.sample_group_index == 100

    source.mark_validation_scheduled(100, rollout_id=5)
    next_group = source.get_samples(1)
    assert next_group[0][0].group_index == 100
    assert source.sample_group_index == 101
    assert source.last_validation_attempt == 0
    assert source.last_validation_scheduled_attempt == 100
    assert source.validation_rollout_ids == {100: 5}

    # Completion is tracked independently and may arrive while later source
    # attempts are already rolling out.
    source.acknowledge_validation(100)
    assert source.last_validation_attempt == 100


def test_batched_draw_cannot_cross_unacknowledged_validation_boundary():
    source = _source(
        sample_group_index=99,
        sample_offset=99,
        last_validation_attempt=0,
    )
    before = (
        source.sample_group_index,
        source.sample_index,
        source.sample_offset,
        source.epoch_id,
    )

    with pytest.raises(
        ValueError,
        match="would cross an unscheduled.*boundary",
    ):
        source.get_samples(2)

    assert (
        source.sample_group_index,
        source.sample_index,
        source.sample_offset,
        source.epoch_id,
    ) == before
    # A draw ending exactly at attempt 100 remains valid.
    group = source.get_samples(1)
    assert group[0][0].group_index == 99
    assert source.sample_group_index == 100


def test_caller_dropped_results_still_advance_attempt_boundary():
    source = _source()

    # Rollout failures and dynamic-filter drops happen after the source draw.
    # Deliberately discard every returned group to model those outcomes.
    for expected_attempt in range(1, 101):
        source.get_samples(1)
        assert source.sample_group_index == expected_attempt

    with pytest.raises(TrainingValidationBoundaryReached) as exc_info:
        source.get_samples(1)
    assert exc_info.value.attempted_instances == 100
    assert exc_info.value.boundary == 100
    assert source.sample_group_index == 100


def test_budget_allows_attempt_1250_then_stops_without_extra_draw():
    source = _source(
        sample_group_index=1249,
        sample_offset=249,
        last_validation_attempt=1200,
    )

    final_group = source.get_samples(1)
    assert final_group[0][0].group_index == 1249
    assert source.sample_group_index == 1250
    assert source.training_progress()["epoch"] == 5.0

    with pytest.raises(TrainingInstanceBudgetExhausted) as exc_info:
        source.get_samples(1)
    assert exc_info.value.attempted_instances == 1250
    assert exc_info.value.budget == 1250
    assert source.sample_group_index == 1250


def test_interval_boundary_precedes_budget_at_exact_multiple():
    source = _source(
        sample_group_index=1200,
        sample_offset=200,
        last_validation_attempt=1100,
    )

    with pytest.raises(TrainingValidationBoundaryReached) as exc_info:
        source.get_samples(1)
    assert exc_info.value.boundary == 1200

    source.mark_validation_scheduled(1200)
    source.acknowledge_validation(1200)
    source.sample_group_index = 1250
    source.sample_offset = 0
    with pytest.raises(TrainingInstanceBudgetExhausted):
        source.get_samples(1)


def test_validation_cannot_complete_before_it_is_scheduled():
    source = _source(sample_group_index=100)

    with pytest.raises(ValueError, match="unscheduled validation"):
        source.acknowledge_validation(100)

    source.mark_validation_scheduled(100)
    source.acknowledge_validation(100)
    assert source.last_validation_attempt == 100


def test_validation_attempt_cannot_be_remapped_to_new_policy_rollout():
    source = _source(sample_group_index=100)

    source.mark_validation_scheduled(100, rollout_id=5)
    source.mark_validation_scheduled(100, rollout_id=5)
    with pytest.raises(ValueError, match="cannot remap validation attempt"):
        source.mark_validation_scheduled(100, rollout_id=7)

    assert source.validation_rollout_ids == {100: 5}


def test_validation_attempt_cannot_be_remapped_to_new_policy_version():
    source = _source(sample_group_index=100)

    source.mark_validation_scheduled(
        100,
        rollout_id=5,
        policy_version="checkpoint-0000004",
    )
    source.mark_validation_scheduled(
        100,
        rollout_id=5,
        policy_version="checkpoint-0000004",
    )
    with pytest.raises(
        ValueError,
        match="different policy version",
    ):
        source.mark_validation_scheduled(
            100,
            rollout_id=5,
            policy_version="checkpoint-0000005",
        )

    assert source.validation_policy_versions == {
        100: "checkpoint-0000004"
    }


def test_validation_rollout_mapping_round_trips_checkpoint(
    tmp_path,
):
    source = _source(
        sample_group_index=156,
        sample_offset=156,
        last_validation_attempt=100,
        last_validation_scheduled_attempt=156,
    )
    source.validation_rollout_ids = {100: 5, 156: 7}
    source.validation_policy_versions = {
        100: "checkpoint-0000004",
        156: "checkpoint-0000006",
    }
    source.args = SimpleNamespace(
        rollout_global_dataset=True,
        save=str(tmp_path),
    )
    source.save(7)

    restored = _source()
    restored.args = SimpleNamespace(
        rollout_global_dataset=True,
        load=str(tmp_path),
        rollout_shuffle=False,
    )
    restored.load(7)

    assert restored.sample_group_index == 156
    assert restored.sample_offset == 156
    assert restored.last_validation_attempt == 100
    assert restored.last_validation_scheduled_attempt == 156
    assert restored.validation_rollout_ids == {100: 5, 156: 7}
    assert restored.validation_policy_versions == {
        100: "checkpoint-0000004",
        156: "checkpoint-0000006",
    }


def test_staged_checkpoint_is_invisible_until_atomic_commit(tmp_path):
    source = _source(
        sample_group_index=251,
        sample_offset=1,
        last_validation_attempt=200,
        last_validation_scheduled_attempt=200,
    )
    source.epoch_id = 1
    source.metadata = {
        "__rler_rollout_collector_state_v1__": {
            "collector": "naive",
            "buffer": [],
            "pending_tasks": [],
        }
    }
    source.args = SimpleNamespace(
        rollout_global_dataset=True,
        save=str(tmp_path),
    )

    source.save(3, staged=True)
    rollout_root = tmp_path / "rollout"
    final_path = rollout_root / "global_dataset_state_dict_3.pt"
    staged_path = rollout_root / "global_dataset_state_dict_3.pt.staged"
    assert staged_path.is_file()
    assert not final_path.exists()
    assert not list(rollout_root.glob(".*.tmp"))

    source.commit_staged_save(3)
    assert final_path.is_file()
    assert not staged_path.exists()

    state = __import__("torch").load(
        final_path,
        weights_only=False,
    )
    assert (
        state["checkpoint_schema_version"]
        == ROLLOUT_CHECKPOINT_SCHEMA_VERSION
    )
    assert state["checkpoint_rollout_id"] == 3
    assert state["sample_group_index"] == 251


def test_epoch_and_offset_resume_selects_next_unconsumed_instance(
    tmp_path,
):
    source = _source(
        sample_group_index=251,
        sample_offset=1,
        last_validation_attempt=200,
        last_validation_scheduled_attempt=200,
    )
    source.epoch_id = 1
    source.args = SimpleNamespace(
        rollout_global_dataset=True,
        save=str(tmp_path),
    )
    source.save(4)

    restored = _source()
    restored.args = SimpleNamespace(
        rollout_global_dataset=True,
        load=str(tmp_path),
        rollout_shuffle=True,
        n_samples_per_prompt=1,
        eval_instance_interval=100,
        train_instance_budget=1250,
    )
    restored.load(4)

    assert restored.epoch_id == 1
    assert restored.sample_offset == 1
    assert restored.dataset.shuffle_calls == [1]
    next_group = restored.get_samples(1)
    assert next_group[0][0].group_index == 251
    assert next_group[0][0].metadata["row"] == 1
    assert restored.sample_group_index == 252
    assert restored.sample_offset == 2


def test_rollout_checkpoint_load_rejects_wrong_schema(tmp_path):
    source = _source(sample_group_index=1, sample_offset=1)
    source.args = SimpleNamespace(
        rollout_global_dataset=True,
        save=str(tmp_path),
    )
    source.save(4)
    path = (
        tmp_path / "rollout" / "global_dataset_state_dict_4.pt"
    )
    torch = __import__("torch")
    state = torch.load(path, weights_only=False)
    state["checkpoint_schema_version"] = (
        ROLLOUT_CHECKPOINT_SCHEMA_VERSION - 1
    )
    torch.save(state, path)

    restored = _source()
    restored.args = SimpleNamespace(
        rollout_global_dataset=True,
        load=str(tmp_path),
        rollout_shuffle=False,
    )
    with pytest.raises(
        RuntimeError,
        match="unsupported rollout data-source checkpoint schema",
    ):
        restored.load(4)

