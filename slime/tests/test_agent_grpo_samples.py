import pytest

from slime.utils.types import Sample
from train_agent.collect_grpo_rollout import build_rollout_samples
from train_agent.contracts import ExportGroup, ExportSample


def _export_group() -> ExportGroup:
    return ExportGroup(
        group_id="group",
        samples=[
            ExportSample(
                sample_id="sample",
                group_id="group",
                prompt=[],
                turns=[],
                reward=0.5,
                metadata={
                    "raw_rubric_score": 0.5,
                    "export_sample_id": "untrusted-sample",
                    "export_group_id": "untrusted-group",
                },
                token_ids=list(range(8)),
                loss_mask=[0, 0, 1, 1, 1, 1, 1, 1],
                response_length=6,
                rollout_logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
            )
        ],
    )


def test_oversized_sample_keeps_aligned_trainable_prefix():
    group = _export_group()

    samples, truncated = build_rollout_samples(
        groups=[group],
        include_turn_rewards=False,
        max_sample_tokens=5,
    )

    assert truncated == 1
    assert len(samples) == 1
    assert samples[0].tokens == [0, 1, 2, 3, 4]
    assert samples[0].response_length == 3
    assert samples[0].loss_mask == [1, 1, 1]
    assert samples[0].rollout_log_probs == [-0.1, -0.2, -0.3]
    assert samples[0].status is Sample.Status.TRUNCATED
    assert samples[0].metadata["right_truncated_tokens"] == 3


def test_missing_exact_rollout_fields_raise():
    group = _export_group()
    group.samples[0].rollout_logprobs = None

    with pytest.raises(ValueError, match="missing exact rollout training fields"):
        build_rollout_samples(
            groups=[group],
            include_turn_rewards=False,
        )


def test_export_ids_are_preserved_as_authoritative_sample_metadata():
    samples, truncated = build_rollout_samples(
        groups=[_export_group()],
        include_turn_rewards=False,
    )

    assert truncated == 0
    assert len(samples) == 1
    assert samples[0].metadata == {
        "raw_rubric_score": 0.5,
        "export_sample_id": "sample",
        "export_group_id": "group",
    }


def test_export_sample_group_id_must_match_its_container_group():
    group = _export_group()
    group.samples[0].group_id = "different-group"

    with pytest.raises(ValueError, match="declares group"):
        build_rollout_samples(
            groups=[group],
            include_turn_rewards=False,
        )
