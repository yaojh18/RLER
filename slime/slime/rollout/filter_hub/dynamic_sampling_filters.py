import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = ["check_direct_judge_variance", "check_reward_nonzero_std"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def check_direct_judge_variance(args, samples: list[Sample], **kwargs):
    """Apply the frozen serial detector, then the required GRPO std gate."""
    predicted = {
        bool((sample.metadata or {}).get("predicted_zero_variance"))
        for sample in samples
    }
    if len(predicted) != 1:
        return DynamicFilterOutput(
            keep=False,
            reason="inconsistent_direct_variance_prediction",
        )
    if predicted == {True}:
        return DynamicFilterOutput(
            keep=False,
            reason="direct_predicted_zero_variance",
        )
    return check_reward_nonzero_std(args, samples, **kwargs)
