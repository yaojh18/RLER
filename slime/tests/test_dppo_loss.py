"""CPU tests for the DPPO binary trust-region policy loss."""

import math

import pytest
import torch

from slime.utils.ppo_utils import compute_binary_divergence, compute_dppo_loss

NUM_GPUS = 0


def test_binary_tv_matches_sampled_token_bernoulli_distance():
    behavior = torch.log(torch.tensor([0.2, 0.7, 0.4]))
    policy = torch.log(torch.tensor([0.5, 0.6, 0.1]))
    response_mask = torch.tensor([True, False, True])

    divergence = compute_binary_divergence(behavior, policy, response_mask, "tv")

    torch.testing.assert_close(divergence, torch.tensor([0.3, 0.0, 0.3]))


def test_binary_kl_matches_bernoulli_kl():
    behavior_probs = torch.tensor([0.2, 0.7])
    policy_probs = torch.tensor([0.5, 0.6])
    response_mask = torch.ones(2, dtype=torch.bool)

    divergence = compute_binary_divergence(
        behavior_probs.log(),
        policy_probs.log(),
        response_mask,
        "kl",
    )
    expected = behavior_probs * (behavior_probs.log() - policy_probs.log()) + (
        1.0 - behavior_probs
    ) * ((1.0 - behavior_probs).log() - (1.0 - policy_probs).log())

    torch.testing.assert_close(divergence, expected)


def test_dppo_masks_only_updates_that_move_farther_outside_trust_region():
    behavior_probs = torch.full((6,), 0.2)
    ratios = torch.tensor([2.0, 0.5, 0.5, 2.0, 1.1, 2.0])
    policy_probs = behavior_probs * ratios
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    response_mask = torch.tensor([True, True, True, True, True, False])

    losses, keep_mask, divergence = compute_dppo_loss(
        behavior_probs.log(),
        policy_probs.log(),
        advantages,
        response_mask,
        divergence_type="tv",
        divergence_threshold=0.05,
    )

    expected_keep_mask = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    torch.testing.assert_close(keep_mask, expected_keep_mask)
    torch.testing.assert_close(losses, -advantages * ratios * expected_keep_mask)
    torch.testing.assert_close(divergence, torch.tensor([0.2, 0.1, 0.1, 0.2, 0.02, 0.0]))


def test_dppo_blocked_tokens_have_zero_policy_gradient():
    behavior_log_probs = torch.log(torch.full((4,), 0.2))
    ratios = torch.tensor([2.0, 0.5, 0.5, 2.0])
    policy_log_probs = torch.tensor(
        [math.log(0.2 * ratio) for ratio in ratios],
        requires_grad=True,
    )
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0])

    losses, keep_mask, _ = compute_dppo_loss(
        behavior_log_probs,
        policy_log_probs,
        advantages,
        torch.ones(4, dtype=torch.bool),
        divergence_type="tv",
        divergence_threshold=0.05,
    )
    losses.sum().backward()

    expected_gradient = -advantages * ratios * keep_mask
    torch.testing.assert_close(policy_log_probs.grad, expected_gradient)


def test_dppo_rejects_unknown_divergence():
    values = torch.zeros(1)
    with pytest.raises(ValueError, match="Unknown DPPO divergence type"):
        compute_binary_divergence(values, values, torch.ones(1, dtype=torch.bool), "js")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
