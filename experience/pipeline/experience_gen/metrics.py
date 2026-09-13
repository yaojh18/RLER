from __future__ import annotations

from typing import Any


EPSILON = 1e-12


def ranking_metrics(
    scores: dict[str, float],
    gt_rewards: dict[str, float],
    node_ids: list[str],
) -> dict[str, float]:
    available = [
        node_id
        for node_id in node_ids
        if node_id in scores and node_id in gt_rewards
    ]
    if len(available) < 2:
        raise ValueError("At least two aligned node scores and rewards are required")
    maximum_score = max(float(scores[node_id]) for node_id in available)
    selected = {
        node_id
        for node_id in available
        if abs(float(scores[node_id]) - maximum_score) <= EPSILON
    }
    maximum_reward = max(float(gt_rewards[node_id]) for node_id in available)
    oracle = {
        node_id
        for node_id in available
        if abs(float(gt_rewards[node_id]) - maximum_reward) <= EPSILON
    }
    tie_aware = len(selected & oracle) / len(selected)

    correct = 0.0
    total = 0
    for left_index, left in enumerate(available):
        for right in available[left_index + 1 :]:
            score_delta = float(scores[left]) - float(scores[right])
            reward_delta = float(gt_rewards[left]) - float(gt_rewards[right])
            total += 1
            if abs(reward_delta) <= EPSILON or abs(score_delta) <= EPSILON:
                correct += 0.5
            elif score_delta * reward_delta > 0:
                correct += 1.0
    return {
        "tie_aware_success": tie_aware,
        "pairwise_accuracy": correct / total if total else 0.5,
    }


def metric_delta(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in ("tie_aware_success", "pairwise_accuracy")
    }


def improved(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    epsilon: float = EPSILON,
) -> bool:
    """Final overfit gate: positive tie-aware OR pairwise improvement."""

    return any(value > epsilon for value in metric_delta(candidate, baseline).values())


