from __future__ import annotations

from collections import defaultdict


def convert_samples_to_train_data(args, samples):
    grouped_indices = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped_indices[str(sample.metadata.get("group_id") or sample.group_index or index)].append(index)

    raw_rewards = [float(sample.reward or 0.0) for sample in samples]
    normalized_rewards = list(raw_rewards)
    for indices in grouped_indices.values():
        rewards = [raw_rewards[index] for index in indices]
        mean = sum(rewards) / len(rewards)
        centered = [reward - mean for reward in rewards]
        if args.grpo_std_normalization and len(rewards) > 1:
            variance = sum(value * value for value in centered) / len(centered)
            std = variance**0.5
            if std > 1e-6:
                centered = [value / std for value in centered]
        for local_index, sample_index in enumerate(indices):
            normalized_rewards[sample_index] = centered[local_index]

    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "rewards": normalized_rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
        "loss_masks": [],
    }
    for sample in samples:
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        train_data["loss_masks"].append(sample.loss_mask)

    if any(sample.train_metadata is not None for sample in samples):
        train_data["metadata"] = [sample.train_metadata for sample in samples]
    return train_data
