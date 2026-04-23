from __future__ import annotations

import torch

try:
    from slime.backends.megatron_utils.loss import compute_policy_loss, get_log_probs_and_entropy
except ModuleNotFoundError:
    from slime.slime.backends.megatron_utils.loss import compute_policy_loss, get_log_probs_and_entropy


def compute_loss(args, batch, logits, sum_of_sample_mean):
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
    )

    log_probs = torch.cat(log_probs_and_entropy["log_probs"], dim=0)
    old_log_probs = torch.cat(batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"], dim=0)
    response_advantages = torch.cat(batch["advantages"], dim=0)

    metadata_list = batch.get("metadata") or [None] * len(batch["loss_masks"])
    token_level_advantages = []
    for loss_mask, metadata in zip(batch["loss_masks"], metadata_list, strict=False):
        local_adv = torch.zeros_like(loss_mask, dtype=torch.float32)
        if not metadata:
            token_level_advantages.append(local_adv)
            continue
        turn_rewards = list(metadata.get("turn_rewards") or [])
        turn_masks = list(metadata.get("turn_loss_masks") or [])
        if not turn_rewards or not turn_masks:
            token_level_advantages.append(local_adv)
            continue
        token_reward = torch.zeros_like(loss_mask, dtype=torch.float32)
        for turn_reward, turn_mask in zip(turn_rewards, turn_masks, strict=False):
            token_reward = token_reward + torch.tensor(turn_mask, device=loss_mask.device, dtype=torch.float32) * float(
                turn_reward
            )
        valid = loss_mask > 0
        if valid.any():
            values = token_reward[valid]
            if values.numel() > 1:
                std = values.std(unbiased=False)
                if float(std) > 1e-6:
                    local_adv[valid] = (values - values.mean()) / (std + 1e-6)
                else:
                    local_adv[valid] = values - values.mean()
            else:
                local_adv[valid] = 0.0
        token_level_advantages.append(local_adv)

    aggregated_advantages = response_advantages + torch.cat(token_level_advantages, dim=0)
    ppo_kl = old_log_probs - log_probs
    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, aggregated_advantages, args.eps_clip, args.eps_clip_high)

    pg_loss = sum_of_sample_mean(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)
    entropy = sum_of_sample_mean(torch.cat(log_probs_and_entropy["entropy"], dim=0))

    loss = pg_loss - args.entropy_coef * entropy
    reported = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
        "response_adv_mean": response_advantages.mean().clone().detach(),
        "token_adv_mean": torch.cat(token_level_advantages, dim=0).mean().clone().detach(),
    }
    return loss, reported
