from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
from slime.backends.megatron_utils.loss import policy_loss_function


def convert_samples_to_train_data(args, samples):
    if samples and isinstance(samples[0], list):
        samples = [sample for group in samples for sample in group]
    grouped_indices = defaultdict(list)
    for index, sample in enumerate(samples):
        group_key = sample.group_index if sample.group_index is not None else index
        grouped_indices[str(group_key)].append(index)

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    rewards = list(raw_rewards)
    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        for indices in grouped_indices.values():
            group_rewards = torch.tensor([raw_rewards[index] for index in indices], dtype=torch.float).view(1, -1)
            group_rewards = group_rewards - group_rewards.mean(dim=-1, keepdim=True)
            if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
                group_rewards = group_rewards / (group_rewards.std(dim=-1, keepdim=True) + 1e-6)
            for sample_index, reward in zip(indices, group_rewards.flatten().tolist(), strict=False):
                rewards[sample_index] = reward

    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
        "loss_masks": [],
    }
    for sample in samples:
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        assert (
            len(sample.loss_mask) == sample.response_length
        ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        train_data["loss_masks"].append(sample.loss_mask)

    if any(sample.metadata and "raw_reward" in sample.metadata for sample in samples):
        train_data["raw_reward"] = [
            sample.metadata["raw_reward"] if sample.metadata and "raw_reward" in sample.metadata else sample.reward
            for sample in samples
        ]
    if samples and samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]
    if samples and samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]
    if samples and samples[0].rollout_routed_experts is not None:
        train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]
    if args.loss_type == "custom_loss" or any(sample.train_metadata is not None for sample in samples):
        train_data["metadata"] = [
            {**(sample.train_metadata or {}), "response_advantage": rewards[index]}
            for index, sample in enumerate(samples)
        ]
    if any(sample.multimodal_train_inputs is not None for sample in samples):
        train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]
    if samples and samples[0].teacher_log_probs is not None:
        train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]
    return train_data


def compute_rubric_loss(args, batch, logits, sum_of_sample_mean):
    alpha = float(getattr(args, "swe_rtt_alpha", 1.0))
    beta = float(getattr(args, "swe_rtt_beta", 1.0))
    metadata_list = batch.get("metadata")
    token_advantages = []
    for loss_mask, metadata in zip(batch["loss_masks"], metadata_list, strict=False):
        local_advantage = torch.zeros_like(loss_mask, dtype=torch.float32)
        turn_rewards = list((metadata).get("turn_rewards"))
        turn_masks = list((metadata).get("turn_loss_masks"))
        if turn_rewards and turn_masks:
            if len(turn_rewards) != len(turn_masks):
                raise RuntimeError(
                    f"rubric turn reward/mask mismatch: rewards={len(turn_rewards)} masks={len(turn_masks)}"
                )
            turn_values = torch.tensor(turn_rewards, device=loss_mask.device, dtype=torch.float32)
            turn_values = turn_values - turn_values.mean()
            if turn_values.numel() > 1:
                turn_values = turn_values / (turn_values.std() + 1e-6)
            for turn_value, turn_mask in zip(turn_values, turn_masks, strict=False):
                turn_mask_tensor = torch.tensor(turn_mask, device=loss_mask.device, dtype=torch.float32)
                if turn_mask_tensor.shape != loss_mask.shape:
                    raise RuntimeError(
                        f"rubric turn mask shape mismatch: mask={turn_mask_tensor.shape} loss_mask={loss_mask.shape}"
                    )
                local_advantage = local_advantage + turn_mask_tensor * turn_value
            local_advantage = local_advantage * (loss_mask > 0)
        token_advantages.append(local_advantage)

    response_advantages = batch["advantages"]
    combined_advantages = []
    for response_advantage, token_advantage in zip(response_advantages, token_advantages, strict=False):
        if response_advantage.shape != token_advantage.shape:
            raise RuntimeError(
                f"rubric advantage shape mismatch: response={response_advantage.shape} token={token_advantage.shape}"
            )
        combined_advantages.append(
            alpha * response_advantage + beta * token_advantage.to(response_advantage.device, response_advantage.dtype)
        )

    loss, metrics = policy_loss_function(args, {**batch, "advantages": combined_advantages}, logits, sum_of_sample_mean)
    metrics["response_adv_mean"] = torch.cat(response_advantages, dim=0).mean().clone().detach()
    metrics["token_adv_mean"] = torch.cat(token_advantages, dim=0).mean().clone().detach()
    metrics["rtt_alpha"] = torch.tensor(alpha, device=logits.device)
    metrics["rtt_beta"] = torch.tensor(beta, device=logits.device)
    return loss, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run online slime GRPO for SWE-agent policy or rubric data.")
    parser.add_argument("--target", choices=["policy", "rubric"], required=True)
    parser.add_argument("--prompt-data", type=Path, required=True)
    parser.add_argument("--hf-checkpoint", type=Path, required=True)
    parser.add_argument("--load-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--ref-load-dir", type=Path)
    parser.add_argument("--config-path", type=Path, default=Path("/workspace/rler/slime/train_agent/configs/grpo.sh"))
    parser.add_argument("--rollout-batch-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--num-rollout", type=int, default=2)
    parser.add_argument("--actor-num-gpus", type=int, default=2)
    parser.add_argument("--rollout-num-gpus", type=int, default=2)
    parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=2)
    parser.add_argument("--ray-num-cpus", type=int, default=32)
    parser.add_argument("--student-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--search-output-root", type=Path, required=True)
    parser.add_argument("--search-m", type=int)
    parser.add_argument("--search-k", type=int)
    parser.add_argument("--search-p", type=int)
    parser.add_argument("--search-max-rounds", type=int)
    parser.add_argument("--search-step-limit", type=int)
    parser.add_argument("--search-workers", type=int, default=1)
    parser.add_argument("--wandb-dir", type=Path)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE") or ("online" if os.environ.get("WANDB_API_KEY") else "offline"))
    parser.add_argument("--wandb-key", default=os.environ.get("WANDB_API_KEY"))
    parser.add_argument("--wandb-team", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-project", default="swe-agent-grpo")
    parser.add_argument("--wandb-group")
    args = parser.parse_args(argv)

    total_gpus = args.actor_num_gpus + args.rollout_num_gpus
    ref_load_dir = args.ref_load_dir or args.load_dir
    wandb_dir = args.wandb_dir or (args.save_dir / "wandb")
    global_batch_size = args.global_batch_size or 1
    wandb_args = ""
    if args.wandb_mode != "disabled":
        pieces = [
            "--use-wandb",
            "--wandb-mode", args.wandb_mode,
            "--wandb-project", args.wandb_project,
            "--wandb-group", args.wandb_group or f"{args.target}-grpo",
            "--wandb-dir", str(wandb_dir),
            "--disable-wandb-random-suffix",
        ]
        if args.wandb_team:
            pieces.extend(["--wandb-team", args.wandb_team])
        wandb_args = shlex.join(pieces)

    command = f"""
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/rler/slime:/workspace/rler:/workspace/rler/agent:/root/Megatron-LM:${{PYTHONPATH:-}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${{CUDA_DEVICE_MAX_CONNECTIONS:-1}}"
export SWE_AGENT_GRPO_TARGET={shlex.quote(args.target)}
export SWE_AGENT_GRPO_OUTPUT_ROOT={shlex.quote(str(args.search_output_root))}
export SWE_AGENT_GRPO_MODEL_NAME={shlex.quote(args.student_model)}
export SWE_AGENT_GRPO_WORKERS={args.search_workers}
export SWE_AGENT_GRPO_M={"" if args.search_m is None else args.search_m}
export SWE_AGENT_GRPO_K={"" if args.search_k is None else args.search_k}
export SWE_AGENT_GRPO_P={"" if args.search_p is None else args.search_p}
export SWE_AGENT_GRPO_MAX_ROUNDS={"" if args.search_max_rounds is None else args.search_max_rounds}
export SWE_AGENT_GRPO_STEP_LIMIT={"" if args.search_step_limit is None else args.search_step_limit}
export SWE_AGENT_PYTHON="${{SWE_AGENT_PYTHON:-/workspace/rler/agent/.venv/bin/python}}"
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT
pkill -9 sglang >/dev/null 2>&1 || true
ray stop --force >/dev/null 2>&1 || true
cd /workspace/rler/slime
source /workspace/rler/slime/train_agent/configs/qwen3.5-9B.sh
source {shlex.quote(str(args.config_path))}
if [ {shlex.quote(args.target)} = "rubric" ]; then
  GRPO_TARGET_ARGS=("${{GRPO_RUBRIC_ARGS[@]}}")
else
  GRPO_TARGET_ARGS=("${{GRPO_POLICY_ARGS[@]}}")
fi
for CHECKPOINT_DIR in {shlex.quote(str(args.load_dir))} {shlex.quote(str(ref_load_dir))}; do
  TRACKER="${{CHECKPOINT_DIR}}/latest_checkpointed_iteration.txt"
  if [ -f "${{TRACKER}}" ] && [ "$(tr -d '[:space:]' < "${{TRACKER}}")" = "0" ] && [ -d "${{CHECKPOINT_DIR}}/iter_0000000" ]; then
    if [ ! -e "${{CHECKPOINT_DIR}}/iter_0000001" ]; then
      cp -al "${{CHECKPOINT_DIR}}/iter_0000000" "${{CHECKPOINT_DIR}}/iter_0000001" 2>/dev/null || cp -a "${{CHECKPOINT_DIR}}/iter_0000000" "${{CHECKPOINT_DIR}}/iter_0000001"
    fi
    printf '1\\n' > "${{TRACKER}}"
  fi
done
RAY_TMPDIR="/tmp/ray-swe-{args.target}-grpo-$$"
mkdir -p "${{RAY_TMPDIR}}"
ray start --head --node-ip-address 127.0.0.1 --num-gpus {total_gpus} --num-cpus {args.ray_num_cpus} --temp-dir "${{RAY_TMPDIR}}" --disable-usage-stats >/dev/null
python3 train_async.py \\
  --actor-num-nodes 1 \\
  --actor-num-gpus-per-node {args.actor_num_gpus} \\
  --rollout-num-gpus {args.rollout_num_gpus} \\
  --rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} \\
  --num-gpus-per-node {total_gpus} \\
  "${{MODEL_ARGS[@]}}" \\
  --hf-checkpoint {shlex.quote(str(args.hf_checkpoint))} \\
  --load {shlex.quote(str(args.load_dir))} \\
  --ref-load {shlex.quote(str(ref_load_dir))} \\
  --save {shlex.quote(str(args.save_dir))} \\
  --num-rollout {args.num_rollout} \\
  --prompt-data {shlex.quote(str(args.prompt_data))} \\
  --rollout-batch-size {args.rollout_batch_size} \\
  --global-batch-size {global_batch_size} \\
  --sglang-served-model-name {shlex.quote(args.student_model)} \\
  "${{GRPO_COMMON_ARGS[@]}}" \\
  "${{GRPO_TARGET_ARGS[@]}}" \\
  "${{GRPO_PARALLEL_ARGS[@]}}" \\
  "${{GRPO_RECOMPUTE_ARGS[@]}}" \\
  "${{GRPO_OPTIMIZER_ARGS[@]}}" \\
  "${{GRPO_SGLANG_ARGS[@]}}" \\
  "${{GRPO_MISC_ARGS[@]}}" \\
  {wandb_args}
"""
    subprocess.run(["bash", "-lc", command], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
