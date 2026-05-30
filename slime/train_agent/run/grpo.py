from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp
from slime.backends.megatron_utils.loss import policy_loss_function


def _apply_deferred_experience_update_rewards(samples) -> None:
    rewards_by_scope_and_instance: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    deferred_by_scope: dict[str, list] = defaultdict(list)
    for sample in samples:
        metadata = sample.metadata or {}
        scope = metadata.get("scope")
        stage = metadata.get("stage")
        deferred_kind = metadata.get("deferred_reward_kind")
        instance_id = metadata.get("instance_id") or metadata.get("source_instance_id") or "unknown"
        if deferred_kind:
            if isinstance(scope, str):
                deferred_by_scope[scope].append(sample)
            continue
        if isinstance(scope, str) and stage in {"retrieve", "generate"}:
            rewards_by_scope_and_instance[scope][str(instance_id)].append(float(sample.reward or 0.0))

    for scope, deferred_samples in deferred_by_scope.items():
        per_instance_means = [
            sum(values) / len(values)
            for values in rewards_by_scope_and_instance.get(scope, {}).values()
            if values
        ]
        if not per_instance_means:
            continue
        deferred_reward = float(sum(per_instance_means) / len(per_instance_means))
        for sample in deferred_samples:
            metadata = dict(sample.metadata or {})
            metadata["raw_deferred_reward"] = float(sample.reward or 0.0)
            metadata["resolved_deferred_reward"] = deferred_reward
            sample.metadata = metadata
            sample.reward = deferred_reward


def convert_samples_to_train_data(args, samples):
    if samples and isinstance(samples[0], list):
        samples = [sample for group in samples for sample in group]
    _apply_deferred_experience_update_rewards(samples)
    grouped_indices = defaultdict(list)
    for index, sample in enumerate(samples):
        group_key = sample.group_index if sample.group_index is not None else index
        grouped_indices[str(group_key)].append(index)

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    rewards = list(raw_rewards)
    # is_dummy samples (infra-failure placeholders with token_ids=[0,0],
    # loss_mask=[0,0], reward=0.0) MUST be excluded from group reward
    # mean/std — otherwise they drag the baseline toward 0 and inflate
    # std, biasing every other sample's advantage. They still occupy a
    # slot in the group to keep group_size consistent for downstream
    # bookkeeping; their loss_mask is zeroed below so they contribute
    # no gradient.
    is_dummy_flags: list[bool] = [
        bool(sample.metadata and sample.metadata.get("is_dummy"))
        for sample in samples
    ]
    # Track samples in degenerate groups (singleton or all-same reward).
    # These carry no GRPO signal and would produce NaN in whitening:
    #   * n=1: PyTorch unbiased std of a single element is NaN
    #          → 0 / (NaN + 1e-6) = NaN, poisons training (51963 step 0)
    #   * std=0 (all-same): 0 / (0 + 1e-6) = 0, no NaN but zero signal
    # In both cases we zero the loss_mask so the sample contributes nothing
    # to the policy update. Caused by oversized-drop reducing a group's
    # surviving sample count below 2.
    samples_to_drop: set[int] = set()
    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        for indices in grouped_indices.values():
            real_indices = [i for i in indices if not is_dummy_flags[i]]
            dummy_indices = [i for i in indices if is_dummy_flags[i]]
            n = len(real_indices)
            if n < 2:
                # Insufficient real samples for GRPO comparison.
                # naive_record_to_bundle drops most of these upstream, but
                # singleton survivors of partial-drop still land here.
                for idx in indices:
                    rewards[idx] = 0.0
                    samples_to_drop.add(idx)
                continue
            real_group_raw = [raw_rewards[i] for i in real_indices]
            group_rewards = torch.tensor(real_group_raw, dtype=torch.float).view(1, -1)
            # unbiased=False is the population std — finite even for tiny groups.
            std = group_rewards.std(dim=-1, keepdim=True, unbiased=False)
            if torch.isnan(std).any() or (std <= 1e-9).all():
                # all-same reward → no signal, drop
                for idx in indices:
                    rewards[idx] = 0.0
                    samples_to_drop.add(idx)
                continue
            centered = group_rewards - group_rewards.mean(dim=-1, keepdim=True)
            if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
                centered = centered / (std + 1e-6)
            for sample_index, reward in zip(real_indices, centered.flatten().tolist(), strict=False):
                rewards[sample_index] = reward
            # Dummies: zero advantage + drop (loss_mask zeroed below).
            for idx in dummy_indices:
                rewards[idx] = 0.0
                samples_to_drop.add(idx)

    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
        "loss_masks": [],
    }
    for idx, sample in enumerate(samples):
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        assert (
            len(sample.loss_mask) == sample.response_length
        ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        if sample.remove_sample or idx in samples_to_drop:
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
    if metadata_list is None:
        raise RuntimeError("rubric custom loss requires per-sample metadata")

    token_advantages = []
    max_seq_lens = batch.get("max_seq_lens")
    for index, (loss_mask, metadata, total_length, response_length) in enumerate(
        zip(batch["loss_masks"], metadata_list, batch["total_lengths"], batch["response_lengths"], strict=False)
    ):
        full_token_advantage = torch.zeros_like(loss_mask, dtype=torch.float32)
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
                full_token_advantage = full_token_advantage + turn_mask_tensor * turn_value
            full_token_advantage = full_token_advantage * (loss_mask > 0)
        max_seq_len = max_seq_lens[index] if max_seq_lens is not None else None
        token_advantages.append(
            slice_log_prob_with_cp(
                full_token_advantage,
                total_length,
                response_length,
                args.qkv_format,
                max_seq_len,
            )
        )

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
    response_values = torch.cat(response_advantages, dim=0)
    token_values = torch.cat(token_advantages, dim=0)
    metrics["response_adv_mean"] = (
        response_values.mean().clone().detach() if response_values.numel() else torch.zeros((), device=logits.device)
    )
    metrics["token_adv_mean"] = (
        token_values.mean().clone().detach() if token_values.numel() else torch.zeros((), device=logits.device)
    )
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
    parser.add_argument("--rler-root", type=Path, default=Path("/workspace/rler"),
                        help="Root dir of the RLER repo as seen by the runtime (used to build PYTHONPATH and cd into slime). "
                             "Override when not running in the original docker layout (e.g. inside pyxis with --container-mounts to a Lustre path).")
    parser.add_argument("--rollout-function-path", default="train_agent.collect_grpo_rollout.generate_rollout")
    parser.add_argument("--num-rollout", type=int)
    parser.add_argument("--rollout-batch-size", type=int, default=2)
    parser.add_argument("--over-sampling-batch-size", type=int,
                        help="If set, slime pulls this many prompts per rollout cycle "
                             "(must be >= --rollout-batch-size). Long-tail buffer.")
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--actor-num-gpus", type=int, default=2,
                        help="Total actor GPUs across all nodes")
    parser.add_argument("--rollout-num-gpus", type=int, default=2,
                        help="Total rollout (sglang) GPUs across all nodes")
    parser.add_argument("--num-nodes", type=int, default=1,
                        help="Total number of physical nodes (>=1). For >1, "
                        "an external Ray cluster must already be running on "
                        "this machine — set --ray-external.")
    parser.add_argument("--num-gpus-per-node", type=int, default=8,
                        help="GPUs per physical node.")
    parser.add_argument("--actor-num-nodes", type=int,
                        help="Number of nodes for actor (defaults to "
                        "--num-nodes when actor uses every node).")
    parser.add_argument("--ray-external", action="store_true",
                        help="Skip the inline 'ray start --head' (cluster is "
                        "brought up externally, e.g. by SLURM srun).")
    parser.add_argument("--context-parallel-size", type=int)
    parser.add_argument("--tensor-model-parallel-size", type=int)
    parser.add_argument("--max-tokens-per-gpu", type=int)
    parser.add_argument("--log-probs-chunk-size", type=int)
    parser.add_argument("--sglang-context-length", type=int)
    parser.add_argument("--rollout-instance-workers", type=int, default=2)
    parser.add_argument("--ray-num-cpus", type=int, default=32)
    parser.add_argument("--student-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--search-output-root", type=Path, required=True)
    parser.add_argument("--search-m", type=int)
    parser.add_argument("--search-n", type=int)
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
    parser.add_argument(
        "--use-tis",
        action="store_true",
        default=False,
        help=(
            "Forwarded to slime train_async.py. Enable Truncated Importance "
            "Sampling for off-policy correction "
            "(https://fengyao.notion.site/off-policy-rl)."
        ),
    )
    parser.add_argument(
        "--tis-clip",
        type=float,
        default=None,
        help="Forwarded to slime train_async.py. TIS upper clip C (default 2.0 inside slime).",
    )
    parser.add_argument(
        "--use-rollout-logprobs",
        action="store_true",
        default=False,
        help=(
            "Forwarded to slime train_async.py. Use sampling-time rollout "
            "log-probs as the PPO 'old' reference instead of running a "
            "separate Megatron compute_log_prob pass. Mutually exclusive "
            "with --use-tis (asserted in slime arguments.py)."
        ),
    )
    parser.add_argument(
        "--dynamic-sampling-filter-path",
        type=str,
        default=None,
        help=(
            "Forwarded to slime train_async.py. Import path to a function that "
            "decides per-group whether to keep or drop the M siblings (e.g. "
            "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
            "drops groups whose rewards have zero std → zero gradient)."
        ),
    )
    parser.add_argument(
        "--save-debug-train-data",
        type=str,
        default=None,
        help=(
            "Forwarded to slime train_async.py. Path template (e.g. "
            "'/path/{rollout_id}_{rank}.pt') for per-rollout per-rank "
            "rollout_data torch.save dumps — used for NaN post-mortem."
        ),
    )
    parser.add_argument(
        "--save-debug-rollout-data",
        type=str,
        default=None,
        help=(
            "Forwarded to slime train_async.py. Path template (e.g. "
            "'/path/{rollout_id}.pt') for per-rollout sample dumps captured "
            "at the rollout manager (one file per rollout, all samples). "
            "Complements --save-debug-train-data which dumps the per-rank "
            "training data after group-by-prompt + advantage compute."
        ),
    )
    args = parser.parse_args(argv)

    total_gpus = args.actor_num_gpus + args.rollout_num_gpus
    actor_num_nodes = args.actor_num_nodes or args.num_nodes
    if args.actor_num_gpus % actor_num_nodes != 0:
        raise SystemExit(
            f"--actor-num-gpus ({args.actor_num_gpus}) must be divisible "
            f"by --actor-num-nodes ({actor_num_nodes})"
        )
    actor_gpus_per_node = args.actor_num_gpus // actor_num_nodes
    ref_load_dir = args.ref_load_dir or args.load_dir
    wandb_dir = args.wandb_dir or (args.save_dir / "wandb")
    global_batch_size = shlex.quote(str(args.global_batch_size or args.rollout_batch_size))
    num_rollout_args = shlex.join(["--num-rollout", str(args.num_rollout)]) if args.num_rollout is not None else ""
    oversample_arg = (
        shlex.join(["--over-sampling-batch-size", str(args.over_sampling_batch_size)])
        if args.over_sampling_batch_size is not None else ""
    )
    dynamic_filter_arg = (
        shlex.join(["--dynamic-sampling-filter-path", args.dynamic_sampling_filter_path])
        if args.dynamic_sampling_filter_path else ""
    )
    tis_arg_parts: list[str] = []
    if args.use_tis:
        tis_arg_parts.append("--use-tis")
    if args.tis_clip is not None:
        tis_arg_parts.extend(["--tis-clip", str(args.tis_clip)])
    tis_arg = shlex.join(tis_arg_parts) if tis_arg_parts else ""
    rollout_logprobs_arg = "--use-rollout-logprobs" if args.use_rollout_logprobs else ""
    save_debug_arg = (
        shlex.join(["--save-debug-train-data", args.save_debug_train_data])
        if args.save_debug_train_data else ""
    )
    save_debug_rollout_arg = (
        shlex.join(["--save-debug-rollout-data", args.save_debug_rollout_data])
        if args.save_debug_rollout_data else ""
    )
    override_lines = []
    if args.context_parallel_size is not None:
        override_lines.append(f"GRPO_PARALLEL_ARGS+=(--context-parallel-size {args.context_parallel_size})")
    if args.tensor_model_parallel_size is not None:
        override_lines.append(f"GRPO_PARALLEL_ARGS+=(--tensor-model-parallel-size {args.tensor_model_parallel_size})")
    if args.max_tokens_per_gpu is not None:
        override_lines.append(f"GRPO_MISC_ARGS+=(--max-tokens-per-gpu {args.max_tokens_per_gpu})")
    if args.log_probs_chunk_size is not None:
        override_lines.append(f"GRPO_COMMON_ARGS+=(--log-probs-chunk-size {args.log_probs_chunk_size})")
    if args.sglang_context_length is not None:
        override_lines.append(f"GRPO_SGLANG_ARGS+=(--sglang-context-length {args.sglang_context_length})")
    config_overrides = "\n".join(override_lines)
    wandb_args = ""
    if args.wandb_mode != "disabled":
        # Default the wandb group/run-name to the SLURM job name when running
        # under SLURM (so each launched job shows up in WandB as its job name
        # like "grpo-naive-cp4vftis-56612" rather than every run colliding on
        # "policy-grpo"). Falls back to <target>-grpo for non-SLURM launches.
        slurm_job_name = os.environ.get("SLURM_JOB_NAME") or ""
        slurm_job_id = os.environ.get("SLURM_JOB_ID") or ""
        if args.wandb_group:
            wandb_group = args.wandb_group
        elif slurm_job_name:
            wandb_group = (
                f"{slurm_job_name}-{slurm_job_id}" if slurm_job_id else slurm_job_name
            )
        else:
            wandb_group = f"{args.target}-grpo"
        pieces = [
            "--use-wandb",
            "--wandb-mode", args.wandb_mode,
            "--wandb-project", args.wandb_project,
            "--wandb-group", wandb_group,
            "--wandb-dir", str(wandb_dir),
            "--disable-wandb-random-suffix",
        ]
        if args.wandb_team:
            pieces.extend(["--wandb-team", args.wandb_team])
        wandb_args = shlex.join(pieces)

    if args.ray_external:
        ray_start_line = "# external ray cluster expected (--ray-external)"
        ray_stop_line = "# skipping ray stop / pkill — external cluster owned by launcher"
        ray_trap_line = "# skipping EXIT trap — external cluster owned by launcher"
    else:
        ray_start_line = (
            f"ray start --head --node-ip-address 127.0.0.1 "
            f"--num-gpus {total_gpus} --num-cpus {args.ray_num_cpus} "
            f"--temp-dir \"${{RAY_TMPDIR}}\" --disable-usage-stats >/dev/null"
        )
        ray_stop_line = "pkill -9 sglang >/dev/null 2>&1 || true\nray stop --force >/dev/null 2>&1 || true"
        ray_trap_line = "trap 'ray stop --force >/dev/null 2>&1 || true' EXIT"

    rler_root = str(args.rler_root)
    # Preserve any PYTHONPATH set by the launcher (e.g. for jsonlines / docker
    # / other agent runtime deps installed into a Lustre-side site dir).
    extra_pp = os.environ.get("EXTRA_PYTHONPATH", "")
    pythonpath = f"{rler_root}/slime:{rler_root}:{rler_root}/agent:/root/Megatron-LM"
    if extra_pp:
        pythonpath = f"{extra_pp}:{pythonpath}"
    command = f"""
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="{pythonpath}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export SWE_AGENT_GRPO_TARGET={shlex.quote(args.target)}
export SWE_AGENT_GRPO_OUTPUT_ROOT={shlex.quote(str(args.search_output_root))}
export SWE_AGENT_GRPO_MODEL_NAME={shlex.quote(args.student_model)}
export SWE_AGENT_GRPO_WORKERS={args.search_workers}
export SWE_AGENT_GRPO_M={"" if args.search_m is None else args.search_m}
export SWE_AGENT_GRPO_N={"" if args.search_n is None else args.search_n}
export SWE_AGENT_GRPO_K={"" if args.search_k is None else args.search_k}
export SWE_AGENT_GRPO_P={"" if args.search_p is None else args.search_p}
export SWE_AGENT_GRPO_MAX_ROUNDS={"" if args.search_max_rounds is None else args.search_max_rounds}
export SWE_AGENT_GRPO_STEP_LIMIT={"" if args.search_step_limit is None else args.search_step_limit}
export SWE_AGENT_GRPO_INSTANCE_WORKERS={args.rollout_instance_workers}
export SWE_AGENT_PYTHON="{rler_root}/agent/.venv/bin/python"
{ray_trap_line}
{ray_stop_line}
cd {rler_root}/slime
source {rler_root}/slime/train_agent/configs/qwen3.5-9B.sh
source {shlex.quote(str(args.config_path))}
{config_overrides}
if [ {shlex.quote(args.target)} = "rubric" ]; then
  GRPO_TARGET_ARGS=("${{GRPO_RUBRIC_ARGS[@]}}")
else
  GRPO_TARGET_ARGS=("${{GRPO_POLICY_ARGS[@]}}")
fi
GLOBAL_BATCH_SIZE={global_batch_size}
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
{ray_start_line}
python3 train_async.py \\
  --actor-num-nodes {actor_num_nodes} \\
  --actor-num-gpus-per-node {actor_gpus_per_node} \\
  --rollout-num-gpus {args.rollout_num_gpus} \\
  --num-gpus-per-node {args.num_gpus_per_node} \\
  "${{MODEL_ARGS[@]}}" \\
  --hf-checkpoint {shlex.quote(str(args.hf_checkpoint))} \\
  --load {shlex.quote(str(args.load_dir))} \\
  --ref-load {shlex.quote(str(ref_load_dir))} \\
  --save {shlex.quote(str(args.save_dir))} \\
  --prompt-data {shlex.quote(str(args.prompt_data))} \\
  --rollout-batch-size {args.rollout_batch_size} \\
  --global-batch-size "${{GLOBAL_BATCH_SIZE}}" \\
  {oversample_arg} \\
  --sglang-served-model-name {shlex.quote(args.student_model)} \\
  "${{GRPO_COMMON_ARGS[@]}}" \\
  --rollout-function-path {shlex.quote(args.rollout_function_path)} \\
  {num_rollout_args} \\
  "${{GRPO_TARGET_ARGS[@]}}" \\
  "${{GRPO_PARALLEL_ARGS[@]}}" \\
  "${{GRPO_RECOMPUTE_ARGS[@]}}" \\
  "${{GRPO_OPTIMIZER_ARGS[@]}}" \\
  "${{GRPO_ROLLOUT_ARGS[@]}}" \\
  "${{GRPO_SGLANG_ARGS[@]}}" \\
  "${{GRPO_MISC_ARGS[@]}}" \\
  {dynamic_filter_arg} \\
  {tis_arg} \\
  {rollout_logprobs_arg} \\
  {save_debug_arg} \\
  {save_debug_rollout_arg} \\
  {wandb_args}
"""
    subprocess.run(["bash", "-lc", command], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
