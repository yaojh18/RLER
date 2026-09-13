from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
from slime.rollout.data_source import (
    ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
    ROLLOUT_COLLECTOR_STATE_METADATA_KEY,
)


def convert_samples_to_train_data(args, samples):
    if samples and isinstance(samples[0], list):
        samples = [sample for group in samples for sample in group]
    grouped_indices = defaultdict(list)
    for index, sample in enumerate(samples):
        group_key = sample.group_index if sample.group_index is not None else index
        grouped_indices[str(group_key)].append(index)

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    rewards = list(raw_rewards)
    # Track samples in degenerate groups (singleton or all-same reward).
    # These carry no GRPO signal and would produce NaN in whitening:
    #   * n=1: PyTorch unbiased std of a single element is NaN
    #          → 0 / (NaN + 1e-6) = NaN, poisons training (51963 step 0)
    #   * std=0 (all-same): 0 / (0 + 1e-6) = 0, no NaN but zero signal
    # In both cases we zero the loss_mask so the sample contributes nothing
    # to the policy update.
    samples_to_drop: set[int] = set()
    if (
        args.advantage_estimator == "grpo"
        and args.rewards_normalization
    ):
        for indices in grouped_indices.values():
            if len(indices) < 2:
                for idx in indices:
                    rewards[idx] = 0.0
                    samples_to_drop.add(idx)
                continue
            group_rewards = torch.tensor(
                [raw_rewards[i] for i in indices], dtype=torch.float
            ).view(1, -1)
            # unbiased=False is the population std — finite even for tiny groups.
            std = group_rewards.std(dim=-1, keepdim=True, unbiased=False)
            if torch.isnan(std).any() or (std <= 1e-9).all():
                # all-same reward → no signal, drop
                for idx in indices:
                    rewards[idx] = 0.0
                    samples_to_drop.add(idx)
                continue
            centered = group_rewards - group_rewards.mean(dim=-1, keepdim=True)
            if args.grpo_std_normalization:
                centered = centered / (std + 1e-6)
            for sample_index, reward in zip(indices, centered.flatten().tolist(), strict=False):
                rewards[sample_index] = reward

    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
        "rollout_ids": [],
        "loss_masks": [],
    }
    rollout_ids = [sample.rollout_id for sample in samples]
    existing_rollout_ids = set(rollout_id for rollout_id in rollout_ids if rollout_id is not None)
    next_rollout_id = 0
    for rollout_id in rollout_ids:
        if rollout_id is None:
            while next_rollout_id in existing_rollout_ids:
                next_rollout_id += 1
            rollout_id = next_rollout_id
            existing_rollout_ids.add(rollout_id)
        train_data["rollout_ids"].append(rollout_id)

    for idx, sample in enumerate(samples):
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        assert (
            len(sample.loss_mask) == sample.response_length
        ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        if sample.remove_sample or idx in samples_to_drop:
            sample.loss_mask = [0] * sample.response_length
        train_data["loss_masks"].append(sample.loss_mask)

    rollout_total_mask: dict[int, int] = {}
    for rollout_id, loss_mask in zip(train_data["rollout_ids"], train_data["loss_masks"], strict=True):
        rollout_total_mask[rollout_id] = rollout_total_mask.get(rollout_id, 0) + sum(loss_mask)
    train_data["rollout_mask_sums"] = [
        rollout_total_mask[rollout_id] for rollout_id in train_data["rollout_ids"]
    ]

    if any(sample.metadata and "raw_reward" in sample.metadata for sample in samples):
        train_data["raw_reward"] = [
            sample.metadata["raw_reward"] if sample.metadata and "raw_reward" in sample.metadata else sample.reward
            for sample in samples
        ]
    if samples and samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]
    return train_data


def _dataset_state_has_exact_collector_position(
    path: Path,
    checkpoint_id: int,
) -> bool:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if not isinstance(state, dict):
        return False
    if int(state.get("checkpoint_schema_version", -1)) != (
        ROLLOUT_CHECKPOINT_SCHEMA_VERSION
    ):
        return False
    if int(state.get("checkpoint_rollout_id", -1)) != int(
        checkpoint_id
    ):
        return False
    metadata = state.get("metadata")
    return (
        isinstance(metadata, dict)
        and isinstance(
            metadata.get(ROLLOUT_COLLECTOR_STATE_METADATA_KEY),
            dict,
        )
    )


def _requires_exact_collector_resume(args) -> bool:
    return any(
        (
            args.train_instance_budget is not None,
            args.eval_instance_interval is not None,
            bool(args.require_train_instance_budget_exhaustion),
        )
    )


def _complete_checkpoint_ids(
    checkpoint_root: Path,
    *,
    require_collector_state: bool = False,
) -> list[int]:
    """Return model checkpoints that have a paired rollout cursor.

    Megatron writes its model checkpoint before the rollout data source writes
    the source-instance cursor.  A Slurm timeout can therefore leave the
    tracker pointing at a model-only checkpoint.  Such a checkpoint is not a
    valid recovery point for online RL and must never be selected silently.
    """

    complete: list[int] = []
    rollout_root = checkpoint_root / "rollout"
    for checkpoint_dir in checkpoint_root.glob("iter_*"):
        match = re.fullmatch(r"iter_(\d+)", checkpoint_dir.name)
        if match is None or not checkpoint_dir.is_dir():
            continue
        checkpoint_id = int(match.group(1))
        dataset_state = (
            rollout_root
            / f"global_dataset_state_dict_{checkpoint_id}.pt"
        )
        if not dataset_state.is_file():
            continue
        if require_collector_state and not (
            _dataset_state_has_exact_collector_position(
                dataset_state,
                checkpoint_id,
            )
        ):
            continue
        complete.append(checkpoint_id)
    return sorted(set(complete))


def _configure_auto_resume(args, parser: argparse.ArgumentParser) -> int | None:
    """Resume from the newest complete checkpoint under ``--save-dir``.

    Returns the selected rollout id, or ``None`` for a fresh run.  When the
    Megatron tracker is ahead of the newest paired data cursor, it is moved
    back atomically to that last complete recovery point.  The newer
    model-only directory is retained for diagnosis.
    """

    if not args.auto_resume:
        return None
    tracker = args.save_dir / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        return None
    if not tracker.is_file():
        parser.error(
            f"--auto-resume checkpoint tracker is not a file: {tracker}"
        )
    tracker_text = ""
    try:
        tracker_text = tracker.read_text(encoding="utf-8").strip()
        tracked_checkpoint_id = int(tracker_text)
    except (OSError, ValueError):
        # A fresh converted checkpoint may use the literal "release".
        # SAVE_DIR should normally be empty, so only accept that marker when
        # no online-training checkpoint/cursor exists beside it.
        if (
            tracker_text == "release"
            and not _complete_checkpoint_ids(args.save_dir)
        ):
            return None
        parser.error(
            "--auto-resume found a non-numeric online checkpoint tracker: "
            f"{tracker}"
        )
    complete_ids = [
        checkpoint_id
        for checkpoint_id in _complete_checkpoint_ids(
            args.save_dir,
            require_collector_state=_requires_exact_collector_resume(args),
        )
        if checkpoint_id <= tracked_checkpoint_id
    ]
    if not complete_ids:
        parser.error(
            "--auto-resume found a numeric Megatron tracker but no model/data "
            f"checkpoint pair at or before {tracked_checkpoint_id}: "
            f"{args.save_dir}"
        )
    resume_rollout_id = complete_ids[-1]
    if resume_rollout_id != tracked_checkpoint_id:
        temporary = tracker.with_name(
            f".{tracker.name}.{os.getpid()}.resume.tmp"
        )
        try:
            temporary.write_text(
                f"{resume_rollout_id}\n",
                encoding="utf-8",
            )
            os.replace(temporary, tracker)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    args.load_dir = args.save_dir
    args.resume = True
    return resume_rollout_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run online Slime DPPO for SWE-agent policy data."
    )
    parser.add_argument("--prompt-data", type=Path, required=True)
    parser.add_argument("--hf-checkpoint", type=Path, required=True)
    parser.add_argument("--load-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    resume_mode = parser.add_mutually_exclusive_group()
    resume_mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted online-training run from --load-dir. "
            "Optimizer/RNG state and start_rollout_id are restored from the "
            "Megatron checkpoint, and the matching global-dataset cursor must "
            "exist under --load-dir/rollout."
        ),
    )
    resume_mode.add_argument(
        "--auto-resume",
        action="store_true",
        help=(
            "On a Slurm restart, automatically load the newest complete "
            "model/global-dataset checkpoint pair from --save-dir. Start "
            "fresh when no online checkpoint exists."
        ),
    )
    parser.add_argument("--ref-load-dir", type=Path)
    parser.add_argument("--rler-root", type=Path, default=Path("/workspace/rler"),
                        help="Root dir of the RLER repo as seen by the runtime (used to build PYTHONPATH and cd into slime). "
                             "Override when not running in the original docker layout (e.g. inside pyxis with --container-mounts to a Lustre path).")
    parser.add_argument("--rollout-function-path", required=True)
    parser.add_argument(
        "--eval-interval",
        type=int,
        help="Run the SWE terminal validation protocol every N rollout steps.",
    )
    parser.add_argument(
        "--eval-instance-interval",
        type=int,
        help=(
            "Run validation after every N attempted source instances. "
            "Invalid and dynamically filtered instances count, and this "
            "cadence supersedes update-based --eval-interval scheduling."
        ),
    )
    parser.add_argument(
        "--defer-instance-validation",
        action="store_true",
        help=(
            "At each source-instance validation cadence, save and mark the "
            "checkpoint but defer rollout/evaluation until after training."
        ),
    )
    eval_source = parser.add_mutually_exclusive_group()
    eval_source.add_argument(
        "--eval-prompt-data",
        nargs="+",
        help="Evaluation dataset name/path pairs, e.g. fold0_val /path/val.jsonl.",
    )
    eval_source.add_argument(
        "--eval-config",
        type=Path,
        help="Slime evaluation YAML/JSON config; overrides legacy name/path pairs.",
    )
    parser.add_argument(
        "--n-samples-per-eval-prompt",
        type=int,
        default=3,
        help="Number of independent terminal samples per SWE validation instance.",
    )
    parser.add_argument(
        "--usage-ledger",
        type=Path,
        help="Append-only request-token ledger. Defaults to SAVE_DIR/usage.jsonl.",
    )
    parser.add_argument("--num-rollout", type=int)
    parser.add_argument("--num-epoch", type=int)
    parser.add_argument(
        "--train-instance-budget",
        type=int,
        help=(
            "Stop normally after this many attempted source instances. "
            "This is independent of the number of optimizer updates."
        ),
    )
    parser.add_argument(
        "--require-train-instance-budget-exhaustion",
        action="store_true",
        help=(
            "For a final invocation, fail before final validation/checkpoint "
            "unless the source cursor exactly exhausts "
            "--train-instance-budget. Intermediate chunk stops are exempt."
        ),
    )
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
    parser.add_argument("--context-parallel-size", type=int, default=2)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--decoder-last-pipeline-num-layers", type=int,
                        help="Pass to megatron when PP>1 with an unbalanced layer split "
                             "(e.g. 64 layers across PP=2 with last stage holding 30).")
    parser.add_argument(
        "--rollout-num-gpus-per-engine",
        type=int,
        default=1,
        help="SGLang tensor parallelism per rollout engine.",
    )
    parser.add_argument("--max-tokens-per-gpu", type=int, default=32768)
    parser.add_argument("--log-probs-chunk-size", type=int, default=256)
    parser.add_argument("--rollout-max-context-len", type=int, default=65536)
    parser.add_argument("--rollout-max-response-len", type=int, default=20480)
    parser.add_argument(
        "--sglang-context-length",
        type=int,
        default=128000,
        help=(
            "Optional SGLang served-context override. Keep this aligned with "
            "--rollout-max-context-len and the agent model context cap."
        ),
    )
    parser.add_argument(
        "--sglang-mem-fraction-static",
        type=float,
        default=0.85,
        help=(
            "Optional SGLang static-memory fraction override. When omitted, "
            "the sourced GRPO config value is preserved."
        ),
    )
    parser.add_argument(
        "--sglang-disable-custom-all-reduce",
        action="store_true",
        help=(
            "Disable SGLang's custom all-reduce fast path. This is a rollout "
            "runtime compatibility override and does not change actor training."
        ),
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help=(
            "Checkpoint interval."
        ),
    )
    parser.add_argument(
        "--checkpoint-retain-latest",
        type=int,
        default=0,
        help=(
            "Retain only the latest N complete model/dataset checkpoint "
            "pairs under --save-dir. Default 0 leaves legacy runs "
            "non-destructive; formal runs opt in explicitly."
        ),
    )
    parser.add_argument("--ray-num-cpus", type=int, default=32)
    parser.add_argument("--student-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--wandb-dir", type=Path)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE") or ("online" if os.environ.get("WANDB_API_KEY") else "offline"))
    parser.add_argument("--wandb-key", default=os.environ.get("WANDB_API_KEY"))
    parser.add_argument("--wandb-team", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-project", default="swe-agent-grpo")
    parser.add_argument("--wandb-group")
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
    args = parser.parse_args(argv)
    eval_enabled = (
        args.eval_interval is not None
        or args.eval_instance_interval is not None
    )
    if args.defer_instance_validation and args.eval_instance_interval is None:
        parser.error(
            "--defer-instance-validation requires --eval-instance-interval"
        )
    if eval_enabled:
        if args.eval_interval is not None and args.eval_interval <= 0:
            parser.error("--eval-interval must be positive")
        if (
            args.eval_instance_interval is not None
            and args.eval_instance_interval <= 0
        ):
            parser.error("--eval-instance-interval must be positive")
        if args.eval_prompt_data is None and args.eval_config is None:
            parser.error(
                "validation scheduling requires --eval-prompt-data or "
                "--eval-config"
            )
        if args.n_samples_per_eval_prompt <= 0:
            parser.error(
                "SWE validation requires positive --n-samples-per-eval-prompt"
            )
        if args.eval_prompt_data is not None and len(args.eval_prompt_data) % 2:
            parser.error(
                "--eval-prompt-data requires dataset name/path pairs"
            )
    elif args.eval_prompt_data is not None or args.eval_config is not None:
        parser.error(
            "--eval-prompt-data/--eval-config requires --eval-interval or "
            "--eval-instance-interval"
        )
    if (
        args.sglang_mem_fraction_static is not None
        and not 0.0 < args.sglang_mem_fraction_static <= 1.0
    ):
        parser.error("--sglang-mem-fraction-static must be in (0, 1]")
    if (
        args.sglang_context_length is not None
        and args.sglang_context_length <= 0
    ):
        parser.error("--sglang-context-length must be positive")
    if (
        args.rollout_max_response_len is not None
        and args.rollout_max_response_len <= 0
    ):
        parser.error("--rollout-max-response-len must be positive")
    if args.save_interval is not None and args.save_interval <= 0:
        parser.error("--save-interval must be positive")
    if (
        args.checkpoint_retain_latest is not None
        and args.checkpoint_retain_latest < 0
    ):
        parser.error("--checkpoint-retain-latest must be non-negative")
    if (
        args.train_instance_budget is not None
        and args.train_instance_budget <= 0
    ):
        parser.error("--train-instance-budget must be positive")
    if (
        args.require_train_instance_budget_exhaustion
        and args.train_instance_budget is None
    ):
        parser.error(
            "--require-train-instance-budget-exhaustion requires "
            "--train-instance-budget"
        )
    original_load_dir = args.load_dir
    resume_rollout_id = _configure_auto_resume(args, parser)
    if args.resume:
        tracker = args.load_dir / "latest_checkpointed_iteration.txt"
        if not tracker.is_file():
            parser.error(
                f"--resume requires a Megatron checkpoint tracker: {tracker}"
            )
        try:
            resume_rollout_id = int(tracker.read_text().strip())
        except (OSError, ValueError):
            parser.error(
                f"--resume requires a numeric rollout id in {tracker}"
            )
        if resume_rollout_id < 0:
            parser.error(
                f"--resume requires a non-negative rollout id in {tracker}"
            )
        checkpoint_dir = args.load_dir / f"iter_{resume_rollout_id:07d}"
        if not checkpoint_dir.is_dir():
            parser.error(
                f"--resume checkpoint directory does not exist: {checkpoint_dir}"
            )
        dataset_state = (
            args.load_dir
            / "rollout"
            / f"global_dataset_state_dict_{resume_rollout_id}.pt"
        )
        if not dataset_state.is_file():
            parser.error(
                "--resume requires the dataset state paired with checkpoint "
                f"{resume_rollout_id}: {dataset_state}"
            )
        if (
            _requires_exact_collector_resume(args)
            and not _dataset_state_has_exact_collector_position(
                dataset_state,
                resume_rollout_id,
            )
        ):
            parser.error(
                "--resume checkpoint does not contain the exact collector "
                "buffer/pending/counter state required to restore the "
                f"training position: {dataset_state}"
            )

    total_gpus = args.actor_num_gpus + args.rollout_num_gpus
    actor_num_nodes = args.actor_num_nodes or args.num_nodes
    if args.actor_num_gpus % actor_num_nodes != 0:
        raise SystemExit(
            f"--actor-num-gpus ({args.actor_num_gpus}) must be divisible "
            f"by --actor-num-nodes ({actor_num_nodes})"
        )
    actor_gpus_per_node = args.actor_num_gpus // actor_num_nodes
    # Auto-resume changes the actor load directory to SAVE_DIR, but the
    # frozen reference policy must remain the original base checkpoint.
    ref_load_dir = args.ref_load_dir or original_load_dir
    wandb_dir = args.wandb_dir or (args.save_dir / "wandb")
    global_batch_size = shlex.quote(str(args.global_batch_size or args.rollout_batch_size))
    num_rollout_args = shlex.join(["--num-rollout", str(args.num_rollout)]) if args.num_rollout is not None else ""
    num_epoch_args = shlex.join(["--num-epoch", str(args.num_epoch)]) if args.num_epoch is not None else ""
    train_instance_budget_arg = (
        shlex.join(
            [
                "--train-instance-budget",
                str(args.train_instance_budget),
            ]
        )
        if args.train_instance_budget is not None
        else ""
    )
    require_budget_exhaustion_arg = (
        "--require-train-instance-budget-exhaustion"
        if args.require_train_instance_budget_exhaustion
        else ""
    )
    oversample_arg = (
        shlex.join(["--over-sampling-batch-size", str(args.over_sampling_batch_size)])
        if args.over_sampling_batch_size is not None else ""
    )
    dynamic_filter_arg = (
        shlex.join(["--dynamic-sampling-filter-path", args.dynamic_sampling_filter_path])
        if args.dynamic_sampling_filter_path else ""
    )
    eval_arg_parts: list[str] = []
    if eval_enabled:
        # Slime's generic argument validation still expects eval_interval
        # whenever evaluation is configured. In instance-scheduled mode a
        # value of 1 is only an enable flag; train_async suppresses the
        # update-based schedule completely.
        framework_eval_interval = (
            args.eval_interval
            if args.eval_interval is not None
            else 1
        )
        eval_arg_parts.extend(
            [
                "--eval-interval",
                str(framework_eval_interval),
                "--n-samples-per-eval-prompt",
                str(args.n_samples_per_eval_prompt),
            ]
        )
        if args.eval_instance_interval is not None:
            eval_arg_parts.extend(
                [
                    "--eval-instance-interval",
                    str(args.eval_instance_interval),
                ]
            )
        if args.defer_instance_validation:
            eval_arg_parts.append("--defer-instance-validation")
        if args.eval_config is not None:
            eval_arg_parts.extend(["--eval-config", str(args.eval_config)])
        else:
            eval_arg_parts.append("--eval-prompt-data")
            eval_arg_parts.extend(str(value) for value in args.eval_prompt_data or [])
    eval_args = shlex.join(eval_arg_parts) if eval_arg_parts else ""
    usage_ledger = args.usage_ledger or (args.save_dir / "usage.jsonl")
    runtime_arg_parts = [
        "--save-interval", str(args.save_interval),
        "--tensor-model-parallel-size", str(args.tensor_model_parallel_size),
        "--pipeline-model-parallel-size", str(args.pipeline_model_parallel_size),
        "--context-parallel-size", str(args.context_parallel_size),
        "--rollout-num-gpus-per-engine", str(args.rollout_num_gpus_per_engine),
        "--max-tokens-per-gpu", str(args.max_tokens_per_gpu),
        "--log-probs-chunk-size", str(args.log_probs_chunk_size),
        "--rollout-max-context-len", str(args.rollout_max_context_len),
        "--rollout-max-response-len", str(args.rollout_max_response_len),
        "--sglang-context-length", str(args.sglang_context_length),
        "--sglang-mem-fraction-static", str(args.sglang_mem_fraction_static),
    ]
    if args.decoder_last_pipeline_num_layers is not None:
        runtime_arg_parts.extend(
            [
                "--decoder-last-pipeline-num-layers",
                str(args.decoder_last_pipeline_num_layers),
            ]
        )
    if args.sglang_disable_custom_all_reduce:
        runtime_arg_parts.append("--sglang-disable-custom-all-reduce")
    runtime_args = shlex.join(runtime_arg_parts)
    fresh_start_args = "" if args.resume else '"${GRPO_FRESH_START_ARGS[@]}"'
    wandb_args = ""
    if args.wandb_mode != "disabled":
        # Default the wandb group/run-name to the SLURM job name when running
        # under SLURM (so each launched job shows up in WandB as its job name
        # like "grpo-naive-cp4vftis-56612" rather than every run colliding on
        # "policy-grpo"). Falls back to policy-grpo for non-SLURM launches.
        slurm_job_name = os.environ.get("SLURM_JOB_NAME") or ""
        slurm_job_id = os.environ.get("SLURM_JOB_ID") or ""
        if args.wandb_group:
            wandb_group = args.wandb_group
        elif slurm_job_name:
            wandb_group = (
                f"{slurm_job_name}-{slurm_job_id}" if slurm_job_id else slurm_job_name
            )
        else:
            wandb_group = "policy-grpo"
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

    if args.resume:
        checkpoint_seed_fixup = (
            "# Explicit resume: preserve the checkpoint's recorded rollout id "
            "and load optimizer/RNG state.\n"
            f"echo 'Resuming optimizer, RNG, and dataset state after rollout "
            f"{resume_rollout_id}.'"
        )
    else:
        checkpoint_seed_fixup = f"""
for CHECKPOINT_DIR in {shlex.quote(str(args.load_dir))} {shlex.quote(str(ref_load_dir))}; do
  TRACKER="${{CHECKPOINT_DIR}}/latest_checkpointed_iteration.txt"
  if [ -f "${{TRACKER}}" ] && [ "$(tr -d '[:space:]' < "${{TRACKER}}")" = "0" ] && [ -d "${{CHECKPOINT_DIR}}/iter_0000000" ]; then
    if [ ! -e "${{CHECKPOINT_DIR}}/iter_0000001" ]; then
      cp -al "${{CHECKPOINT_DIR}}/iter_0000000" "${{CHECKPOINT_DIR}}/iter_0000001" 2>/dev/null || cp -a "${{CHECKPOINT_DIR}}/iter_0000000" "${{CHECKPOINT_DIR}}/iter_0000001"
    fi
    printf '1\\n' > "${{TRACKER}}"
  fi
done
""".strip()

    rler_root = str(args.rler_root)
    training_config_path = os.environ.get(
        "TRAINING_CONFIG_PATH",
        f"{rler_root}/slime/train_agent/configs/qwen3.5-9B.sh",
    )
    # Preserve runtime dependencies installed into a Lustre-side site dir.
    extra_pp = os.environ.get("EXTRA_PYTHONPATH", "")
    pythonpath = f"{rler_root}/slime:{rler_root}:{rler_root}/agent:/root/Megatron-LM"
    if extra_pp:
        pythonpath = f"{extra_pp}:{pythonpath}"
    command = f"""
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="{pythonpath}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export SWE_AGENT_GRPO_MODEL_NAME={shlex.quote(args.student_model)}
export SWE_AGENT_PYTHON="${{SWE_AGENT_PYTHON:-{rler_root}/agent/.venv/bin/python}}"
export RLER_USAGE_LEDGER_PATH={shlex.quote(str(usage_ledger))}
export RLER_USAGE_RESUME={"1" if args.resume else "0"}
export RLER_CHECKPOINT_RETAIN_LATEST={args.checkpoint_retain_latest}
{ray_trap_line}
{ray_stop_line}
cd {rler_root}/slime
source {shlex.quote(training_config_path)}
GLOBAL_BATCH_SIZE={global_batch_size}
{checkpoint_seed_fixup}
RAY_TMPDIR="/tmp/ray-swe-policy-grpo-$$"
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
  {fresh_start_args} \\
  --rollout-function-path {shlex.quote(args.rollout_function_path)} \\
  {eval_args} \\
  {num_rollout_args} \\
  {num_epoch_args} \\
  {train_instance_budget_arg} \\
  {require_budget_exhaustion_arg} \\
  "${{GRPO_PARALLEL_ARGS[@]}}" \\
  "${{GRPO_RECOMPUTE_ARGS[@]}}" \\
  "${{GRPO_OPTIMIZER_ARGS[@]}}" \\
  "${{GRPO_SGLANG_ARGS[@]}}" \\
  "${{GRPO_MISC_ARGS[@]}}" \\
  {runtime_args} \\
  {dynamic_filter_arg} \\
  {wandb_args}
"""
    subprocess.run(["bash", "-lc", command], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
