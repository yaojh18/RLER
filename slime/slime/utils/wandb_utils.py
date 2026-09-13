import logging
import os
from copy import deepcopy

import wandb

logger = logging.getLogger(__name__)

_WANDB_METRIC_ALLOWLIST_ENV = "SLIME_WANDB_METRIC_ALLOWLIST"

# Keep W&B config focused on knobs that change optimization, sampling, or the
# train/eval schedule. Static experiment identity, filesystem paths,
# credentials, and proxy settings are intentionally excluded.
_WANDB_CONFIG_ALLOWLIST = (
    # Duration, batching, and dataset traversal.
    "num_epoch",
    "num_rollout",
    "train_instance_budget",
    "eval_instance_interval",
    "train_iters",
    "rollout_batch_size",
    "over_sampling_batch_size",
    "global_batch_size",
    "micro_batch_size",
    "num_steps_per_rollout",
    "update_weights_interval",
    "use_dynamic_batch_size",
    "max_tokens_per_gpu",
    "n_samples_per_prompt",
    "n_samples_per_eval_prompt",
    "rollout_global_dataset",
    "rollout_shuffle",
    "seed",
    "rollout_seed",
    # Objective and policy update.
    "advantage_estimator",
    "rewards_normalization",
    "grpo_std_normalization",
    "loss_type",
    "calculate_per_token_loss",
    "eps_clip",
    "eps_clip_high",
    "eps_clip_c",
    "grpo_kl_beta",
    "grpo_entropy_term_weight",
    "entropy_coef",
    "use_kl_loss",
    "kl_coef",
    "kl_loss_coef",
    "kl_loss_type",
    "use_unbiased_kl",
    "use_tis",
    "tis_clip",
    "tis_clip_low",
    "rl_importance_sampling_truncation_coef",
    "use_rollout_logprobs",
    "dynamic_sampling_filter_path",
    # Optimizer and learning-rate schedule.
    "optimizer",
    "lr",
    "min_lr",
    "lr_decay_style",
    "lr_warmup_fraction",
    "lr_warmup_iters",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
    "clip_grad",
    # Policy and validation sampling.
    "rollout_temperature",
    "rollout_top_p",
    "rollout_top_k",
    "rollout_max_context_len",
    "rollout_max_response_len",
    "eval_temperature",
    "eval_top_p",
    "eval_top_k",
    "eval_max_context_len",
    "eval_max_response_len",
    # Validation/checkpoint cadence.
    "eval_interval",
    "save_interval",
)

# The SWE-agent wrappers intentionally consume their method-specific CLI
# arguments before forwarding into Slime.  These environment variables are
# therefore the authoritative values used by the collectors, and must be
# merged back into W&B config rather than reporting Slime's unrelated generic
# eval defaults.
_WANDB_DYNAMIC_ENV_CONFIG = {
    "SWE_AGENT_MODEL_CONTEXT_LENGTH": ("model_context_length", int),
    "SWE_AGENT_VALIDATION_INSTANCE_WORKERS": (
        "validation_instance_workers",
        int,
    ),
    "SWE_AGENT_VALIDATION_STEP_LIMIT": ("validation_step_limit", int),
    "SWE_AGENT_VALIDATION_COMPLETION_MAX_TOKENS": (
        "validation_completion_max_tokens",
        int,
    ),
    "SWE_AGENT_VALIDATION_GT_EVAL_TIMEOUT": (
        "validation_gt_eval_timeout",
        lambda value: min(int(value), 600),
    ),
    "SWE_AGENT_VALIDATION_TEMPERATURE": (
        "validation_temperature",
        float,
    ),
    "SWE_AGENT_VALIDATION_TOP_P": ("validation_top_p", float),
    "SWE_AGENT_NAIVE_M": ("naive_m", int),
    "SWE_AGENT_NAIVE_STEP_LIMIT": ("naive_step_limit", int),
    "SWE_AGENT_NAIVE_INSTANCE_WORKERS": (
        "naive_instance_workers",
        int,
    ),
    "SWE_AGENT_NAIVE_MAX_PENDING": ("naive_max_pending", int),
    "SWE_AGENT_NAIVE_COMPLETION_MAX_TOKENS": (
        "naive_completion_max_tokens",
        int,
    ),
    "SWE_AGENT_NAIVE_GT_EVAL_TIMEOUT": (
        "naive_gt_eval_timeout",
        lambda value: min(int(value), 600),
    ),
    "SWE_AGENT_NAIVE_POLICY_TEMPERATURE": (
        "naive_policy_temperature",
        float,
    ),
    "SWE_AGENT_NAIVE_POLICY_TOP_P": ("naive_policy_top_p", float),
    "SWE_AGENT_NAIVE_FALLBACK_PATCH_PENALTY": (
        "naive_fallback_patch_penalty",
        float,
    ),
    "SWE_AGENT_NAIVE_NO_ACTION_PATCH_PENALTY": (
        "naive_no_action_patch_penalty",
        float,
    ),
    "SWE_AGENT_NAIVE_REWARD_KIND": ("naive_reward_kind", str),
    "SWE_AGENT_NAIVE_JOINT_ALPHA": ("naive_joint_alpha", float),
    "SWE_AGENT_NAIVE_ALL_PASS_REWARD": (
        "naive_all_pass_reward",
        float,
    ),
    "SWE_AGENT_LANES_TOPOLOGY": ("lanes_topology", str),
    "SWE_AGENT_LANES_M": ("lanes_m", int),
    "SWE_AGENT_LANES_BEAM_PARENTS": ("lanes_beam_parents", int),
    "SWE_AGENT_LANES_STEPS_PER_ROUND": (
        "lanes_steps_per_round",
        int,
    ),
    "SWE_AGENT_LANES_STEP_LIMIT": ("lanes_step_limit", int),
    "SWE_AGENT_LANES_INSTANCE_WORKERS": (
        "lanes_instance_workers",
        int,
    ),
    "SWE_AGENT_LANES_MAX_PENDING": ("lanes_max_pending", int),
    "SWE_AGENT_LANES_COMPLETION_MAX_TOKENS": (
        "lanes_completion_max_tokens",
        int,
    ),
    "SWE_AGENT_LANES_JUDGE_MAX_TOKENS": (
        "lanes_judge_max_tokens",
        int,
    ),
    "SWE_AGENT_LANES_POLICY_TEMPERATURE": (
        "lanes_policy_temperature",
        float,
    ),
    "SWE_AGENT_LANES_POLICY_TOP_P": ("lanes_policy_top_p", float),
    "SWE_AGENT_LANES_LANE_B_TEMPERATURE": (
        "lanes_lane_b_temperature",
        float,
    ),
    "SWE_AGENT_LANES_LANE_B_TOP_P": (
        "lanes_lane_b_top_p",
        float,
    ),
    "SWE_AGENT_LANES_FALLBACK_PATCH_PENALTY": (
        "lanes_fallback_patch_penalty",
        float,
    ),
    "SWE_AGENT_LANES_NO_ACTION_PATCH_PENALTY": (
        "lanes_no_action_patch_penalty",
        float,
    ),
    "SWE_AGENT_LANES_REWARD_KIND": ("lanes_reward_kind", str),
    "SWE_AGENT_LANES_JOINT_ALPHA": ("lanes_joint_alpha", float),
    "SWE_AGENT_LANES_ALL_PASS_REWARD": (
        "lanes_all_pass_reward",
        float,
    ),
    "SWE_AGENT_LANES_TERMINAL_ROLLOUT": (
        "lanes_terminal_rollout",
        lambda value: value.strip().lower() in {"1", "true", "yes", "on"},
    ),
}


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Prepare wandb init parameters
    # add random 6 length string with characters
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }
    # A Slurm requeue restarts the Python driver.  Let the frozen launcher
    # provide one stable run id so optimizer, validation, heartbeat, and usage
    # curves continue in the same W&B run instead of silently fragmenting.
    external_run_id = str(os.environ.get("WANDB_RUN_ID") or "").strip()
    if external_run_id:
        init_kwargs["id"] = external_run_id
        init_kwargs["resume"] = "allow"

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id


def _compute_config_for_logging(args):
    output = _args_to_config_dict(args)

    if getattr(args, "use_critic", False):
        critic_args = _get_role_args_for_logging(args, role="critic")
        output.update(_prefix_config_keys(_args_to_config_dict(critic_args), "critic"))

    return output


def filter_metrics_for_logging(metrics, *, step_key: str):
    """Apply an optional exact-name allowlist to W&B history metrics.

    Training and local logs retain the complete metric dictionary.  The
    allowlist only limits W&B history/dashboard cardinality, and is opt-in so
    other Slime workloads keep their existing telemetry unchanged.
    """

    raw_allowlist = os.environ.get(_WANDB_METRIC_ALLOWLIST_ENV, "")
    if not raw_allowlist.strip():
        return metrics
    allowed = {
        name.strip()
        for name in raw_allowlist.split(",")
        if name.strip()
    }
    allowed.add(step_key)
    return {key: value for key, value in metrics.items() if key in allowed}


def _args_to_config_dict(args):
    values = vars(args)
    output = {
        key: deepcopy(values[key])
        for key in _WANDB_CONFIG_ALLOWLIST
        if key in values and values[key] is not None
    }
    for env_name, (config_name, converter) in (
        _WANDB_DYNAMIC_ENV_CONFIG.items()
    ):
        raw_value = os.environ.get(env_name)
        if raw_value is None or not raw_value.strip():
            continue
        try:
            output[config_name] = converter(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid dynamic W&B config {env_name}={raw_value!r}"
            ) from exc
    if any(
        os.environ.get(name)
        for name in (
            "SWE_AGENT_NAIVE_POLICY_TEMPERATURE",
            "SWE_AGENT_LANES_POLICY_TEMPERATURE",
        )
    ):
        # These generic SGLang defaults do not control SWE-agent requests;
        # the method-specific collector values above do.
        for key in ("rollout_temperature", "rollout_top_p", "rollout_top_k"):
            output.pop(key, None)
    if os.environ.get("SWE_AGENT_VALIDATION_TEMPERATURE"):
        for key in ("eval_temperature", "eval_top_p", "eval_top_k"):
            output.pop(key, None)
    return output


def _prefix_config_keys(config, prefix):
    return {f"{prefix}/{key}": value for key, value in config.items()}


def _get_role_args_for_logging(args, role):
    if getattr(args, "megatron_config_path", None) is None:
        return args

    from slime.utils.arguments import parse_megatron_role_args

    return parse_megatron_role_args(args, args.megatron_config_path, role=role)


def _compute_secondary_config_for_logging(args, role=None):
    config = _args_to_config_dict(args)
    if role == "critic":
        return _prefix_config_keys(config, "critic")
    return config


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, role=None):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": _compute_secondary_config_for_logging(args, role=role),
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
    wandb.define_metric("swe_agent/*", step_metric="rollout/step")
    # Long hosted-judge waits need a progress axis independent of optimizer
    # updates and completed rollout batches.
    wandb.define_metric("heartbeat/event_step")
    wandb.define_metric("heartbeat/*", step_metric="heartbeat/event_step")
    # Request-level model usage has its own monotonically increasing event
    # axis. It intentionally does not share rollout/step: filtered and invalid
    # groups still consume Qwen/GLM tokens even when they produce no rollout or
    # optimizer update.
    wandb.define_metric("usage/event_step")
    wandb.define_metric("usage/*", step_metric="usage/event_step")
    # W&B only accepts a wildcard as a suffix.  Register cumulative series
    # explicitly so each run summary retains the experiment-wide maximum
    # without relying on the invalid ``usage/*_cumulative`` middle glob.
    for metric_name in (
        "total_tokens_cumulative",
        "train_tokens_cumulative",
        "validation_tokens_cumulative",
        "qwen_tokens_cumulative",
        "glm_tokens_cumulative",
    ):
        wandb.define_metric(
            f"usage/{metric_name}",
            step_metric="usage/event_step",
            summary="max",
        )
