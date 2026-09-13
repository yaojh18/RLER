"""DPPO entry point for the final M-rollout SWE-agent methods.

Parses the collector-specific arguments, exports the matching
``SWE_AGENT_NAIVE_*`` environment contract, then invokes the shared policy
training driver with the naive collector.

Usage:

    python -m train_agent.run.grpo_async_naive \\
        --prompt-data ... --hf-checkpoint ... --load-dir ... --save-dir ... \\
        --naive-instance-workers 8 \\
        --naive-m 8 --naive-step-limit 120 ...
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_naive_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Pull naive-specific args out before forwarding the rest to grpo.main."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--naive-instance-workers", type=int, default=16)
    p.add_argument("--naive-max-pending", type=int, default=0)
    p.add_argument("--naive-min-ready-groups", type=int, default=0)
    p.add_argument("--naive-wait-timeout", type=int, default=10800)
    p.add_argument("--naive-output-root", default="")
    p.add_argument("--naive-m", type=int, default=8)
    p.add_argument("--naive-step-limit", type=int, default=120)
    p.add_argument(
        "--naive-train-assistant-step-limit",
        type=int,
        default=0,
        help=(
            "Complete and evaluate the terminal trajectory but export only "
            "the first N assistant turns; zero keeps the full trajectory."
        ),
    )
    p.add_argument("--naive-completion-max-tokens", type=int, default=20480)
    p.add_argument(
        "--model-context-length",
        type=int,
        default=int(os.environ.get("MODEL_CONTEXT_LENGTH", "65536")),
    )
    p.add_argument("--naive-gt-eval-workers", type=int, default=8)
    p.add_argument("--naive-gt-eval-timeout", type=int, default=600)
    p.add_argument("--naive-rollout-pool-size", type=int, default=0)
    p.add_argument("--naive-rollout-max-attempts", type=int, default=8)
    p.add_argument("--naive-policy-temperature", type=float, default=1.0,
                   help="Sampling temperature for the M independent rollouts.")
    p.add_argument("--naive-policy-top-p", type=float, default=0.95)
    p.add_argument("--naive-fallback-patch-penalty", type=float, default=0.5)
    p.add_argument("--naive-no-action-patch-penalty", type=float, default=-0.1)
    p.add_argument(
        "--naive-reward-kind",
        choices=("hard", "soft", "joint", "f2p_only"),
        default="joint",
    )
    p.add_argument("--naive-joint-alpha", type=float, default=1.0)
    p.add_argument("--naive-all-pass-reward", type=float, default=1.0)
    p.add_argument(
        "--naive-direct-reward-mode",
        choices=("none", "direct_judge"),
        default="none",
    )
    p.add_argument("--naive-direct-rubric-bank", default="")
    p.add_argument(
        "--naive-direct-disable-variance-detector",
        action="store_true",
    )
    p.add_argument(
        "--naive-direct-collapse-reward-margin", type=float, default=0.5
    )
    p.add_argument("--naive-direct-judge-context-length", type=int, default=256000)
    p.add_argument("--naive-direct-judge-max-tokens", type=int, default=20480)
    p.add_argument("--naive-direct-judge-temperature", type=float, default=0.02)
    p.add_argument("--naive-direct-judge-top-p", type=float, default=1.0)
    p.add_argument(
        "--naive-direct-judge-model",
        default="openai/azure/openai/gpt-5.6-luna",
    )
    p.add_argument(
        "--naive-direct-judge-api-base",
        default="https://inference-api.nvidia.com/v1",
    )
    p.add_argument("--validation-instance-workers", type=int, default=50)
    p.add_argument(
        "--validation-process-workers",
        type=int,
        default=int(os.environ.get("VALIDATION_PROCESS_WORKERS", "8")),
    )
    p.add_argument("--validation-step-limit", type=int, default=120)
    p.add_argument("--validation-completion-max-tokens", type=int, default=10240)
    p.add_argument("--validation-context-length", type=int, default=128000)
    p.add_argument("--validation-gt-eval-timeout", type=int, default=600)
    p.add_argument(
        "--validation-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used only by terminal validation.",
    )
    p.add_argument("--validation-top-p", type=float, default=0.95)
    return p.parse_known_args(argv)


def _export_naive_env(ns: argparse.Namespace) -> None:
    if ns.naive_output_root:
        os.environ["SWE_AGENT_NAIVE_OUTPUT_ROOT"] = ns.naive_output_root

    os.environ["SWE_AGENT_NAIVE_INSTANCE_WORKERS"] = str(ns.naive_instance_workers)
    if ns.naive_max_pending:
        os.environ["SWE_AGENT_NAIVE_MAX_PENDING"] = str(ns.naive_max_pending)
    if ns.naive_min_ready_groups:
        os.environ["SWE_AGENT_NAIVE_MIN_READY_GROUPS"] = str(ns.naive_min_ready_groups)
    os.environ["SWE_AGENT_NAIVE_WAIT_TIMEOUT"] = str(ns.naive_wait_timeout)

    os.environ["SWE_AGENT_NAIVE_M"] = str(ns.naive_m)
    os.environ["SWE_AGENT_NAIVE_STEP_LIMIT"] = str(ns.naive_step_limit)
    os.environ["SWE_AGENT_NAIVE_TRAIN_ASSISTANT_STEP_LIMIT"] = str(
        ns.naive_train_assistant_step_limit
    )
    os.environ["SWE_AGENT_NAIVE_COMPLETION_MAX_TOKENS"] = str(ns.naive_completion_max_tokens)
    os.environ["SWE_AGENT_MODEL_CONTEXT_LENGTH"] = str(ns.model_context_length)
    os.environ["RLER_POLICY_MODEL_CONTEXT_LENGTH"] = str(
        ns.model_context_length
    )
    os.environ.setdefault("RLER_HOSTED_MODEL_CONTEXT_LENGTH", "256000")
    os.environ.setdefault("RLER_HOSTED_MAX_COMPLETION_TOKENS", "20480")
    os.environ["SWE_AGENT_NAIVE_GT_EVAL_WORKERS"] = str(ns.naive_gt_eval_workers)
    os.environ["SWE_AGENT_NAIVE_GT_EVAL_TIMEOUT"] = str(
        min(ns.naive_gt_eval_timeout, 600)
    )
    if ns.naive_rollout_pool_size:
        os.environ["SWE_AGENT_NAIVE_ROLLOUT_POOL_SIZE"] = str(ns.naive_rollout_pool_size)
    os.environ["SWE_AGENT_NAIVE_ROLLOUT_MAX_ATTEMPTS"] = str(
        ns.naive_rollout_max_attempts
    )
    os.environ["SWE_AGENT_NAIVE_POLICY_TEMPERATURE"] = str(ns.naive_policy_temperature)
    os.environ["SWE_AGENT_NAIVE_POLICY_TOP_P"] = str(ns.naive_policy_top_p)
    os.environ["SWE_AGENT_NAIVE_FALLBACK_PATCH_PENALTY"] = str(ns.naive_fallback_patch_penalty)
    os.environ["SWE_AGENT_NAIVE_NO_ACTION_PATCH_PENALTY"] = str(ns.naive_no_action_patch_penalty)
    os.environ["SWE_AGENT_NAIVE_REWARD_KIND"] = ns.naive_reward_kind
    os.environ["SWE_AGENT_NAIVE_JOINT_ALPHA"] = str(ns.naive_joint_alpha)
    os.environ["SWE_AGENT_NAIVE_ALL_PASS_REWARD"] = str(ns.naive_all_pass_reward)
    os.environ["SWE_AGENT_NAIVE_DIRECT_REWARD_MODE"] = ns.naive_direct_reward_mode
    if ns.naive_direct_rubric_bank:
        os.environ["SWE_AGENT_NAIVE_DIRECT_RUBRIC_BANK"] = (
            ns.naive_direct_rubric_bank
        )
    os.environ["SWE_AGENT_NAIVE_DIRECT_ENABLE_VARIANCE_DETECTOR"] = (
        "0" if ns.naive_direct_disable_variance_detector else "1"
    )
    os.environ["SWE_AGENT_NAIVE_DIRECT_COLLAPSE_REWARD_MARGIN"] = str(
        ns.naive_direct_collapse_reward_margin
    )
    os.environ["SWE_AGENT_NAIVE_DIRECT_JUDGE_CONTEXT_LENGTH"] = str(
        ns.naive_direct_judge_context_length
    )
    os.environ["SWE_AGENT_NAIVE_DIRECT_JUDGE_MAX_TOKENS"] = str(
        ns.naive_direct_judge_max_tokens
    )
    os.environ["SWE_AGENT_NAIVE_DIRECT_JUDGE_TEMPERATURE"] = str(
        ns.naive_direct_judge_temperature
    )
    os.environ["SWE_AGENT_NAIVE_DIRECT_JUDGE_TOP_P"] = str(
        ns.naive_direct_judge_top_p
    )
    os.environ["SWE_AGENT_NAIVE_DIRECT_JUDGE_MODEL"] = (
        ns.naive_direct_judge_model
    )
    os.environ["SWE_AGENT_NAIVE_DIRECT_JUDGE_API_BASE"] = (
        ns.naive_direct_judge_api_base
    )
    os.environ["SWE_AGENT_VALIDATION_INSTANCE_WORKERS"] = str(ns.validation_instance_workers)
    os.environ["SWE_AGENT_VALIDATION_PROCESS_WORKERS"] = str(
        ns.validation_process_workers
    )
    os.environ["SWE_AGENT_VALIDATION_STEP_LIMIT"] = str(ns.validation_step_limit)
    os.environ["SWE_AGENT_VALIDATION_COMPLETION_MAX_TOKENS"] = str(
        ns.validation_completion_max_tokens
    )
    os.environ["SWE_AGENT_VALIDATION_CONTEXT_LENGTH"] = str(
        ns.validation_context_length
    )
    os.environ["SWE_AGENT_VALIDATION_GT_EVAL_TIMEOUT"] = str(
        min(ns.validation_gt_eval_timeout, 600)
    )
    os.environ["SWE_AGENT_VALIDATION_TEMPERATURE"] = str(
        ns.validation_temperature
    )
    os.environ["SWE_AGENT_VALIDATION_TOP_P"] = str(ns.validation_top_p)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    naive_args, forwarded = _parse_naive_args(argv)
    _export_naive_env(naive_args)
    if not any(
        arg == "--dynamic-sampling-filter-path"
        or arg.startswith("--dynamic-sampling-filter-path=")
        for arg in forwarded
    ):
        forwarded.extend(
            [
                "--dynamic-sampling-filter-path",
                "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
            ]
        )
    from train_agent.run.grpo import main as grpo_main
    return grpo_main(
        [
            "--rollout-function-path",
            "train_agent.collect_naive_rollout_async.generate_rollout",
            *forwarded,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
