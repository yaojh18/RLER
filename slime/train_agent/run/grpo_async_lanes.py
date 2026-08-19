"""GRPO async entry point backed by lane-based trajectory search.

Mirror of grpo_async_pds.py — pulls lane-specific args out, sets matching
SWE_AGENT_LANES_* env vars, then forwards the rest to train_agent.run.grpo.main
with --rollout-function-path pointing at the v1 collect module.

Usage:

    python -m train_agent.run.grpo_async_lanes --target policy \\
        --prompt-data ... --hf-checkpoint ... --load-dir ... --save-dir ... \\
        --policy-ports 30000,30001,30002,30003,30004,30005 \\
        --lanes-instance-workers 8 \\
        --lanes-m 8 --lanes-rollout-mode rollout40 ...
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_lanes_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Pull lane-specific args out before forwarding the rest to grpo.main."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--policy-ports", default="")
    p.add_argument("--api-host", default="http://127.0.0.1")
    p.add_argument("--lanes-instance-workers", type=int, default=8)
    p.add_argument("--lanes-max-pending", type=int, default=0)
    p.add_argument("--lanes-wait-timeout", type=int, default=10800)
    p.add_argument("--lanes-output-root", default="")
    p.add_argument("--lanes-m", type=int, default=8)
    p.add_argument("--lanes-steps-per-round", type=int, default=20)
    p.add_argument(
        "--lanes-rollout-mode",
        choices=("custom", "rollout40", "depth2"),
        default="custom",
        help=(
            "rollout40 samples a depth1 trajectory to terminal but judges "
            "and trains only its first 40 assistant steps; depth2 selects "
            "the legacy Lane-A 20 + Lane-B 20-40 topology; custom preserves "
            "the explicit topology, step, and terminal-rollout arguments"
        ),
    )
    p.add_argument("--lanes-step-limit", type=int, default=120)
    p.add_argument("--lanes-completion-max-tokens", type=int, default=20480)
    p.add_argument(
        "--model-context-length",
        type=int,
        default=int(os.environ.get("MODEL_CONTEXT_LENGTH", "65536")),
    )
    p.add_argument("--lanes-judge-max-tokens", type=int, default=20480)
    p.add_argument("--lanes-judge-context-length", type=int, default=256000)
    p.add_argument("--lanes-judge-temperature", type=float, default=0.02)
    p.add_argument("--lanes-judge-top-p", type=float, default=1.0)
    p.add_argument("--lanes-gt-eval-workers", type=int, default=8)
    p.add_argument("--lanes-lane-b-pool-size", type=int, default=0)
    p.add_argument("--lanes-policy-temperature", type=float, default=1.0,
                   help="Lane A sampling temperature (the linear spine).")
    p.add_argument("--lanes-policy-top-p", type=float, default=0.95)
    p.add_argument("--lanes-lane-b-temperature", type=float, default=1.0,
                   help="Lane B (fork) sampling temperature — higher for "
                        "intra-group diversity. Default 1.0.")
    p.add_argument("--lanes-lane-b-top-p", type=float, default=0.95)
    p.add_argument("--lanes-fallback-patch-penalty", type=float, default=0.5)
    p.add_argument("--lanes-no-action-patch-penalty", type=float, default=-0.1)
    p.add_argument(
        "--lanes-reward-kind",
        choices=("hard", "soft", "joint", "f2p_only"),
        default="joint",
    )
    p.add_argument("--lanes-joint-alpha", type=float, default=1.0)
    p.add_argument("--lanes-all-pass-reward", type=float, default=1.0)
    p.add_argument(
        "--lanes-topology",
        choices=("depth1", "depth2"),
        default="depth1",
        help="depth1 exports the root 8-way group; depth2 also exports a 2x4 beam group.",
    )
    p.add_argument("--lanes-beam-parents", type=int, default=2)
    p.add_argument(
        "--lanes-terminal-rollout",
        action="store_true",
        help="Continue prefixes to terminal and run GT evaluation. Disabled for train.",
    )
    p.add_argument("--validation-instance-workers", type=int, default=8)
    p.add_argument(
        "--validation-process-workers",
        type=int,
        default=int(os.environ.get("VALIDATION_PROCESS_WORKERS", "8")),
    )
    p.add_argument("--validation-step-limit", type=int, default=120)
    p.add_argument("--validation-completion-max-tokens", type=int, default=20480)
    p.add_argument("--validation-gt-eval-timeout", type=int, default=1800)
    p.add_argument(
        "--validation-temperature",
        type=float,
        default=0.2,
        help="Low-temperature sampling used only by terminal validation.",
    )
    p.add_argument("--validation-top-p", type=float, default=0.95)
    default_lane_c_model = "openai/azure/openai/gpt-5.6-luna"
    p.add_argument(
        "--lanes-judge-model",
        default=os.environ.get(
            "SWE_AGENT_LANES_JUDGE_MODEL",
            default_lane_c_model,
        ),
    )
    p.add_argument(
        "--lanes-rubric-api-base",
        default="https://inference-api.nvidia.com/v1",
    )
    p.add_argument(
        "--lanes-direct-rubric-bank",
        default=os.environ.get("SWE_AGENT_LANES_DIRECT_RUBRIC_BANK", ""),
    )
    p.add_argument("--lanes-disable-variance-detector", action="store_true")
    p.add_argument(
        "--lanes-gt-on-submit",
        action="store_true",
        help=(
            "Use binary SWE-bench reward for formally submitted branches; "
            "direct rubric reward remains the default."
        ),
    )
    p.add_argument(
        "--lanes-collapse-reward-margin",
        type=float,
        default=0.0,
        help="Positive margin enables the conservative prefix collapse penalty.",
    )
    p.add_argument("--lanes-disable-rubric", action="store_true",
                   help="Skip Lane C (rubric+judge) entirely. Reward becomes "
                        "the centrally computed branch.gt_score. Saves the "
                        "rubric token spend per ForkGroup.")
    return p.parse_known_args(argv)


def _export_lanes_env(ns: argparse.Namespace) -> None:
    if ns.policy_ports:
        os.environ["SWE_AGENT_LANES_POLICY_PORTS"] = ns.policy_ports
    if ns.api_host:
        os.environ["SWE_AGENT_LANES_API_HOST"] = ns.api_host
    if ns.lanes_output_root:
        os.environ["SWE_AGENT_LANES_OUTPUT_ROOT"] = ns.lanes_output_root

    os.environ["SWE_AGENT_LANES_INSTANCE_WORKERS"] = str(ns.lanes_instance_workers)
    if ns.lanes_max_pending:
        os.environ["SWE_AGENT_LANES_MAX_PENDING"] = str(ns.lanes_max_pending)
    os.environ["SWE_AGENT_LANES_WAIT_TIMEOUT"] = str(ns.lanes_wait_timeout)

    topology = ns.lanes_topology
    steps_per_round = ns.lanes_steps_per_round
    terminal_rollout = ns.lanes_terminal_rollout
    if ns.lanes_rollout_mode == "rollout40":
        topology = "depth1"
        steps_per_round = 40
        terminal_rollout = True
    elif ns.lanes_rollout_mode == "depth2":
        topology = "depth2"
        steps_per_round = 20
        terminal_rollout = False

    os.environ["SWE_AGENT_LANES_M"] = str(ns.lanes_m)
    os.environ["SWE_AGENT_LANES_STEPS_PER_ROUND"] = str(steps_per_round)
    os.environ["SWE_AGENT_LANES_STEP_LIMIT"] = str(ns.lanes_step_limit)
    os.environ["SWE_AGENT_LANES_COMPLETION_MAX_TOKENS"] = str(ns.lanes_completion_max_tokens)
    os.environ["SWE_AGENT_MODEL_CONTEXT_LENGTH"] = str(ns.model_context_length)
    os.environ["RLER_POLICY_MODEL_CONTEXT_LENGTH"] = str(
        ns.model_context_length
    )
    os.environ.setdefault("RLER_HOSTED_MODEL_CONTEXT_LENGTH", "256000")
    os.environ.setdefault("RLER_HOSTED_MAX_COMPLETION_TOKENS", "20480")
    os.environ["SWE_AGENT_LANES_JUDGE_MAX_TOKENS"] = str(ns.lanes_judge_max_tokens)
    os.environ["SWE_AGENT_LANES_JUDGE_CONTEXT_LENGTH"] = str(
        ns.lanes_judge_context_length
    )
    os.environ["SWE_AGENT_LANES_JUDGE_TEMPERATURE"] = str(
        ns.lanes_judge_temperature
    )
    os.environ["SWE_AGENT_LANES_JUDGE_TOP_P"] = str(ns.lanes_judge_top_p)
    os.environ["SWE_AGENT_LANES_GT_EVAL_WORKERS"] = str(ns.lanes_gt_eval_workers)
    if ns.lanes_lane_b_pool_size:
        os.environ["SWE_AGENT_LANES_LANE_B_POOL_SIZE"] = str(ns.lanes_lane_b_pool_size)
    os.environ["SWE_AGENT_LANES_POLICY_TEMPERATURE"] = str(ns.lanes_policy_temperature)
    os.environ["SWE_AGENT_LANES_POLICY_TOP_P"] = str(ns.lanes_policy_top_p)
    os.environ["SWE_AGENT_LANES_LANE_B_TEMPERATURE"] = str(ns.lanes_lane_b_temperature)
    os.environ["SWE_AGENT_LANES_LANE_B_TOP_P"] = str(ns.lanes_lane_b_top_p)
    os.environ["SWE_AGENT_LANES_FALLBACK_PATCH_PENALTY"] = str(ns.lanes_fallback_patch_penalty)
    os.environ["SWE_AGENT_LANES_NO_ACTION_PATCH_PENALTY"] = str(ns.lanes_no_action_patch_penalty)
    os.environ["SWE_AGENT_LANES_REWARD_KIND"] = ns.lanes_reward_kind
    os.environ["SWE_AGENT_LANES_JOINT_ALPHA"] = str(ns.lanes_joint_alpha)
    os.environ["SWE_AGENT_LANES_ALL_PASS_REWARD"] = str(ns.lanes_all_pass_reward)
    os.environ["SWE_AGENT_LANES_TOPOLOGY"] = topology
    os.environ["SWE_AGENT_LANES_BEAM_PARENTS"] = str(ns.lanes_beam_parents)
    os.environ["SWE_AGENT_LANES_TERMINAL_ROLLOUT"] = (
        "1" if terminal_rollout else "0"
    )
    os.environ["SWE_AGENT_VALIDATION_INSTANCE_WORKERS"] = str(ns.validation_instance_workers)
    os.environ["SWE_AGENT_VALIDATION_PROCESS_WORKERS"] = str(
        ns.validation_process_workers
    )
    os.environ["SWE_AGENT_VALIDATION_STEP_LIMIT"] = str(ns.validation_step_limit)
    os.environ["SWE_AGENT_VALIDATION_COMPLETION_MAX_TOKENS"] = str(
        ns.validation_completion_max_tokens
    )
    os.environ["SWE_AGENT_VALIDATION_GT_EVAL_TIMEOUT"] = str(
        ns.validation_gt_eval_timeout
    )
    os.environ["SWE_AGENT_VALIDATION_TEMPERATURE"] = str(
        ns.validation_temperature
    )
    os.environ["SWE_AGENT_VALIDATION_TOP_P"] = str(ns.validation_top_p)
    if ns.lanes_judge_model:
        os.environ["SWE_AGENT_LANES_JUDGE_MODEL"] = ns.lanes_judge_model
    os.environ["SWE_AGENT_LANES_RUBRIC_API_BASE"] = ns.lanes_rubric_api_base
    if ns.lanes_direct_rubric_bank:
        os.environ["SWE_AGENT_LANES_DIRECT_RUBRIC_BANK"] = (
            ns.lanes_direct_rubric_bank
        )
    else:
        os.environ.pop("SWE_AGENT_LANES_DIRECT_RUBRIC_BANK", None)
    os.environ["SWE_AGENT_LANES_ENABLE_VARIANCE_DETECTOR"] = (
        "0" if ns.lanes_disable_variance_detector else "1"
    )
    if ns.lanes_collapse_reward_margin > 0.0:
        os.environ["SWE_AGENT_LANES_COLLAPSE_REWARD_MARGIN"] = str(
            ns.lanes_collapse_reward_margin
        )
    else:
        os.environ.pop("SWE_AGENT_LANES_COLLAPSE_REWARD_MARGIN", None)
    os.environ["SWE_AGENT_LANES_DISABLE_RUBRIC"] = (
        "1" if ns.lanes_disable_rubric else "0"
    )
    os.environ["SWE_AGENT_LANES_GT_ON_SUBMIT"] = (
        "1" if ns.lanes_gt_on_submit else "0"
    )


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    lanes_args, forwarded = _parse_lanes_args(argv)
    _export_lanes_env(lanes_args)
    if not any(
        arg == "--dynamic-sampling-filter-path"
        or arg.startswith("--dynamic-sampling-filter-path=")
        for arg in forwarded
    ):
        filter_name = (
            "check_reward_nonzero_std"
            if lanes_args.lanes_disable_rubric
            else "check_direct_judge_variance"
        )
        forwarded.extend(
            [
                "--dynamic-sampling-filter-path",
                "slime.rollout.filter_hub.dynamic_sampling_filters."
                + filter_name,
            ]
        )
    from train_agent.run.grpo import main as grpo_main
    return grpo_main(
        [
            "--rollout-function-path",
            "train_agent.collect_lanes_rollout_async.generate_rollout",
            *forwarded,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
