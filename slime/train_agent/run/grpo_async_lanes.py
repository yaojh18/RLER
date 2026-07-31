"""GRPO async entry point backed by v1 lane-based trajectory search.

Mirror of grpo_async_pds.py — pulls v1-specific args out, sets matching
SWE_AGENT_LANES_* env vars, then forwards the rest to train_agent.run.grpo.main
with --rollout-function-path pointing at the v1 collect module.

Usage:

    python -m train_agent.run.grpo_async_lanes --target policy \\
        --prompt-data ... --hf-checkpoint ... --load-dir ... --save-dir ... \\
        --policy-ports 30000,30001,30002,30003,30004,30005 \\
        --rubric-ports 30006,30007 \\
        --lanes-instance-workers 8 \\
        --lanes-m 8 --lanes-max-mid-cps 6 ...
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_lanes_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Pull lane-specific args out before forwarding the rest to grpo.main."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--policy-ports", default="")
    p.add_argument(
        "--rubric-ports",
        default="",
        help="Deprecated: Lane C is hosted GLM and cannot use local policy ports.",
    )
    p.add_argument("--api-host", default="http://127.0.0.1")
    p.add_argument("--lanes-instance-workers", type=int, default=8)
    p.add_argument("--lanes-max-pending", type=int, default=0)
    p.add_argument("--lanes-wait-timeout", type=int, default=10800)
    p.add_argument("--lanes-output-root", default="")
    p.add_argument("--lanes-m", type=int, default=8)
    p.add_argument("--lanes-max-mid-cps", type=int, default=6)
    p.add_argument("--lanes-steps-per-round", type=int, default=20)
    p.add_argument("--lanes-step-limit", type=int, default=120)
    p.add_argument("--lanes-completion-max-tokens", type=int, default=20480)
    p.add_argument("--model-context-length", type=int, default=128000)
    p.add_argument("--lanes-rubric-max-tokens", type=int, default=20480)
    p.add_argument("--lanes-judge-max-tokens", type=int, default=20480)
    p.add_argument("--lanes-seed", type=int, default=0)
    p.add_argument("--lanes-gt-eval-workers", type=int, default=8)
    p.add_argument("--lanes-lane-b-pool-size", type=int, default=0)
    p.add_argument("--lanes-policy-alpha", type=float, default=1.0)
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
    p.add_argument("--lanes-rubric-model", default="nvidia/zai-org/glm-5.2")
    p.add_argument("--lanes-judge-model", default="nvidia/zai-org/glm-5.2")
    p.add_argument(
        "--lanes-rubric-api-base",
        default="https://inference-api.nvidia.com/v1",
    )
    p.add_argument("--lanes-experience-bank", default="")
    p.add_argument("--lanes-disable-rubric", action="store_true",
                   help="Skip Lane C (rubric+judge) entirely. Reward becomes "
                        "the centrally computed branch.gt_score. Saves the "
                        "rubric token spend per ForkGroup.")
    return p.parse_known_args(argv)


def _export_lanes_env(ns: argparse.Namespace) -> None:
    if ns.policy_ports:
        os.environ["SWE_AGENT_LANES_POLICY_PORTS"] = ns.policy_ports
    if ns.rubric_ports:
        raise ValueError(
            "--rubric-ports is not supported: all Lane C calls must use "
            "hosted NVIDIA GLM-5.2"
        )
    if ns.api_host:
        os.environ["SWE_AGENT_LANES_API_HOST"] = ns.api_host
    if ns.lanes_output_root:
        os.environ["SWE_AGENT_LANES_OUTPUT_ROOT"] = ns.lanes_output_root

    os.environ["SWE_AGENT_LANES_INSTANCE_WORKERS"] = str(ns.lanes_instance_workers)
    if ns.lanes_max_pending:
        os.environ["SWE_AGENT_LANES_MAX_PENDING"] = str(ns.lanes_max_pending)
    os.environ["SWE_AGENT_LANES_WAIT_TIMEOUT"] = str(ns.lanes_wait_timeout)

    os.environ["SWE_AGENT_LANES_M"] = str(ns.lanes_m)
    os.environ["SWE_AGENT_LANES_MAX_MID_CPS"] = str(ns.lanes_max_mid_cps)
    os.environ["SWE_AGENT_LANES_STEPS_PER_ROUND"] = str(ns.lanes_steps_per_round)
    os.environ["SWE_AGENT_LANES_STEP_LIMIT"] = str(ns.lanes_step_limit)
    os.environ["SWE_AGENT_LANES_COMPLETION_MAX_TOKENS"] = str(ns.lanes_completion_max_tokens)
    os.environ["SWE_AGENT_MODEL_CONTEXT_LENGTH"] = str(ns.model_context_length)
    os.environ["SWE_AGENT_LANES_SGLANG_CTX"] = str(ns.model_context_length)
    os.environ["RLER_POLICY_MODEL_CONTEXT_LENGTH"] = str(
        ns.model_context_length
    )
    os.environ.setdefault("RLER_HOSTED_MODEL_CONTEXT_LENGTH", "128000")
    os.environ.setdefault("RLER_HOSTED_MAX_COMPLETION_TOKENS", "20480")
    os.environ["SWE_AGENT_LANES_RUBRIC_MAX_TOKENS"] = str(ns.lanes_rubric_max_tokens)
    os.environ["SWE_AGENT_LANES_JUDGE_MAX_TOKENS"] = str(ns.lanes_judge_max_tokens)
    os.environ["SWE_AGENT_LANES_GT_EVAL_WORKERS"] = str(ns.lanes_gt_eval_workers)
    if ns.lanes_lane_b_pool_size:
        os.environ["SWE_AGENT_LANES_LANE_B_POOL_SIZE"] = str(ns.lanes_lane_b_pool_size)
    if ns.lanes_seed:
        os.environ["SWE_AGENT_LANES_SEED"] = str(ns.lanes_seed)
    # lanes_policy_alpha removed (task 4): reward is now pure rubric in
    # lane_to_grpo_bundle._build_branch_sample. The flag is silently
    # ignored upstream when passed by older sbatches.
    os.environ["SWE_AGENT_LANES_POLICY_TEMPERATURE"] = str(ns.lanes_policy_temperature)
    os.environ["SWE_AGENT_LANES_POLICY_TOP_P"] = str(ns.lanes_policy_top_p)
    os.environ["SWE_AGENT_LANES_LANE_B_TEMPERATURE"] = str(ns.lanes_lane_b_temperature)
    os.environ["SWE_AGENT_LANES_LANE_B_TOP_P"] = str(ns.lanes_lane_b_top_p)
    os.environ["SWE_AGENT_LANES_FALLBACK_PATCH_PENALTY"] = str(ns.lanes_fallback_patch_penalty)
    os.environ["SWE_AGENT_LANES_NO_ACTION_PATCH_PENALTY"] = str(ns.lanes_no_action_patch_penalty)
    os.environ["SWE_AGENT_LANES_REWARD_KIND"] = ns.lanes_reward_kind
    os.environ["SWE_AGENT_LANES_JOINT_ALPHA"] = str(ns.lanes_joint_alpha)
    os.environ["SWE_AGENT_LANES_ALL_PASS_REWARD"] = str(ns.lanes_all_pass_reward)
    os.environ["SWE_AGENT_LANES_TOPOLOGY"] = ns.lanes_topology
    os.environ["SWE_AGENT_LANES_BEAM_PARENTS"] = str(ns.lanes_beam_parents)
    os.environ["SWE_AGENT_LANES_TERMINAL_ROLLOUT"] = (
        "1" if ns.lanes_terminal_rollout else "0"
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
    if ns.lanes_rubric_model:
        os.environ["SWE_AGENT_LANES_RUBRIC_MODEL"] = ns.lanes_rubric_model
    if ns.lanes_judge_model:
        os.environ["SWE_AGENT_LANES_JUDGE_MODEL"] = ns.lanes_judge_model
    os.environ["SWE_AGENT_LANES_RUBRIC_API_BASE"] = ns.lanes_rubric_api_base
    if ns.lanes_experience_bank:
        os.environ["SWE_AGENT_LANES_EXPERIENCE_BANK"] = ns.lanes_experience_bank
    if ns.lanes_disable_rubric:
        os.environ["SWE_AGENT_LANES_DISABLE_RUBRIC"] = "1"


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
            "train_agent.collect_lanes_rollout_async.generate_rollout",
            *forwarded,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
