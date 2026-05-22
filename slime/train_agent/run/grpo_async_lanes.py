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
        --lanes-m 8 --lanes-n 1 --lanes-k 20 --lanes-p 1 --lanes-max-rounds 5 ...
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_lanes_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Pull lane-specific args out before forwarding the rest to grpo.main."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--policy-ports", default="")
    p.add_argument("--rubric-ports", default="")
    p.add_argument("--api-host", default="http://127.0.0.1")
    p.add_argument("--lanes-instance-workers", type=int, default=8)
    p.add_argument("--lanes-max-pending", type=int, default=0)
    p.add_argument("--lanes-min-ready-groups", type=int, default=0)
    p.add_argument("--lanes-wait-timeout", type=int, default=10800)
    p.add_argument("--lanes-output-root", default="")
    p.add_argument("--lanes-m", type=int, default=8)
    p.add_argument("--lanes-n", type=int, default=1)
    p.add_argument("--lanes-k", type=int, default=20)
    p.add_argument("--lanes-p", type=int, default=1)
    p.add_argument("--lanes-max-rounds", type=int, default=5)
    p.add_argument("--lanes-step-limit", type=int, default=100)
    p.add_argument("--lanes-max-active-rubrics", type=int, default=6)
    p.add_argument("--lanes-completion-max-tokens", type=int, default=4096)
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
    p.add_argument("--lanes-rubric-model", default="")
    p.add_argument("--lanes-judge-model", default="")
    return p.parse_known_args(argv)


def _export_lanes_env(ns: argparse.Namespace) -> None:
    if ns.policy_ports:
        os.environ["SWE_AGENT_LANES_POLICY_PORTS"] = ns.policy_ports
    if ns.rubric_ports:
        os.environ["SWE_AGENT_LANES_RUBRIC_PORTS"] = ns.rubric_ports
    if ns.api_host:
        os.environ["SWE_AGENT_LANES_API_HOST"] = ns.api_host
    if ns.lanes_output_root:
        os.environ["SWE_AGENT_LANES_OUTPUT_ROOT"] = ns.lanes_output_root

    os.environ["SWE_AGENT_LANES_INSTANCE_WORKERS"] = str(ns.lanes_instance_workers)
    if ns.lanes_max_pending:
        os.environ["SWE_AGENT_LANES_MAX_PENDING"] = str(ns.lanes_max_pending)
    if ns.lanes_min_ready_groups:
        os.environ["SWE_AGENT_LANES_MIN_READY_GROUPS"] = str(ns.lanes_min_ready_groups)
    os.environ["SWE_AGENT_LANES_WAIT_TIMEOUT"] = str(ns.lanes_wait_timeout)

    os.environ["SWE_AGENT_LANES_M"] = str(ns.lanes_m)
    os.environ["SWE_AGENT_LANES_N"] = str(ns.lanes_n)
    os.environ["SWE_AGENT_LANES_K"] = str(ns.lanes_k)
    os.environ["SWE_AGENT_LANES_P"] = str(ns.lanes_p)
    os.environ["SWE_AGENT_LANES_MAX_ROUNDS"] = str(ns.lanes_max_rounds)
    os.environ["SWE_AGENT_LANES_STEP_LIMIT"] = str(ns.lanes_step_limit)
    os.environ["SWE_AGENT_LANES_MAX_ACTIVE_RUBRICS"] = str(ns.lanes_max_active_rubrics)
    os.environ["SWE_AGENT_LANES_COMPLETION_MAX_TOKENS"] = str(ns.lanes_completion_max_tokens)
    os.environ["SWE_AGENT_LANES_GT_EVAL_WORKERS"] = str(ns.lanes_gt_eval_workers)
    if ns.lanes_lane_b_pool_size:
        os.environ["SWE_AGENT_LANES_LANE_B_POOL_SIZE"] = str(ns.lanes_lane_b_pool_size)
    os.environ["SWE_AGENT_LANES_POLICY_ALPHA"] = str(ns.lanes_policy_alpha)
    os.environ["SWE_AGENT_LANES_POLICY_TEMPERATURE"] = str(ns.lanes_policy_temperature)
    os.environ["SWE_AGENT_LANES_POLICY_TOP_P"] = str(ns.lanes_policy_top_p)
    os.environ["SWE_AGENT_LANES_LANE_B_TEMPERATURE"] = str(ns.lanes_lane_b_temperature)
    os.environ["SWE_AGENT_LANES_LANE_B_TOP_P"] = str(ns.lanes_lane_b_top_p)
    os.environ["SWE_AGENT_LANES_FALLBACK_PATCH_PENALTY"] = str(ns.lanes_fallback_patch_penalty)
    if ns.lanes_rubric_model:
        os.environ["SWE_AGENT_LANES_RUBRIC_MODEL"] = ns.lanes_rubric_model
    if ns.lanes_judge_model:
        os.environ["SWE_AGENT_LANES_JUDGE_MODEL"] = ns.lanes_judge_model


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    lanes_args, forwarded = _parse_lanes_args(argv)
    _export_lanes_env(lanes_args)
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
