"""GRPO async entry point for the naive M-rollout baseline.

Sibling of grpo_async_lanes.py — pulls naive-specific args out, sets
matching SWE_AGENT_NAIVE_* env vars, then forwards the rest to
train_agent.run.grpo.main with --rollout-function-path pointing at the
naive collect module.

Usage:

    python -m train_agent.run.grpo_async_naive --target policy \\
        --prompt-data ... --hf-checkpoint ... --load-dir ... --save-dir ... \\
        --policy-ports 30000,30001,30002,30003,30004,30005 \\
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
    p.add_argument("--policy-ports", default="")
    p.add_argument("--api-host", default="http://127.0.0.1")
    p.add_argument("--naive-instance-workers", type=int, default=8)
    p.add_argument("--naive-max-pending", type=int, default=0)
    p.add_argument("--naive-min-ready-groups", type=int, default=0)
    p.add_argument("--naive-wait-timeout", type=int, default=10800)
    p.add_argument("--naive-output-root", default="")
    p.add_argument("--naive-m", type=int, default=8)
    p.add_argument("--naive-step-limit", type=int, default=120)
    p.add_argument("--naive-completion-max-tokens", type=int, default=4096)
    p.add_argument("--naive-seed", type=int, default=0)
    p.add_argument("--naive-gt-eval-workers", type=int, default=8)
    p.add_argument("--naive-rollout-pool-size", type=int, default=0)
    p.add_argument("--naive-policy-temperature", type=float, default=1.0,
                   help="Sampling temperature for the M independent rollouts.")
    p.add_argument("--naive-policy-top-p", type=float, default=0.95)
    p.add_argument("--naive-fallback-patch-penalty", type=float, default=0.5)
    p.add_argument("--naive-no-action-patch-penalty", type=float, default=0.0,
                   help="Multiplier applied when rollout emitted zero env actions. "
                        "0.0 = hard zero reward (default), 1.0 = no penalty.")
    p.add_argument("--naive-format-error-per-step-penalty", type=float, default=0.0,
                   help="Linear per-turn format-error reward decay coefficient k. "
                        "Reward *= max(0, 1 - n_format_errors * k). 0.0 = disabled "
                        "(default). k=0.02 zeros reward at 50 format-error turns.")
    p.add_argument("--naive-format-ok-gate-threshold", type=float, default=0.0,
                   help="Hard gate: zero reward when format-error-rate over assistant "
                        "turns exceeds this threshold. 0.0 = disabled (default). "
                        "0.5 = kill reward on any trajectory >50%% malformed turns.")
    p.add_argument("--naive-kill-stale-docker-threshold", type=int, default=0,
                   help="Stale-docker kill threshold. When a running rollout's "
                        "intra-trajectory weight-version spread "
                        "(latest_turn_wv - eldest_observed_wv) exceeds this value, "
                        "the docker env is aborted and the rollout is routed to a "
                        "dummy sample (excluded from baseline + zero loss_mask). "
                        "0 = disabled (default), preserves the original single-call "
                        "agent loop.")
    return p.parse_known_args(argv)


def _export_naive_env(ns: argparse.Namespace) -> None:
    if ns.policy_ports:
        os.environ["SWE_AGENT_NAIVE_POLICY_PORTS"] = ns.policy_ports
    if ns.api_host:
        os.environ["SWE_AGENT_NAIVE_API_HOST"] = ns.api_host
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
    os.environ["SWE_AGENT_NAIVE_COMPLETION_MAX_TOKENS"] = str(ns.naive_completion_max_tokens)
    os.environ["SWE_AGENT_NAIVE_GT_EVAL_WORKERS"] = str(ns.naive_gt_eval_workers)
    if ns.naive_rollout_pool_size:
        os.environ["SWE_AGENT_NAIVE_ROLLOUT_POOL_SIZE"] = str(ns.naive_rollout_pool_size)
    if ns.naive_seed:
        os.environ["SWE_AGENT_NAIVE_SEED"] = str(ns.naive_seed)
    os.environ["SWE_AGENT_NAIVE_POLICY_TEMPERATURE"] = str(ns.naive_policy_temperature)
    os.environ["SWE_AGENT_NAIVE_POLICY_TOP_P"] = str(ns.naive_policy_top_p)
    os.environ["SWE_AGENT_NAIVE_FALLBACK_PATCH_PENALTY"] = str(ns.naive_fallback_patch_penalty)
    os.environ["SWE_AGENT_NAIVE_NO_ACTION_PATCH_PENALTY"] = str(ns.naive_no_action_patch_penalty)
    os.environ["SWE_AGENT_NAIVE_FORMAT_ERROR_PER_STEP_PENALTY"] = str(
        ns.naive_format_error_per_step_penalty
    )
    os.environ["SWE_AGENT_NAIVE_FORMAT_OK_GATE_THRESHOLD"] = str(
        ns.naive_format_ok_gate_threshold
    )
    os.environ["SWE_AGENT_NAIVE_KILL_STALE_DOCKER_THRESHOLD"] = str(
        ns.naive_kill_stale_docker_threshold
    )


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    naive_args, forwarded = _parse_naive_args(argv)
    _export_naive_env(naive_args)
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
