#!/usr/bin/env python3
"""Async rollout function backed by NaiveSearchRunner (naive baseline).

Drop-in for collect_lanes_rollout_async.generate_rollout — same signature,
same return type. Each instance produces ONE ExportGroup of M=8 samples,
one per independent linear rollout. No Lane A spine, no Lane B forks from
intermediate states, no Lane C rubric/judge. GT reward only.

ENV CONTRACT (set by slime/train_agent/run/grpo_async_naive.py from the
SLURM script's tunables):
  SWE_AGENT_NAIVE_M                  int    rollouts per instance (default 8)
  SWE_AGENT_NAIVE_STEP_LIMIT         int    hard cap per rollout (default 120)
  SWE_AGENT_NAIVE_INSTANCE_WORKERS   int    concurrent instance subprocesses (default 8)
  SWE_AGENT_NAIVE_GT_EVAL_WORKERS    int    per-instance GT eval threads (default 8)
  SWE_AGENT_NAIVE_COMPLETION_MAX_TOKENS int per-call max_new_tokens (default 16384)
  SWE_AGENT_NAIVE_POLICY_TEMPERATURE float  sampling temp (default 1.0)
  SWE_AGENT_NAIVE_POLICY_TOP_P       float  default 0.95
  SWE_AGENT_NAIVE_FALLBACK_PATCH_PENALTY float default 0.5
  SWE_AGENT_NAIVE_NO_ACTION_PATCH_PENALTY float default -0.1
  SWE_AGENT_NAIVE_OUTPUT_ROOT        str    where to write per-instance run dirs
  SWE_AGENT_NAIVE_ROLLOUT_POOL_SIZE  int    per-instance ThreadPoolExecutor cap (default = M)
  SWE_AGENT_NAIVE_SEED               int    seed for sampling (optional)
  SWE_AGENT_NAIVE_REWARD_KIND        str    hard, soft, delta, joint, or f2p_only;
                                            default delta
Per-engine endpoints come from args.sglang_model_engines (populated at engine
init in slime/ray/rollout.py). The legacy SWE_AGENT_NAIVE_POLICY_PORTS env
var is no longer consulted.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import call_dynamic_filter
from slime.utils.misc import load_function
from slime.utils.types import Sample

from train_agent.collect_grpo_rollout import build_rollout_samples
from train_agent.serving.sglang_chat_service import start_slime_policy_route_warmup


logger = logging.getLogger("train_agent.collect_naive_rollout_async")


# ---------------------------------------------------------------------------
# Subprocess entry: run one instance end-to-end and return its bundle
# ---------------------------------------------------------------------------


def _naive_bundle_task(task: dict[str, Any]) -> dict[str, Any]:
    """Worker-process entry. Builds backend + naive runner for one instance,
    runs M independent linear rollouts, GT-scores each, returns the
    GRPOExportBundle (one ExportGroup of M samples).
    """
    try:
        os.environ["SEARCH_SWE_SLIME_API_BASE"] = task["policy_base_url"]
        os.environ["SEARCH_SWE_SLIME_API_KEY"] = task["api_key"]
        from swe_agent.backend import SWEAgentRolloutBackend
        from swe_agent.naive_search import NaiveSearchConfig, NaiveSearchRunner
        from swe_agent.naive_to_grpo_bundle import naive_record_to_bundle
        from swe_agent.run.benchmarks.swebench import (
            build_swebench_config,
            get_swebench_docker_image_name,
            get_swebench_harness_namespace,
            get_swebench_singularity_image_name,
            load_swebench_instances_by_id,
            select_container_environment_class,
        )
        from swe_agent.run.run_swe_agent import SWE_AGENT_TEXTBASED_CONFIG
        from swe_agent.utils.serialize import recursive_merge

        instance_id = task["instance_id"]
        subset = task["subset"]
        split = task["split"]
        model_name = task["model_name"]
        policy_base_url = task["policy_base_url"]
        policy_base_urls = task.get("policy_base_urls") or [policy_base_url]
        api_key = task["api_key"]
        output_root = Path(task["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)

        instances = load_swebench_instances_by_id(subset, split, [instance_id])
        if not instances:
            return {
                "index": task["index"],
                "instance_id": instance_id,
                "bundle": None,
                "error": f"instance {instance_id} not found in {subset}/{split}",
            }
        instance = instances[0]

        api_base = policy_base_url.rstrip("/")
        if not api_base.endswith("/v1"):
            api_base = api_base + "/v1"
        base_config = build_swebench_config(
            config_spec=[str(SWE_AGENT_TEXTBASED_CONFIG)],
            model=model_name,
            model_class="route_textbased",
        )
        image_name = get_swebench_docker_image_name(instance)
        environment_class = select_container_environment_class(
            base_config.get("environment", {}).get("environment_class", "docker")
        )
        environment_overrides = {
            "image": image_name if environment_class == "docker" else get_swebench_singularity_image_name(instance),
            "environment_class": environment_class,
        }
        if instance.get("expected_output_json"):
            environment_overrides["cwd"] = "/testbed"
            environment_overrides["dataset_name"] = "r2egym"
        overrides = {
            "agent": {"step_limit": task["step_limit"]},
            "model": {
                "model_kwargs": {
                    "api_base": api_base,
                    "api_key": api_key,
                    "max_tokens": task.get("completion_max_tokens", 16384),
                    "extra_body": (
                        {"chat_template_kwargs": {"enable_thinking": True}}
                        if "qwen" in model_name.lower() else {}
                    ),
                }
            },
            "environment": environment_overrides,
        }
        config = recursive_merge(base_config, overrides)
        backend = SWEAgentRolloutBackend(
            model=config.get("model", {}),
            environment=config.get("environment", {}),
            agent=config.get("agent", {}),
            default_agent_type="default",
            default_environment_type=config.get("environment", {})
                                          .get("environment_class", "docker"),
        )

        reward_kind = task.get("reward_kind", "delta")
        if reward_kind not in {"hard", "soft", "delta", "joint", "f2p_only"}:
            raise ValueError(f"unsupported naive reward_kind={reward_kind!r}")
        cfg = NaiveSearchConfig(
            m=task["m"],
            step_limit=task["step_limit"],
            seed=task.get("seed"),
            gt_eval_workers=task.get("gt_eval_workers", 8),
            rollout_pool_size=task.get("rollout_pool_size") or task["m"],
            policy_temperature=task.get("policy_temperature", 1.0),
            policy_top_p=task.get("policy_top_p", 0.95),
            fallback_patch_penalty=task.get("fallback_patch_penalty", 0.5),
            no_action_patch_penalty=task.get("no_action_patch_penalty", -0.1),
            reward_kind=reward_kind,
            joint_alpha=task.get("joint_alpha", 1.0),
            all_pass_reward=task.get("all_pass_reward", 2.0),
        )
        run_dir = (
            output_root / instance_id
            / time.strftime("%Y%m%d-%H%M%S")
            / f"task-{task['index']:06d}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        runner = NaiveSearchRunner(
            instance=instance,
            backend=backend,
            run_dir=run_dir,
            policy_model_name=model_name,
            config=cfg,
            harness_namespace=get_swebench_harness_namespace(instance),
            policy_base_url=policy_base_url,
            policy_base_urls=policy_base_urls,
            api_key=api_key,
        )
        record = asyncio.run(runner.run())
        bundle = naive_record_to_bundle(record)
        result = {
            "index": task["index"],
            "instance_id": instance_id,
            "bundle": bundle,
            "error": "",
        }
    except Exception as exc:
        result = {
            "index": task.get("index", -1),
            "instance_id": task.get("instance_id", "?"),
            "bundle": None,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }
    finally:
        # Per-task subprocess cleanup. Same OOM mitigation as the lanes
        # collector: ProcessPoolExecutor(max_tasks_per_child=5) recycles
        # the worker, but we still drop the heaviest locals + malloc_trim
        # to bound RAM growth between tasks within a worker's lifetime.
        try: del record
        except UnboundLocalError: pass
        try: del runner
        except UnboundLocalError: pass
        try: del backend
        except UnboundLocalError: pass
        try: del instance
        except UnboundLocalError: pass
        try: del instances
        except UnboundLocalError: pass
        try: del config
        except UnboundLocalError: pass
        try: del cfg
        except UnboundLocalError: pass
        try: del base_config
        except UnboundLocalError: pass
        try: del overrides
        except UnboundLocalError: pass
        import gc as _gc
        _gc.collect()
        try:
            import ctypes as _ctypes
            _ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Module-level rollout state (mirrors lanes pattern)
# ---------------------------------------------------------------------------


@dataclass
class _BufferedGroup:
    samples: list[Sample]


_BUFFER: list[_BufferedGroup] = []
_TASK_INDEX = 0
_WARMUP_DONE = False
_FAILED_INSTANCES = 0
_TRUNCATED_OVERSIZED = 0
_FILTER_DROPPED_GROUPS = 0
_FILTER_DROP_REASONS: dict[str, int] = {}
# Count attempted on-policy groups and groups rejected because at least one
# sibling trajectory was invalid. Dynamic-filter drops are deliberately
# excluded because they are model-based filtering, not infra failures.
_INFRA_DROPPED_GROUPS = 0
_INFRA_DROP_REASONS: dict[str, int] = {}
_TOTAL_GROUPS_ATTEMPTED = 0
# Cumulative totals at the close of the previous perf log line. Subtract from
# current totals to expose per-step deltas in wandb alongside the cumulative
# counters (no manual diff() needed in the dashboard).
_PREV_FILTER_DROPPED_GROUPS = 0
_PREV_INFRA_DROPPED_GROUPS = 0
_PREV_TOTAL_GROUPS_ATTEMPTED = 0
# Per-endpoint in-flight trial count, used by _pick_least_loaded_endpoints
# to route each new instance's M trials across the engines that are currently
# least loaded. Bumped when an instance is dispatched, decremented when its
# bundle returns (success or error). Key is (host, port) so
# we route across multiple physical nodes, not just multiple ports on one
# host. With the default sglang_router cache_aware policy this is the only
# way to get per-trial fanout: trials hitting the router collapse onto one
# engine because their starting prompts share a long prefix.
_ENDPOINT_INFLIGHT: dict[tuple[str, int], int] = {}

_NODE_WORKERS: list[Any] = []
_NODE_WORKER_IPS: list[str] = []
_PENDING: dict[Any, dict[str, Any]] = {}
_DISPATCH_COUNTER = 0


@ray.remote(num_cpus=2)
class _NaiveNodeWorker:
    """Per-physical-node executor for naive bundle tasks. Same pattern
    as lanes _LanesNodeWorker — atomic ProcessPoolExecutor with
    max_tasks_per_child=100 to bound cross-task RAM leaks while keeping
    subprocess recycle rare enough to avoid the SemLock spawn race
    (cpython #84559: a recycled child unpickling SemLock can race the
    dying parent's resource_tracker unlink → FileNotFoundError →
    BrokenProcessPool that never self-heals). submit_task wraps
    BrokenProcessPool with a rebuild-and-retry-once safety net."""

    def __init__(self, name: str = "", max_workers: int = 8, max_tasks_per_child: int = 100):
        self.name = name
        self.max_workers = max_workers
        self.max_tasks_per_child = max_tasks_per_child
        self._build_executor()
        import socket
        self.ip = socket.gethostbyname(socket.gethostname())
        logger.info(
            f"[NaiveNodeWorker {self.name}] up on ip={self.ip} "
            f"max_workers={max_workers} max_tasks_per_child={max_tasks_per_child}"
        )

    def _build_executor(self) -> None:
        from concurrent.futures import ProcessPoolExecutor
        try:
            self._executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                max_tasks_per_child=self.max_tasks_per_child,
            )
        except TypeError:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)

    def get_ip(self) -> str:
        return self.ip

    def submit_task(self, task: dict[str, Any]) -> dict[str, Any]:
        from concurrent.futures.process import BrokenProcessPool
        try:
            future = self._executor.submit(_naive_bundle_task, task)
            return future.result()
        except BrokenProcessPool:
            logger.warning(
                "[NaiveNodeWorker %s] BrokenProcessPool — rebuilding "
                "executor and retrying instance=%s",
                self.name, task.get("instance_id", "?"),
            )
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._build_executor()
            future = self._executor.submit(_naive_bundle_task, task)
            return future.result()

    def memory_usage(self) -> dict:
        try:
            import psutil
            actor_proc = psutil.Process()
            actor_rss = actor_proc.memory_info().rss / (1024**3)
            sub_rss = []
            for sp in getattr(self._executor, "_processes", {}).values():
                try:
                    sub_rss.append(psutil.Process(sp.pid).memory_info().rss / (1024**3))
                except Exception:
                    pass
            sub_rss.sort(reverse=True)
            return {
                "actor_rss_GB": round(actor_rss, 2),
                "subprocess_rss_GB_total": round(sum(sub_rss), 2),
                "n_subprocesses": len(sub_rss),
                "top_subproc_rss_GB": [round(x, 2) for x in sub_rss[:5]],
            }
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def shutdown(self) -> None:
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


def _spawn_node_workers(per_node_concurrency: int) -> None:
    global _NODE_WORKERS, _NODE_WORKER_IPS
    if _NODE_WORKERS:
        return
    # SWE_AGENT_NAIVE_SKIP_NODE_IPS: csv of IPs/hostnames to exclude from
    # the NodeWorker pool. Used when the actor lives on a dedicated node
    # and we don't want rollout subprocess pressure on it (round-1 OOM at
    # step 23 was on the actor node co-hosting 6 instance workers).
    skip = {
        s.strip()
        for s in os.environ.get("SWE_AGENT_NAIVE_SKIP_NODE_IPS", "").split(",")
        if s.strip()
    }
    nodes = []
    for n in ray.nodes():
        if not n.get("Alive"):
            continue
        if float(n.get("Resources", {}).get("GPU", 0)) <= 0:
            continue
        node_ip = n.get("NodeManagerAddress") or ""
        node_name = n.get("NodeName") or ""
        if node_ip in skip or node_name in skip:
            logger.info(
                f"[naive-async] SKIP node {node_name or node_ip} "
                f"(matched SWE_AGENT_NAIVE_SKIP_NODE_IPS)"
            )
            continue
        nodes.append(n)
    if not nodes:
        raise RuntimeError("No GPU-bearing Ray nodes found for Naive worker spawn")
    actor_concurrency = per_node_concurrency + 4
    logger.info(
        f"[naive-async] spawning {len(nodes)} node workers "
        f"(skipped={len(skip)}), "
        f"per-node concurrency={per_node_concurrency} "
        f"actor_concurrency={actor_concurrency}"
    )
    for n in nodes:
        node_id = n["NodeID"]
        node_name = n.get("NodeName") or n.get("NodeManagerAddress") or node_id[:8]
        worker = _NaiveNodeWorker.options(
            num_cpus=2,
            max_concurrency=actor_concurrency,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False,
            ),
            name=f"naive-node-worker-{node_name}",
        ).remote(name=node_name, max_workers=per_node_concurrency)
        ip = ray.get(worker.get_ip.remote())
        _NODE_WORKERS.append(worker)
        _NODE_WORKER_IPS.append(ip)
        logger.info(f"[naive-async]   spawned worker on node={node_name} ip={ip}")


def _shutdown_node_workers() -> None:
    global _NODE_WORKERS, _NODE_WORKER_IPS
    for w in _NODE_WORKERS:
        try:
            ray.kill(w, no_restart=True)
        except Exception:
            pass
    _NODE_WORKERS = []
    _NODE_WORKER_IPS = []


atexit.register(_shutdown_node_workers)


def _naive_values_from_env() -> dict[str, Any]:
    """Read naive-specific knobs from env."""
    def _int(name, default=None):
        v = os.environ.get(name, "")
        return int(v) if v else default

    def _float(name, default):
        v = os.environ.get(name, "")
        return float(v) if v else default

    def _bool(name, default):
        v = os.environ.get(name, "")
        if not v:
            return default
        return v.strip().lower() in ("1", "true", "yes", "on")

    return {
        "m": _int("SWE_AGENT_NAIVE_M", 8),
        "step_limit": _int("SWE_AGENT_NAIVE_STEP_LIMIT", 120),
        "seed": _int("SWE_AGENT_NAIVE_SEED"),
        "gt_eval_workers": _int("SWE_AGENT_NAIVE_GT_EVAL_WORKERS", 8),
        "rollout_pool_size": _int("SWE_AGENT_NAIVE_ROLLOUT_POOL_SIZE"),
        "completion_max_tokens": _int("SWE_AGENT_NAIVE_COMPLETION_MAX_TOKENS", 16384),
        "policy_temperature": _float("SWE_AGENT_NAIVE_POLICY_TEMPERATURE", 1.0),
        "policy_top_p": _float("SWE_AGENT_NAIVE_POLICY_TOP_P", 0.95),
        "fallback_patch_penalty": _float("SWE_AGENT_NAIVE_FALLBACK_PATCH_PENALTY", 0.5),
        "no_action_patch_penalty": _float("SWE_AGENT_NAIVE_NO_ACTION_PATCH_PENALTY", -0.1),
        "reward_kind": (os.environ.get("SWE_AGENT_NAIVE_REWARD_KIND") or "delta").lower(),
        "joint_alpha": _float("SWE_AGENT_NAIVE_JOINT_ALPHA", 1.0),
        "all_pass_reward": _float("SWE_AGENT_NAIVE_ALL_PASS_REWARD", 2.0),
    }


def _hash_to_index(s: str, n: int) -> int:
    import hashlib
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h, 16) % n


def _pick_least_loaded_endpoints(
    m: int, endpoints: list[tuple[str, int]]
) -> tuple[list[str], list[tuple[str, int]]]:
    """Pick M URLs (with repetition if m > len(endpoints)) by current in-flight
    load: each call selects the (host,port) with the smallest
    _ENDPOINT_INFLIGHT count, pre-incrementing that count so the next pick
    within the same call sees the updated load. Returns (urls, endpoints_picked)
    so the caller can later decrement the same endpoints when the bundle
    returns.

    Ties are broken by ascending (host, port), giving deterministic spread
    when all engines start empty: the first M instances dispatched will
    spread across the first M engines, second batch across next M, etc.
    Each trial keeps its endpoint for the full multi-turn rollout so the
    engine's prefix cache stays warm — exactly the per-trial sticky pattern
    sglang_router's cache_aware policy can't deliver because it sees all
    trials of an instance as cache-identical at trial-start.
    """
    picks: list[tuple[str, int]] = []
    for _ in range(m):
        ep = min(endpoints, key=lambda hp: (_ENDPOINT_INFLIGHT.get(hp, 0), hp[0], hp[1]))
        _ENDPOINT_INFLIGHT[ep] = _ENDPOINT_INFLIGHT.get(ep, 0) + 1
        picks.append(ep)
    urls = [f"http://{h}:{p}" for (h, p) in picks]
    return urls, picks


def _release_endpoints(endpoints_picked: list[tuple[str, int]]) -> None:
    for ep in endpoints_picked:
        cur = _ENDPOINT_INFLIGHT.get(ep, 0)
        _ENDPOINT_INFLIGHT[ep] = max(0, cur - 1)


def _submit_until_full(
    *,
    args,
    data_buffer,
    rollout_id: int,
    output_root: Path,
    model_name: str,
    max_pending: int,
    target_groups: int,
) -> int:
    global _TASK_INDEX, _DISPATCH_COUNTER
    submitted = 0
    naive_values = _naive_values_from_env()

    # Per-engine endpoints discovered at engine init in slime/ray/rollout.py.
    # Each entry is (host, port) for one sglang engine; the policy dispatcher
    # picks the least-loaded one per trial, then pins all turns of that trial
    # to the chosen endpoint (sticky per trial -> KV cache reuse within the
    # trial, spread across trials of an instance).
    engines_map = getattr(args, "sglang_model_engines", None) or {}
    endpoints: list[tuple[str, int]] = list(engines_map.get(model_name, []))
    if not endpoints and len(engines_map) == 1:
        # Upstream slime keys this dict by server name ("default"), not by
        # served model name. When there is exactly one server we can use
        # its endpoints unambiguously regardless of the key.
        endpoints = list(next(iter(engines_map.values())))
    if not endpoints:
        # Fallback: route everything through the single router. Loses
        # per-trial spread but keeps the collector functional even if the
        # ray.get_url discovery failed for this server.
        router_ip, router_port = (getattr(args, "sglang_model_routers", None) or {}).get(
            model_name, (args.sglang_router_ip, args.sglang_router_port),
        )
        endpoints = [(router_ip, router_port)]
        logger.warning(
            "[naive-async] no per-engine endpoints for model=%s — "
            "falling back to single router %s:%s (per-trial routing dormant)",
            model_name, router_ip, router_port,
        )

    api_key = os.environ.get("SEARCH_SWE_SLIME_API_KEY", "EMPTY")

    while (
        len(_PENDING) < max_pending
        and len(_BUFFER) + len(_PENDING) < target_groups
    ):
        prompt_groups = data_buffer.get_samples(1)
        if not prompt_groups:
            break
        (prompt_group,) = prompt_groups
        # slime expands n_samples_per_prompt by duplicating the prompt; the
        # naive runner itself creates the M=8 sibling group, so we just take
        # the first duplicate. n_samples_per_prompt MUST equal SWE_AGENT_NAIVE_M.
        metadata = prompt_group[0].metadata
        instance_id = metadata["instance_id"]
        # Per-trial routing: pick M URLs by current per-engine load. The M
        # trials of this instance get spread across distinct engines (cycling
        # back to least-loaded if M > num_engines), so all engines stay busy
        # even when max_pending < num_engines.
        m_trials = int(naive_values.get("m", 8))
        policy_base_urls, endpoints_picked = _pick_least_loaded_endpoints(
            m_trials, endpoints,
        )

        task = {
            "index": _TASK_INDEX,
            "rollout_id": rollout_id,
            "instance_id": instance_id,
            "subset": metadata["subset"],
            "split": metadata["split"],
            "output_root": str(output_root / f"rollout_{rollout_id:04d}"),
            "model_name": model_name,
            # Legacy single-URL key (kept for any consumer that hasn't been
            # updated yet; worker prefers policy_base_urls when present).
            "policy_base_url": policy_base_urls[0],
            "policy_base_urls": policy_base_urls,
            "_endpoints_picked": endpoints_picked,
            "api_key": api_key,
            **naive_values,
        }
        _TASK_INDEX += 1
        worker = _NODE_WORKERS[_DISPATCH_COUNTER % len(_NODE_WORKERS)]
        _DISPATCH_COUNTER += 1
        ref = worker.submit_task.remote(task)
        _PENDING[ref] = task
        submitted += 1
    return submitted


def _harvest_ready(
    *,
    args,
    target: str,
    block: bool,
) -> int:
    """Reap Ray Futures whose subprocess fully returned. For each completed
    bundle, expand its policy_groups into Samples and append to _BUFFER."""
    global _FAILED_INSTANCES, _TRUNCATED_OVERSIZED
    global _FILTER_DROPPED_GROUPS, _FILTER_DROP_REASONS
    global _INFRA_DROPPED_GROUPS, _INFRA_DROP_REASONS, _TOTAL_GROUPS_ATTEMPTED
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if getattr(args, "dynamic_sampling_filter_path", None) else None
    )
    harvested = 0
    max_sample_tokens: int | None = None
    try:
        mt = int(getattr(args, "max_tokens_per_gpu", 0) or 0)
        cp = int(getattr(args, "context_parallel_size", 1) or 1)
        if mt > 0:
            max_sample_tokens = mt * cp
    except Exception:
        max_sample_tokens = None

    if not _PENDING:
        return harvested
    pending_refs = list(_PENDING.keys())
    if block and harvested == 0:
        ready, _ = ray.wait(pending_refs, num_returns=1, timeout=None)
    else:
        ready, _ = ray.wait(pending_refs, num_returns=len(pending_refs), timeout=0)
    for ref in ready:
        task = _PENDING.pop(ref)
        # The bundle is no longer in-flight on any engine — release the
        # M endpoint slots we reserved at dispatch, regardless of outcome.
        # Tuples come back from Ray as lists; coerce to tuple for dict key.
        _release_endpoints([
            tuple(ep) for ep in task.get("_endpoints_picked", [])
        ])
        try:
            result = ray.get(ref)
        except Exception as exc:
            _FAILED_INSTANCES += 1
            logger.warning(
                "[naive-async] instance %s failed: %s",
                task.get("instance_id"), str(exc)[:200],
            )
            del ref
            continue
        if result.get("error"):
            _FAILED_INSTANCES += 1
            logger.warning(
                "[naive-async] instance %s failed: %s",
                result.get("instance_id"), result["error"][:200],
            )
            del result, ref
            continue
        bundle = result.get("bundle")
        if bundle is None:
            del result, ref
            continue
        # Naive baseline is policy-only. target=rubric is invalid here.
        groups = (
            bundle.policy_groups if target == "policy" else bundle.rubric_groups
        )
        # Naive produces exactly one attempted ExportGroup per instance. An
        # invalid sibling leaves policy_groups empty but records the reason.
        if target == "policy":
            _TOTAL_GROUPS_ATTEMPTED += 1
            drop_reason = (bundle.metadata or {}).get("group_dropped_reason")
            if drop_reason:
                _INFRA_DROPPED_GROUPS += 1
                key = str(drop_reason).split(" ", 1)[0]
                _INFRA_DROP_REASONS[key] = _INFRA_DROP_REASONS.get(key, 0) + 1
        for group in groups:
            samples, truncated_here = build_rollout_samples(
                groups=[group],
                include_turn_rewards=False,
                max_sample_tokens=max_sample_tokens,
            )
            _TRUNCATED_OVERSIZED += truncated_here
            for sample in samples:
                sample.metadata = {
                    **(sample.metadata or {}),
                    "instance_id": task["instance_id"],
                }
            if dynamic_filter is not None and samples:
                filter_out = call_dynamic_filter(dynamic_filter, args, samples)
                if not filter_out.keep:
                    _FILTER_DROPPED_GROUPS += 1
                    reason = filter_out.reason or "unspecified"
                    _FILTER_DROP_REASONS[reason] = _FILTER_DROP_REASONS.get(reason, 0) + 1
                    continue
            _BUFFER.append(
                _BufferedGroup(samples=samples)
            )
            harvested += 1
        del result, bundle, groups, ref
    return harvested


def _pop_groups(min_groups: int) -> list[_BufferedGroup]:
    # Shuffle for the same reason as the lanes collector: easy instances
    # finish first and would dominate early batches otherwise.
    if os.environ.get("SWE_AGENT_NAIVE_NO_SHUFFLE", "0") != "1":
        random.shuffle(_BUFFER)
    groups = _BUFFER[:min_groups]
    del _BUFFER[:min_groups]
    return groups


# ---------------------------------------------------------------------------
# Slime rollout entry
# ---------------------------------------------------------------------------


def generate_rollout(args, rollout_id: int, data_buffer, evaluation: bool = False):
    """Slime rollout entry — called once per training rollout cycle."""
    global _WARMUP_DONE
    if evaluation:
        return RolloutFnEvalOutput(data={}, metrics={})

    started = time.perf_counter()
    target = os.environ.get("SWE_AGENT_GRPO_TARGET", "policy")
    output_root = Path(
        os.environ.get(
            "SWE_AGENT_NAIVE_OUTPUT_ROOT",
            "/workspace/rler/agent/outputs/naive_outputs/train_async_naive",
        )
    )
    model_name = os.environ.get("SWE_AGENT_GRPO_MODEL_NAME") or "Qwen/Qwen3.5-9B"

    if not _WARMUP_DONE:
        router_ip, router_port = (getattr(args, "sglang_model_routers", None) or {}).get(
            model_name, (args.sglang_router_ip, args.sglang_router_port),
        )
        try:
            start_slime_policy_route_warmup(
                base_url=f"http://{router_ip}:{router_port}",
                api_key=os.environ.get("SEARCH_SWE_SLIME_API_KEY", "EMPTY"),
                model_name=model_name,
                requests=max(
                    1,
                    int(args.rollout_num_gpus or 1)
                    // int(args.rollout_num_gpus_per_engine or 1),
                ),
            )
        except Exception as exc:
            logger.warning("[naive-async] warmup failed (continuing): %s", exc)
        _WARMUP_DONE = True

    instance_workers = max(1, int(os.environ.get("SWE_AGENT_NAIVE_INSTANCE_WORKERS", "8")))
    max_pending = max(1, int(
        os.environ.get("SWE_AGENT_NAIVE_MAX_PENDING", str(instance_workers))
    ))
    min_ready_groups = max(1, int(
        os.environ.get("SWE_AGENT_NAIVE_MIN_READY_GROUPS", str(args.rollout_batch_size))
    ))
    timeout = float(os.environ.get("SWE_AGENT_NAIVE_WAIT_TIMEOUT", "10800"))

    if _PENDING or _BUFFER:
        raise RuntimeError(
            "Naive collector retained work across official generate() boundaries: "
            f"pending={len(_PENDING)} buffer={len(_BUFFER)}"
        )

    if not _NODE_WORKERS:
        skip = {
            s.strip()
            for s in os.environ.get("SWE_AGENT_NAIVE_SKIP_NODE_IPS", "").split(",")
            if s.strip()
        }
        n_nodes = max(1, len([
            n for n in ray.nodes()
            if n.get("Alive")
            and float(n.get("Resources", {}).get("GPU", 0)) > 0
            and (n.get("NodeManagerAddress") or "") not in skip
            and (n.get("NodeName") or "") not in skip
        ]))
        per_node = max(1, (instance_workers + n_nodes - 1) // n_nodes)
        _spawn_node_workers(per_node_concurrency=per_node)

    logger.info(
        "[naive-async] generate_rollout START rollout_id=%d target=%s "
        "min_ready_groups=%d max_pending=%d buffer_at_entry=%d pending_at_entry=%d",
        rollout_id, target, min_ready_groups, max_pending,
        len(_BUFFER), len(_PENDING),
    )

    submitted = _submit_until_full(
        args=args, data_buffer=data_buffer, rollout_id=rollout_id,
        output_root=output_root, model_name=model_name, max_pending=max_pending,
        target_groups=min_ready_groups,
    )
    logger.info(
        "[naive-async] post-init submitted=%d buffer=%d pending=%d",
        submitted, len(_BUFFER), len(_PENDING),
    )

    wait_started = time.perf_counter()
    last_pulse = wait_started
    while len(_BUFFER) < min_ready_groups:
        if time.perf_counter() - wait_started > timeout:
            raise RuntimeError(
                f"Naive async timed out waiting for samples: "
                f"buffer={len(_BUFFER)} pending={len(_PENDING)}"
            )
        _harvest_ready(
            args=args, target=target, block=bool(_PENDING),
        )
        submitted += _submit_until_full(
            args=args, data_buffer=data_buffer, rollout_id=rollout_id,
            output_root=output_root, model_name=model_name, max_pending=max_pending,
            target_groups=min_ready_groups,
        )
        if time.perf_counter() - last_pulse > 30:
            last_pulse = time.perf_counter()
            mems = []
            for w_idx, worker in enumerate(_NODE_WORKERS):
                try:
                    m = ray.get(worker.memory_usage.remote(), timeout=10)
                    ip = _NODE_WORKER_IPS[w_idx] if w_idx < len(_NODE_WORKER_IPS) else "?"
                    mems.append(
                        f"{ip}:actor={m.get('actor_rss_GB', '?')}GB,"
                        f"subprocs={m.get('subprocess_rss_GB_total', '?')}GB"
                        f"(n={m.get('n_subprocesses', '?')},"
                        f"top={m.get('top_subproc_rss_GB', '?')})"
                    )
                except Exception as exc:
                    mems.append(f"err={type(exc).__name__}")
            logger.info(
                "[naive-async] WAITING rollout_id=%d elapsed=%.0fs "
                "buffer=%d/%d pending=%d submitted_total=%d",
                rollout_id, time.perf_counter() - wait_started,
                len(_BUFFER), min_ready_groups, len(_PENDING), submitted,
            )
            logger.info("[naive-mem] %s", " | ".join(mems))
        if not _PENDING and len(_BUFFER) < min_ready_groups:
            raise RuntimeError(
                f"Naive async produced no {target} samples and has no pending instance."
            )

    selected_groups = _pop_groups(min_ready_groups)
    if _PENDING or _BUFFER:
        raise RuntimeError(
            "Naive collector finished a batch with hidden work: "
            f"pending={len(_PENDING)} buffer={len(_BUFFER)}"
        )
    samples = [sample for group in selected_groups for sample in group.samples]
    for group_index, group in enumerate(selected_groups):
        for sample in group.samples:
            sample.group_index = group_index
    for index, sample in enumerate(samples):
        sample.index = index

    # --- per-batch metrics aggregated from sample.metadata + sample fields ---
    def _safe_mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0
    cont_steps = [
        int(s.metadata.get("n_continuation_steps", 0) or 0)
        for s in samples if s.metadata
    ]
    full_steps = [
        int(s.metadata.get("n_full_trace_steps", 0) or 0)
        for s in samples if s.metadata
    ]
    real_samples = samples
    gts = [
        float(s.metadata.get("raw_gt_score") or 0.0)
        for s in real_samples
        if s.metadata and s.metadata.get("raw_gt_score") is not None
    ]
    n_passed = sum(1 for g in gts if g >= 0.5)
    submit_count = sum(
        1 for s in real_samples
        if s.metadata and s.metadata.get("terminated_early")
    )
    # Per-instance pass: instance succeeds if ANY of its rollouts pass.
    by_inst_max: dict[str, float] = {}
    for s in real_samples:
        if not s.metadata:
            continue
        iid = s.metadata.get("instance_id") or ""
        gt = s.metadata.get("raw_gt_score")
        if gt is None:
            continue
        by_inst_max[iid] = max(by_inst_max.get(iid, 0.0), float(gt))
    distinct_instances = len(by_inst_max)
    instance_pass = (
        sum(1 for v in by_inst_max.values() if v >= 0.5) / distinct_instances
        if distinct_instances else 0.0
    )
    try:
        from slime.utils.types import Sample as _SlimeSample
        trunc_count = sum(
            1 for s in samples
            if getattr(s, "status", None) == _SlimeSample.Status.TRUNCATED
        )
    except Exception:
        trunc_count = 0
    # Raw entropy proxy from rollout-time logprobs.
    rollout_lps: list[float] = []
    for s in samples:
        lps = getattr(s, "rollout_log_probs", None)
        if lps:
            for x in lps:
                if isinstance(x, (int, float)):
                    rollout_lps.append(float(x))

    # --- Patch + eval-based ratios over the post-filter training batch ---
    n_real = len(real_samples)
    formal_submit = sum(
        1 for s in real_samples
        if s.metadata and s.metadata.get("terminated_early")
    )
    informal_submit = sum(
        1 for s in real_samples
        if s.metadata
        and not s.metadata.get("terminated_early")
        and int(s.metadata.get("terminal_patch_len") or 0) > 0
    )
    zero_patch = sum(
        1 for s in real_samples
        if s.metadata and int(s.metadata.get("terminal_patch_len") or 0) == 0
    )
    turns_for_avg: list[int] = []
    fe_rates: list[float] = []
    fe_counts: list[int] = []
    n_fe_gated = 0
    n_fe_killed = 0
    for s in real_samples:
        if s.metadata and s.metadata.get("n_full_trace_steps") is not None:
            turns_for_avg.append(int(s.metadata.get("n_full_trace_steps") or 0))
        if s.metadata and s.metadata.get("n_assistant_turns") is not None:
            rate = float(s.metadata.get("format_error_rate") or 0.0)
            count = int(s.metadata.get("n_format_errors") or 0)
            fe_rates.append(rate)
            fe_counts.append(count)
            # A trajectory is "gated" if more than half its asst turns were
            # format errors — same signal the runtime gate uses (default 0.5).
            if rate > 0.5:
                n_fe_gated += 1
        if s.metadata and s.metadata.get("format_error_killed"):
            n_fe_killed += 1
    global _PREV_FILTER_DROPPED_GROUPS
    global _PREV_INFRA_DROPPED_GROUPS, _PREV_TOTAL_GROUPS_ATTEMPTED
    filter_dropped_step = _FILTER_DROPPED_GROUPS - _PREV_FILTER_DROPPED_GROUPS
    infra_dropped_step = _INFRA_DROPPED_GROUPS - _PREV_INFRA_DROPPED_GROUPS
    attempted_step = _TOTAL_GROUPS_ATTEMPTED - _PREV_TOTAL_GROUPS_ATTEMPTED
    _PREV_FILTER_DROPPED_GROUPS = _FILTER_DROPPED_GROUPS
    _PREV_INFRA_DROPPED_GROUPS = _INFRA_DROPPED_GROUPS
    _PREV_TOTAL_GROUPS_ATTEMPTED = _TOTAL_GROUPS_ATTEMPTED
    n_groups_kept = len(selected_groups)
    filter_drop_rate_step = (
        filter_dropped_step / (filter_dropped_step + n_groups_kept)
        if (filter_dropped_step + n_groups_kept) > 0 else 0.0
    )
    infra_drop_rate_step = (
        infra_dropped_step / attempted_step if attempted_step > 0 else 0.0
    )
    # L3 circuit-breaker. Raise if too many groups were dropped for infra
    # reasons (docker pull misses, cache failures, etc.). Default 50% with
    # a minimum of 8 attempted groups in the step, both overridable via env
    # so we can tune without redeploying. Setting limit >= 1.0 disables.
    try:
        _infra_limit = float(os.getenv("NAIVE_INFRA_DROP_RATE_LIMIT", "0.5"))
    except ValueError:
        _infra_limit = 0.5
    try:
        _infra_min = int(os.getenv("NAIVE_INFRA_DROP_MIN_ATTEMPTED", "8"))
    except ValueError:
        _infra_min = 8
    if (
        _infra_limit < 1.0
        and attempted_step >= _infra_min
        and infra_drop_rate_step > _infra_limit
    ):
        top_reasons = sorted(
            _INFRA_DROP_REASONS.items(), key=lambda kv: -kv[1]
        )[:3]
        raise RuntimeError(
            f"[naive-async] L3 circuit-breaker tripped: infra_drop_rate="
            f"{infra_drop_rate_step:.1%} ({infra_dropped_step}/{attempted_step}) "
            f"> limit={_infra_limit:.1%} (min_attempted={_infra_min}). "
            f"Top reasons (cumulative): {top_reasons}. "
            f"Likely a docker-cache miss / registry rate-limit storm — "
            f"check the missing-image list and the docker_lustre_wrapper logs "
            f"before resubmitting. Set NAIVE_INFRA_DROP_RATE_LIMIT=1.0 to "
            f"disable (NOT recommended)."
        )

    return RolloutFnTrainOutput(
        samples=samples,
        metrics={
            "swe_agent/target": target,
            "swe_agent/samples": len(samples),
            "swe_agent/groups": len(selected_groups),
            "swe_agent/pending_instances": len(_PENDING),
            "swe_agent/buffer_groups": len(_BUFFER),
            "swe_agent/submitted_instances": submitted,
            "swe_agent/failed_instances_total": _FAILED_INSTANCES,
            "swe_agent/truncated_oversized_samples_total": _TRUNCATED_OVERSIZED,
            "swe_agent/filter_dropped_groups_total": _FILTER_DROPPED_GROUPS,
            "swe_agent/infra_dropped_groups_total": _INFRA_DROPPED_GROUPS,
            "swe_agent/total_groups_attempted_total": _TOTAL_GROUPS_ATTEMPTED,
            # --- per-step drop deltas (easier to chart than diff of totals) ---
            "swe_agent/dynamic_filter_dropped_groups": filter_dropped_step,
            "swe_agent/dynamic_filter_drop_rate": filter_drop_rate_step,
            "swe_agent/infra_dropped_groups": infra_dropped_step,
            "swe_agent/groups_attempted": attempted_step,
            "swe_agent/infra_drop_rate": infra_drop_rate_step,
            "swe_agent/wait_seconds": time.perf_counter() - wait_started,
            "swe_agent/seconds": time.perf_counter() - started,
            "swe_agent/source": "naive",
            # --- per-batch rollout shape ---
            "swe_agent/sample_cont_steps_mean": _safe_mean(cont_steps),
            "swe_agent/sample_full_steps_mean": _safe_mean(full_steps),
            "swe_agent/sample_gt_mean": _safe_mean(gts),
            "swe_agent/sample_pass_at_0.5": (n_passed / len(gts)) if gts else 0.0,
            "swe_agent/submit_rate": (submit_count / len(real_samples)) if real_samples else 0.0,
            "swe_agent/truncation_rate": (trunc_count / len(samples)) if samples else 0.0,
            "swe_agent/distinct_instances_in_batch": distinct_instances,
            "swe_agent/instance_pass_at_0.5": instance_pass,
            "swe_agent/rollout_logprob_mean": _safe_mean(rollout_lps),
            "swe_agent/rollout_logprob_token_count": len(rollout_lps),
            # --- patch + eval ratios over post-filter batch ---
            "swe_agent/ratio_formal_submit": (formal_submit / n_real) if n_real else 0.0,
            "swe_agent/ratio_informal_submit": (informal_submit / n_real) if n_real else 0.0,
            "swe_agent/ratio_zero_patch": (zero_patch / n_real) if n_real else 0.0,
            "swe_agent/avg_turns": _safe_mean([float(x) for x in turns_for_avg]),
            # --- format-error health (job 58062 mode-collapse signal) ---
            "swe_agent/format_error_rate_mean": _safe_mean(fe_rates),
            "swe_agent/format_error_count_mean": _safe_mean(
                [float(x) for x in fe_counts]
            ),
            "swe_agent/ratio_fe_gated_trajectories": (
                (n_fe_gated / n_real) if n_real else 0.0
            ),
            # Format-error circuit breaker fired on this rollout — runner
            # aborted after N consecutive trailing format errors.
            "swe_agent/n_format_error_killed": n_fe_killed,
            "swe_agent/ratio_format_error_killed": (
                (n_fe_killed / n_real) if n_real else 0.0
            ),
        },
    )
