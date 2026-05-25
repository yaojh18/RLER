#!/usr/bin/env python3
"""Async rollout function backed by TrajectorySearchParallelRunner (v1).

Drop-in for collect_grpo_rollout_async.generate_rollout — same signature,
same return type. Uses the v1 lane-based search to produce per-instance
GRPOExportBundle.policy_groups (one ExportGroup per fork-group = per
mid_cp). Atomic mode only — no streaming queue.

Selects target='policy' (rubric_groups not generated in v1 first pass).

ENV CONTRACT (set by slime/train_agent/run/grpo_async_lanes.py to match
the SLURM script's tunables):
  SWE_AGENT_LANES_M                  int    forks per mid_cp (default 8)
	  SWE_AGENT_LANES_MAX_ROUNDS         int    fork-group cap per instance, root counts (default 5)
	  SWE_AGENT_LANES_K                  int    asst turns between mid_cps (default 20)
	  SWE_AGENT_LANES_P                  int    active parents, currently must be 1
  SWE_AGENT_LANES_STEP_LIMIT         int    hard cap for the whole trajectory (default 100)
  SWE_AGENT_LANES_INSTANCE_WORKERS   int    concurrent instance subprocesses (default 8)
  SWE_AGENT_LANES_GT_EVAL_WORKERS    int    per-instance GT eval threads (default 8)
  SWE_AGENT_LANES_COMPLETION_MAX_TOKENS int per-call max_new_tokens (default 4096)
  SWE_AGENT_LANES_POLICY_GT_WEIGHT   float  policy GT reward weight (default 0.0)
  SWE_AGENT_LANES_POLICY_SIBLINGS_WEIGHT float sibling rubric reward weight (default 0.5)
  SWE_AGENT_LANES_POLICY_PC_WEIGHT   float  parent-child rubric reward weight (default 0.5)
  SWE_AGENT_LANES_POLICY_TEMPERATURE float  Lane A sampling temp (default 1.0)
  SWE_AGENT_LANES_POLICY_TOP_P       float  default 0.95
  SWE_AGENT_LANES_LANE_B_TEMPERATURE float  Lane B sampling temp (default 1.0)
  SWE_AGENT_LANES_LANE_B_TOP_P       float  default 0.95
  SWE_AGENT_LANES_FALLBACK_PATCH_PENALTY float default 0.5
  SWE_AGENT_LANES_OUTPUT_ROOT        str    where to write per-instance run dirs
  SWE_AGENT_LANES_API_HOST           str    sglang policy host (default http://127.0.0.1)
  SWE_AGENT_LANES_POLICY_PORTS       str    csv of ports (one per sglang engine)
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
from slime.utils.types import Sample

# Reuse v0's tokenizer cache + rollout-sample assembler — they're independent
# of the search topology. build_rollout_samples consumes ExportGroup -> Sample.
from train_agent.collect_grpo_rollout import _rollout_tokenizer, build_rollout_samples
from train_agent.serving.sglang_chat_service import start_slime_policy_route_warmup


logger = logging.getLogger("train_agent.collect_lanes_rollout_async")


# ---------------------------------------------------------------------------
# Subprocess entry: run one instance end-to-end and return its bundle
# ---------------------------------------------------------------------------


def _lanes_bundle_task(task: dict[str, Any]) -> dict[str, Any]:
    """Worker-process entry. Builds backend + runner for one instance,
    runs the v1 lane search, returns the GRPOExportBundle.

    Atomic-only: no per-group streaming queue. The whole bundle is built
    after runner.run() returns; harvest reads it from the Ray Future.
    """
    try:
        os.environ["SEARCH_SWE_SLIME_API_BASE"] = task["policy_base_url"]
        os.environ["SEARCH_SWE_SLIME_API_KEY"] = task["api_key"]
        # Imports inside the subprocess so env-var-dependent module init runs
        # AFTER we've set the slime API base.
        from swe_agent.backend import SWEAgentRolloutBackend
        from swe_agent.run.benchmarks.swebench import (
            build_swebench_config,
            get_swebench_docker_image_name,
            get_swebench_harness_namespace,
            load_swebench_instances_by_id,
        )
        from swe_agent.run.run_swe_agent import SWE_AGENT_TEXTBASED_CONFIG
        from swe_agent.trajectory_search_parallel import (
            LaneGRPOCollector,
            ParallelSearchConfig,
            TrajectorySearchParallelRunner,
        )
        from swe_agent.utils.serialize import recursive_merge

        instance_id = task["instance_id"]
        subset = task["subset"]
        split = task["split"]
        model_name = task["model_name"]
        policy_base_url = task["policy_base_url"]
        rubric_base_url = task["rubric_base_url"]
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
        overrides = {
            "agent": {"step_limit": task["step_limit"]},
            "model": {
                "model_kwargs": {
                    "api_base": api_base,
                    "api_key": api_key,
                    "max_tokens": task.get("completion_max_tokens", 4096),
                    "extra_body": (
                        {"chat_template_kwargs": {"enable_thinking": True}}
                        if "qwen" in model_name.lower() else {}
                    ),
                }
            },
            "environment": {"image": get_swebench_docker_image_name(instance)},
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

        cfg = ParallelSearchConfig(
            m=task["m"],
            n=task["n"],
            k=task["k"],
            p=task["p"],
            max_rounds=task["max_rounds"],
            step_limit=task["step_limit"],
            max_active_rubrics=task["max_active_rubrics"],
            gt_eval_workers=task.get("gt_eval_workers", 8),
            lane_b_pool_size=task.get("lane_b_pool_size") or (task["m"] * task["max_rounds"]),
            keep_images=False,
            policy_temperature=task.get("policy_temperature", 1.0),
            policy_top_p=task.get("policy_top_p", 0.95),
            lane_b_temperature=task.get("lane_b_temperature", 1.0),
            lane_b_top_p=task.get("lane_b_top_p", 0.95),
            fallback_patch_penalty=task.get("fallback_patch_penalty", 0.5),
        )
        run_dir = (
            output_root / instance_id
            / time.strftime("%Y%m%d-%H%M%S")
            / f"task-{task['index']:06d}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        runner = TrajectorySearchParallelRunner(
            instance=instance,
            backend=backend,
            run_dir=run_dir,
            policy_model_name=model_name,
            rubric_model_name=task.get("rubric_model_name") or model_name,
            judge_model_name=task.get("judge_model_name") or model_name,
            config=cfg,
            harness_namespace=get_swebench_harness_namespace(instance),
            policy_base_url=policy_base_url,
            rubric_base_url=rubric_base_url,
            api_key=api_key,
        )
        record = asyncio.run(runner.run())
        bundle = LaneGRPOCollector(
            policy_gt_weight=task.get("policy_gt_weight", 0.0),
            policy_siblings_weight=task.get("policy_siblings_weight", 0.5),
            policy_pc_weight=task.get("policy_pc_weight", 0.5),
        ).instance_record_to_bundle(record)
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
        # Per-task subprocess cleanup to bound RAM growth in a persistent
        # ProcessPoolExecutor worker (52084 leaked ~25 GB/task → OOM at step 42).
        # Drop refs to the heaviest per-task locals BEFORE returning so the
        # subprocess can release pages back to the OS via malloc_trim.
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
# Module-level rollout state (mirrors v0's pattern)
# ---------------------------------------------------------------------------


@dataclass
class _BufferedGroup:
    rollout_id: int
    samples: list[Sample]


_BUFFER: list[_BufferedGroup] = []
_TASK_INDEX = 0
_WARMUP_DONE = False
_STALE_DROPPED_GROUPS = 0
_FAILED_INSTANCES = 0
_DROPPED_OVERSIZED = 0

_NODE_WORKERS: list[Any] = []          # list[ray.actor.ActorHandle]
_NODE_WORKER_IPS: list[str] = []
_PENDING: dict[Any, dict[str, Any]] = {}   # ObjectRef -> task dict
_DISPATCH_COUNTER = 0


@ray.remote(num_cpus=2)
class _LanesNodeWorker:
    """Per-physical-node executor for v1 lane bundle tasks.

    Identical pattern to v0's _PDSNodeWorker but atomic-only (no streaming
    queue → no multiprocessing.Manager involved → no initializer needed)."""

    def __init__(self, name: str = "", max_workers: int = 8, max_tasks_per_child: int = 5):
        self.name = name
        self.max_workers = max_workers
        self.max_tasks_per_child = max_tasks_per_child
        from concurrent.futures import ProcessPoolExecutor
        # max_tasks_per_child recycles each subprocess after N tasks. Bounds
        # any cross-task RAM leak (52084 grew ~25 GB/task → OOM at step 42).
        try:
            self._executor = ProcessPoolExecutor(
                max_workers=max_workers,
                max_tasks_per_child=max_tasks_per_child,
            )
        except TypeError:
            # Python < 3.11 fallback
            self._executor = ProcessPoolExecutor(max_workers=max_workers)
        import socket
        self.ip = socket.gethostbyname(socket.gethostname())
        logger.info(
            f"[LanesNodeWorker {self.name}] up on ip={self.ip} "
            f"max_workers={max_workers}"
        )

    def get_ip(self) -> str:
        return self.ip

    def submit_task(self, task: dict[str, Any]) -> dict[str, Any]:
        future = self._executor.submit(_lanes_bundle_task, task)
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
    nodes = [
        n for n in ray.nodes()
        if n.get("Alive") and float(n.get("Resources", {}).get("GPU", 0)) > 0
    ]
    if not nodes:
        raise RuntimeError("No GPU-bearing Ray nodes found for Lanes worker spawn")
    actor_concurrency = per_node_concurrency + 4
    logger.info(
        f"[lanes-async] spawning {len(nodes)} node workers, "
        f"per-node concurrency={per_node_concurrency} "
        f"actor_concurrency={actor_concurrency}"
    )
    for n in nodes:
        node_id = n["NodeID"]
        node_name = n.get("NodeName") or n.get("NodeManagerAddress") or node_id[:8]
        worker = _LanesNodeWorker.options(
            num_cpus=2,
            max_concurrency=actor_concurrency,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False,
            ),
            name=f"lanes-node-worker-{node_name}",
        ).remote(name=node_name, max_workers=per_node_concurrency)
        ip = ray.get(worker.get_ip.remote())
        _NODE_WORKERS.append(worker)
        _NODE_WORKER_IPS.append(ip)
        logger.info(f"[lanes-async]   spawned worker on node={node_name} ip={ip}")


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


def _lanes_values_from_env() -> dict[str, Any]:
    """Read v1 lane-specific knobs from env."""
    def _int(name, default=None):
        v = os.environ.get(name, "")
        return int(v) if v else default

    def _float(name, default):
        v = os.environ.get(name, "")
        return float(v) if v else default

    return {
        "m": _int("SWE_AGENT_LANES_M", 8),
        "n": _int("SWE_AGENT_LANES_N", 1),
        "k": _int("SWE_AGENT_LANES_K", 20),
        "p": _int("SWE_AGENT_LANES_P", 1),
        "max_rounds": _int("SWE_AGENT_LANES_MAX_ROUNDS", 5),
        "step_limit": _int("SWE_AGENT_LANES_STEP_LIMIT", 100),
        "max_active_rubrics": _int("SWE_AGENT_LANES_MAX_ACTIVE_RUBRICS", 6),
        "gt_eval_workers": _int("SWE_AGENT_LANES_GT_EVAL_WORKERS", 8),
        "lane_b_pool_size": _int("SWE_AGENT_LANES_LANE_B_POOL_SIZE"),
        "completion_max_tokens": _int("SWE_AGENT_LANES_COMPLETION_MAX_TOKENS", 4096),
        "policy_gt_weight": _float(
            "SWE_AGENT_LANES_POLICY_GT_WEIGHT",
            _float("SWE_AGENT_LANES_POLICY_ALPHA", 0.0),
        ),
        "policy_siblings_weight": _float("SWE_AGENT_LANES_POLICY_SIBLINGS_WEIGHT", 0.5),
        "policy_pc_weight": _float("SWE_AGENT_LANES_POLICY_PC_WEIGHT", 0.5),
        "policy_temperature": _float("SWE_AGENT_LANES_POLICY_TEMPERATURE", 1.0),
        "policy_top_p": _float("SWE_AGENT_LANES_POLICY_TOP_P", 0.95),
        "lane_b_temperature": _float("SWE_AGENT_LANES_LANE_B_TEMPERATURE", 1.0),
        "lane_b_top_p": _float("SWE_AGENT_LANES_LANE_B_TOP_P", 0.95),
        "fallback_patch_penalty": _float("SWE_AGENT_LANES_FALLBACK_PATCH_PENALTY", 0.5),
    }


def _hash_to_index(s: str, n: int) -> int:
    import hashlib
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h, 16) % n


def _route_for_instance(instance_id: str, ports: list[int], host: str) -> str:
    p = ports[_hash_to_index(instance_id, len(ports))]
    return f"{host.rstrip('/')}:{p}"


def _submit_until_full(
    *,
    args,
    data_buffer,
    rollout_id: int,
    output_root: Path,
    model_name: str,
    max_pending: int,
) -> int:
    global _TASK_INDEX, _DISPATCH_COUNTER
    submitted = 0
    lanes_values = _lanes_values_from_env()

    policy_ports = [
        int(p) for p in os.environ.get("SWE_AGENT_LANES_POLICY_PORTS", "").split(",") if p
    ]
    rubric_ports = [
        int(p) for p in os.environ.get("SWE_AGENT_LANES_RUBRIC_PORTS", "").split(",") if p
    ]
    if not policy_ports:
        router_ip, router_port = (getattr(args, "sglang_model_routers", None) or {}).get(
            model_name, (args.sglang_router_ip, args.sglang_router_port),
        )
        policy_ports = [router_port]
        api_host = f"http://{router_ip}"
    else:
        api_host = os.environ.get("SWE_AGENT_LANES_API_HOST", "http://127.0.0.1")
    if not rubric_ports:
        rubric_ports = list(policy_ports)

    api_key = os.environ.get("SEARCH_SWE_SLIME_API_KEY", "EMPTY")

    while len(_PENDING) < max_pending:
        prompt_groups = data_buffer.get_samples(1)
        if not prompt_groups:
            break
        (prompt_group,) = prompt_groups
        # slime expands n_samples_per_prompt by duplicating the prompt; the
        # runner itself creates the M=8 sibling group, so we just take the
        # first duplicate. n_samples_per_prompt MUST equal SWE_AGENT_LANES_M.
        metadata = prompt_group[0].metadata
        instance_id = metadata["instance_id"]
        policy_base_url = _route_for_instance(instance_id, policy_ports, api_host)
        rubric_base_url = _route_for_instance(instance_id, rubric_ports, api_host)

        task = {
            "index": _TASK_INDEX,
            "rollout_id": rollout_id,
            "instance_id": instance_id,
            "subset": metadata["subset"],
            "split": metadata["split"],
            "output_root": str(output_root / f"rollout_{rollout_id:04d}"),
            "model_name": model_name,
            "rubric_model_name": os.environ.get("SWE_AGENT_LANES_RUBRIC_MODEL") or model_name,
            "judge_model_name": os.environ.get("SWE_AGENT_LANES_JUDGE_MODEL") or model_name,
            "policy_base_url": policy_base_url,
            "rubric_base_url": rubric_base_url,
            "api_key": api_key,
            **lanes_values,
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
    tokenizer,
    current_rollout_id: int,
    block: bool,
) -> int:
    """Reap Ray Futures whose subprocess fully returned. For each completed
    bundle, expand its policy_groups into Samples and append to _BUFFER."""
    global _FAILED_INSTANCES, _STALE_DROPPED_GROUPS, _DROPPED_OVERSIZED
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
        try:
            result = ray.get(ref)
        except Exception as exc:
            _FAILED_INSTANCES += 1
            logger.warning(
                "[lanes-async] instance %s failed: %s",
                task.get("instance_id"), str(exc)[:200],
            )
            del ref
            continue
        if result.get("error"):
            _FAILED_INSTANCES += 1
            logger.warning(
                "[lanes-async] instance %s failed: %s",
                result.get("instance_id"), result["error"][:200],
            )
            del result, ref
            continue
        bundle = result.get("bundle")
        if bundle is None:
            del result, ref
            continue
        # Target selection: v1 first pass only generates policy_groups.
        groups = (
            bundle.policy_groups if target == "policy" else bundle.rubric_groups
        )
        if task["rollout_id"] < current_rollout_id - 1:
            _STALE_DROPPED_GROUPS += len(groups)
            del result, bundle, groups, ref
            continue
        for group in groups:
            samples, dropped_here = build_rollout_samples(
                groups=[group],
                tokenizer=tokenizer,
                loss_mask_type=getattr(args, "loss_mask_type", "qwen3_5"),
                include_turn_rewards=False,
                max_sample_tokens=max_sample_tokens,
            )
            _DROPPED_OVERSIZED += dropped_here
            policy_version = f"rollout-{task['rollout_id']}"
            for sample in samples:
                sample.weight_versions = [policy_version]
                sample.metadata = {
                    **(sample.metadata or {}),
                    "policy_version": policy_version,
                    "source_rollout_id": task["rollout_id"],
                    "instance_id": task["instance_id"],
                }
            _BUFFER.append(
                _BufferedGroup(rollout_id=task["rollout_id"], samples=samples)
            )
            harvested += 1
        del result, bundle, groups, ref
    return harvested


def _drop_stale_buffer(current_rollout_id: int) -> None:
    global _STALE_DROPPED_GROUPS
    kept = [g for g in _BUFFER if g.rollout_id >= current_rollout_id - 1]
    _STALE_DROPPED_GROUPS += len(_BUFFER) - len(kept)
    _BUFFER[:] = kept


def _pop_groups(min_groups: int) -> list[_BufferedGroup]:
    # Shuffle the buffer before slicing to break completion-order bias.
    # Without this, easy instances (which finish fast → land in buffer first)
    # dominate the early training steps, and reward statistics drift down
    # as harder instances trickle in. Stale-drop ensures the buffer only
    # holds rollout_id in {current-1, current}, so shuffling across that
    # boundary is harmless. Reproducibility is not needed at this layer.
    # Set SWE_AGENT_LANES_NO_SHUFFLE=1 to disable (for A/B against v0 order).
    if os.environ.get("SWE_AGENT_LANES_NO_SHUFFLE", "0") != "1":
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
            "SWE_AGENT_LANES_OUTPUT_ROOT",
            "/workspace/rler/agent/outputs/lane_outputs/train_async_lanes",
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
            logger.warning("[lanes-async] warmup failed (continuing): %s", exc)
        _WARMUP_DONE = True

    tokenizer = _rollout_tokenizer(args)
    instance_workers = max(1, int(os.environ.get("SWE_AGENT_LANES_INSTANCE_WORKERS", "8")))
    max_pending = max(1, int(
        os.environ.get("SWE_AGENT_LANES_MAX_PENDING", str(instance_workers))
    ))
    min_ready_groups = max(1, int(
        os.environ.get("SWE_AGENT_LANES_MIN_READY_GROUPS", str(args.rollout_batch_size))
    ))
    timeout = float(os.environ.get("SWE_AGENT_LANES_WAIT_TIMEOUT", "10800"))

    if not _NODE_WORKERS:
        n_nodes = max(1, len([
            n for n in ray.nodes()
            if n.get("Alive") and float(n.get("Resources", {}).get("GPU", 0)) > 0
        ]))
        per_node = max(1, (instance_workers + n_nodes - 1) // n_nodes)
        _spawn_node_workers(per_node_concurrency=per_node)

    logger.info(
        "[lanes-async] generate_rollout START rollout_id=%d target=%s "
        "min_ready_groups=%d max_pending=%d buffer_at_entry=%d pending_at_entry=%d",
        rollout_id, target, min_ready_groups, max_pending,
        len(_BUFFER), len(_PENDING),
    )

    _drop_stale_buffer(rollout_id)
    _harvest_ready(
        args=args, target=target, tokenizer=tokenizer,
        current_rollout_id=rollout_id, block=False,
    )
    submitted = _submit_until_full(
        args=args, data_buffer=data_buffer, rollout_id=rollout_id,
        output_root=output_root, model_name=model_name, max_pending=max_pending,
    )
    logger.info(
        "[lanes-async] post-init submitted=%d buffer=%d pending=%d",
        submitted, len(_BUFFER), len(_PENDING),
    )

    wait_started = time.perf_counter()
    last_pulse = wait_started
    while len(_BUFFER) < min_ready_groups:
        if time.perf_counter() - wait_started > timeout:
            raise RuntimeError(
                f"Lanes async timed out waiting for samples: "
                f"buffer={len(_BUFFER)} pending={len(_PENDING)}"
            )
        _harvest_ready(
            args=args, target=target, tokenizer=tokenizer,
            current_rollout_id=rollout_id, block=bool(_PENDING),
        )
        submitted += _submit_until_full(
            args=args, data_buffer=data_buffer, rollout_id=rollout_id,
            output_root=output_root, model_name=model_name, max_pending=max_pending,
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
                "[lanes-async] WAITING rollout_id=%d elapsed=%.0fs "
                "buffer=%d/%d pending=%d submitted_total=%d",
                rollout_id, time.perf_counter() - wait_started,
                len(_BUFFER), min_ready_groups, len(_PENDING), submitted,
            )
            logger.info("[lanes-mem] %s", " | ".join(mems))
        if not _PENDING and len(_BUFFER) < min_ready_groups:
            raise RuntimeError(
                f"Lanes async produced no {target} samples and has no pending instance."
            )

    selected_groups = _pop_groups(min_ready_groups)
    _submit_until_full(
        args=args, data_buffer=data_buffer, rollout_id=rollout_id,
        output_root=output_root, model_name=model_name, max_pending=max_pending,
    )
    samples = [sample for group in selected_groups for sample in group.samples]
    for group_index, group in enumerate(selected_groups):
        for sample in group.samples:
            sample.group_index = group_index
    for index, sample in enumerate(samples):
        sample.index = index

    # --- per-batch metrics aggregated from sample.metadata + sample fields ---
    # Stashed by LaneGRPOCollector._build_branch_sample:
    #   n_continuation_steps, n_parent_steps, n_full_trace_steps,
    #   raw_gt_score, terminated_early, instance_id, is_dummy
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
    parent_steps = [
        int(s.metadata.get("n_parent_steps", 0) or 0)
        for s in samples if s.metadata
    ]
    # Exclude dummy samples (loss_mask all-zero placeholders for empty branches)
    # from reward/pass stats — they're not real model outputs.
    real_samples = [s for s in samples if not (s.metadata and s.metadata.get("is_dummy"))]
    n_dummy = len(samples) - len(real_samples)
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
    # Per-instance pass: instance succeeds if ANY of its branches pass.
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
    # Groups-per-instance = how many mid_cps per instance landed in this batch.
    from collections import Counter
    inst_group_count = Counter(
        s.metadata.get("instance_id") or "" for s in samples if s.metadata
    )
    m = max(1, int(os.environ.get("SWE_AGENT_LANES_M", "8")))
    groups_per_inst = (
        _safe_mean([c // m for c in inst_group_count.values()])
        if inst_group_count else 0.0
    )
    # Truncation rate: fraction of samples whose status reached the trajectory
    # step_limit without submitting (slime sets Sample.Status.TRUNCATED in that
    # case; we approximate from .status if available).
    try:
        from slime.utils.types import Sample as _SlimeSample
        trunc_count = sum(
            1 for s in samples
            if getattr(s, "status", None) == _SlimeSample.Status.TRUNCATED
        )
    except Exception:
        trunc_count = 0
    # Raw entropy proxy: mean of rollout-time per-token logprobs. More
    # negative = more entropy. Drift toward zero across rollouts signals
    # the policy is getting more confident (potential mode collapse).
    rollout_lps: list[float] = []
    for s in samples:
        lps = getattr(s, "rollout_log_probs", None)
        if lps:
            for x in lps:
                if isinstance(x, (int, float)):
                    rollout_lps.append(float(x))

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
            "swe_agent/stale_dropped_groups_total": _STALE_DROPPED_GROUPS,
            "swe_agent/dropped_oversized_samples_total": _DROPPED_OVERSIZED,
            "swe_agent/wait_seconds": time.perf_counter() - wait_started,
            "swe_agent/seconds": time.perf_counter() - started,
            "swe_agent/source": "lanes",
            # --- per-batch rollout shape (requested) ---
            "swe_agent/sample_cont_steps_mean": _safe_mean(cont_steps),
            "swe_agent/sample_full_steps_mean": _safe_mean(full_steps),
            "swe_agent/sample_parent_steps_mean": _safe_mean(parent_steps),
            "swe_agent/sample_gt_mean": _safe_mean(gts),
            "swe_agent/sample_pass_at_0.5": (n_passed / len(gts)) if gts else 0.0,
            "swe_agent/submit_rate": (submit_count / len(real_samples)) if real_samples else 0.0,
            "swe_agent/truncation_rate": (trunc_count / len(samples)) if samples else 0.0,
            "swe_agent/distinct_instances_in_batch": distinct_instances,
            "swe_agent/instance_pass_at_0.5": instance_pass,
            "swe_agent/avg_groups_per_instance_in_batch": groups_per_inst,
            "swe_agent/dummy_sample_count": n_dummy,
            "swe_agent/dummy_sample_rate": (n_dummy / len(samples)) if samples else 0.0,
            "swe_agent/rollout_logprob_mean": _safe_mean(rollout_lps),
            "swe_agent/rollout_logprob_token_count": len(rollout_lps),
        },
    )
