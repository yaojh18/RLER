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
  SWE_AGENT_NAIVE_COMPLETION_MAX_TOKENS int per-call max_new_tokens (default 20480)
  SWE_AGENT_NAIVE_POLICY_TEMPERATURE float  sampling temp (default 1.0)
  SWE_AGENT_NAIVE_POLICY_TOP_P       float  default 0.95
  SWE_AGENT_NAIVE_FALLBACK_PATCH_PENALTY float default 0.5
  SWE_AGENT_NAIVE_NO_ACTION_PATCH_PENALTY float default -0.1
  SWE_AGENT_NAIVE_OUTPUT_ROOT        str    where to write per-instance run dirs
  SWE_AGENT_NAIVE_ROLLOUT_POOL_SIZE  int    per-instance ThreadPoolExecutor cap (default = M)
  SWE_AGENT_NAIVE_SEED               int    seed for sampling (optional)
  SWE_AGENT_NAIVE_REWARD_KIND        str    hard, soft, joint, or f2p_only;
                                            default joint
  SWE_AGENT_NAIVE_ALL_PASS_REWARD    float  resolved reward multiplier (default 1)
Per-engine endpoints come from args.sglang_model_engines (populated at engine
init in slime/ray/rollout.py). The legacy SWE_AGENT_NAIVE_POLICY_PORTS env
var is no longer consulted.
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import json
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
from slime.rollout.data_source import (
    TrainingInstanceBudgetExhausted,
    TrainingValidationBoundaryReached,
)
from slime.rollout.filter_hub.base_types import call_dynamic_filter
from slime.utils.misc import load_function
from slime.utils.types import Sample

from train_agent.collect_grpo_rollout import build_rollout_samples
from train_agent.collector_checkpoint import (
    COLLECTOR_CHECKPOINT_SCHEMA_VERSION,
    checkpoint_task_source_group_index,
    deserialize_budget_error,
    deserialize_buffer,
    deserialize_validation_error,
    serialize_budget_error,
    serialize_buffer,
    serialize_pending_tasks,
    serialize_validation_error,
)
from train_agent.serving.sglang_chat_service import start_slime_policy_route_warmup
from swe_agent.exceptions import PolicyVersionMismatch
from swe_agent.policy_version import (
    checkpoint_policy_stale_lag,
    committed_policy_version,
)
from swe_agent.usage import (
    UsageMetricsTracker,
    build_usage_group_prefix,
    configure_usage_ledger,
    configure_usage_resume_window,
    infer_model_family,
    record_group_disposition,
    usage_ledger_offset,
    usage_context,
)


logger = logging.getLogger("train_agent.collect_naive_rollout_async")


class CollectorInfrastructureError(RuntimeError):
    """Fatal Ray worker/node failure that must stop source consumption."""


_FATAL_RAY_ERROR_NAMES = {
    "ActorDiedError",
    "LocalRayletDiedError",
    "NodeDiedError",
    "OutOfMemoryError",
    "RaySystemError",
    "WorkerCrashedError",
}
_FATAL_RAY_ERROR_MARKERS = (
    "worker(s) were killed due to the node running low on memory",
    "worker died unexpectedly while executing this task",
    "actor died unexpectedly before finishing this task",
    "local raylet died",
    "has been marked dead because the detector has missed too many heartbeats",
)


def _is_fatal_ray_infrastructure_error(exc: BaseException) -> bool:
    """Recognize node/worker loss without treating application errors as fatal.

    Ray may return the concrete exception directly or wrap it through
    ``__cause__``/``__context__``.  The message fallback covers the exact OOM
    signature observed in the long validation run while preserving ordinary
    ``RayTaskError`` instance failures as drop-and-refill events.
    """

    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if any(
            cls.__name__ in _FATAL_RAY_ERROR_NAMES
            for cls in type(current).__mro__
        ):
            return True
        message = str(current).lower()
        if any(marker in message for marker in _FATAL_RAY_ERROR_MARKERS):
            return True
        for nested in (current.__cause__, current.__context__):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


def _raise_fatal_ray_infrastructure_error(
    exc: BaseException,
    *,
    collector: str,
    instance_id: Any,
) -> None:
    if not _is_fatal_ray_infrastructure_error(exc):
        return
    raise CollectorInfrastructureError(
        f"{collector} collector lost a Ray worker/node while processing "
        f"instance {instance_id}; aborting before drawing replacement source "
        f"instances: {type(exc).__name__}: {str(exc)[:500]}"
    ) from exc


# ---------------------------------------------------------------------------
# Subprocess entry: run one instance end-to-end and return its bundle
# ---------------------------------------------------------------------------


def _naive_bundle_task(task: dict[str, Any]) -> dict[str, Any]:
    """Worker-process entry. Builds backend + naive runner for one instance,
    runs M independent linear rollouts, GT-scores each, returns the
    GRPOExportBundle (one ExportGroup of M samples).
    """
    try:
        configure_usage_ledger(task.get("usage_ledger") or None)
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
                "context_length": int(task.get("context_length", 128000)),
                "model_kwargs": {
                    "api_base": api_base,
                    "api_key": api_key,
                    "max_tokens": task.get("completion_max_tokens", 20480),
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

        reward_kind = task.get("reward_kind", "joint")
        if reward_kind not in {"hard", "soft", "joint", "f2p_only"}:
            raise ValueError(f"unsupported naive reward_kind={reward_kind!r}")
        cfg = NaiveSearchConfig(
            m=task["m"],
            step_limit=task["step_limit"],
            seed=task.get("seed"),
            gt_eval_workers=task.get("gt_eval_workers", 8),
            gt_eval_timeout=task.get("gt_eval_timeout", 600),
            evaluate_gt=not bool(task.get("defer_gt_evaluation")),
            rollout_pool_size=task.get("rollout_pool_size") or task["m"],
            policy_temperature=task.get("policy_temperature", 1.0),
            policy_top_p=task.get("policy_top_p", 0.95),
            fallback_patch_penalty=task.get("fallback_patch_penalty", 0.5),
            no_action_patch_penalty=task.get("no_action_patch_penalty", -0.1),
            reward_kind=reward_kind,
            joint_alpha=task.get("joint_alpha", 1.0),
            all_pass_reward=task.get("all_pass_reward", 1.0),
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
            policy_version=task.get("policy_version"),
            enforce_policy_version=bool(task.get("validation")),
            config=cfg,
            harness_namespace=get_swebench_harness_namespace(instance),
            policy_base_url=policy_base_url,
            policy_base_urls=policy_base_urls,
            api_key=api_key,
        )
        with usage_context(
            phase=task.get("usage_phase", "train"),
            group_id=task.get("usage_group_id", ""),
            model_family=infer_model_family(model_name),
            model_role="policy",
            suppress_accounting=bool(
                task.get("suppress_usage_accounting")
            ),
        ):
            record = asyncio.run(runner.run())
        if task.get("validation"):
            (run_dir / "validation_policy.json").write_text(
                json.dumps(
                    {
                        "policy_version": task.get("policy_version"),
                        "rollout_id": int(task["rollout_id"]),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        bundle = (
            None
            if task.get("defer_gt_evaluation")
            else naive_record_to_bundle(record)
        )
        validation = None
        validation_policy = None
        if task.get("validation"):
            rollout = record.rollouts[0] if len(record.rollouts) == 1 else None
            payload = rollout.evaluation_payload if rollout is not None else None
            if task.get("defer_gt_evaluation"):
                rollout_dir = (
                    runner._rollout_run_dir(rollout.rollout_index)
                    if rollout is not None
                    else run_dir
                )
                validation_policy = {
                    "rollout_dir": str(rollout_dir),
                    "terminal_patch": str(
                        rollout.terminal_patch if rollout is not None else ""
                    ),
                    "rollout_error": (
                        rollout.error
                        if rollout is not None
                        else f"expected one validation rollout, got {len(record.rollouts)}"
                    ),
                    "terminated_early": bool(
                        rollout.terminated_early if rollout is not None else False
                    ),
                    "rollout_status": str(
                        rollout.status if rollout is not None else "error"
                    ),
                    "total_tokens": dict(
                        rollout.total_tokens if rollout is not None else {}
                    ),
                    "policy_version": task.get("policy_version"),
                    "stale_policy_aborted": bool(
                        "PolicyVersionMismatch"
                        in str(
                            rollout.error
                            if rollout is not None
                            else record.error
                        )
                    ),
                }
            else:
                status = str((payload or {}).get("status") or "error")
                validation = {
                    "status": status,
                    "resolved": status == "resolved",
                    "infrastructure_error": bool(
                        (payload or {}).get("metainfo", {}).get("infrastructure_error")
                    ),
                    "rollout_error": rollout.error if rollout is not None else (
                        f"expected one validation rollout, got {len(record.rollouts)}"
                    ),
                    "terminated_early": bool(
                        rollout.terminated_early if rollout is not None else False
                    ),
                    "rollout_status": str(
                        rollout.status if rollout is not None else "error"
                    ),
                    "total_tokens": dict(
                        rollout.total_tokens if rollout is not None else {}
                    ),
                    "policy_version": task.get("policy_version"),
                    "stale_policy_aborted": bool(
                        "PolicyVersionMismatch"
                        in str(
                            rollout.error
                            if rollout is not None
                            else record.error
                        )
                    ),
                }
        result = {
            "index": task["index"],
            "instance_id": instance_id,
            "bundle": bundle,
            "validation": validation,
            "validation_policy": validation_policy,
            "error": "",
            "usage_group_id": task.get("usage_group_id", ""),
            "policy_version": task.get("policy_version"),
        }
    except Exception as exc:
        result = {
            "index": task.get("index", -1),
            "instance_id": task.get("instance_id", "?"),
            "bundle": None,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            "usage_group_id": task.get("usage_group_id", ""),
            "policy_version": task.get("policy_version"),
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


def _naive_validation_policy_batch_task(
    tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run many single-rollout validation instances in one process.

    Training already runs multiple independent sessions as threads inside
    each instance process. Reusing that proven shape gives validation the
    same policy-request concurrency without creating one heavyweight Python
    process per validation instance.
    """

    from concurrent.futures import ThreadPoolExecutor

    if not tasks:
        return []
    with ThreadPoolExecutor(
        max_workers=len(tasks),
        thread_name_prefix="naive-validation-policy",
    ) as executor:
        return list(executor.map(_naive_bundle_task, tasks))


def _naive_validation_gt_task(task: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a completed validation policy rollout in a bounded process."""

    descriptor = dict(task.get("validation_policy") or {})
    rollout_error = descriptor.get("rollout_error")
    base_validation = {
        "rollout_error": rollout_error,
        "terminated_early": bool(descriptor.get("terminated_early")),
        "rollout_status": str(descriptor.get("rollout_status") or "error"),
        "total_tokens": dict(descriptor.get("total_tokens") or {}),
        "policy_version": descriptor.get("policy_version"),
        "stale_policy_aborted": bool(descriptor.get("stale_policy_aborted")),
    }
    if rollout_error:
        return {
            **task,
            "bundle": None,
            "validation": {
                **base_validation,
                "status": "error",
                "resolved": False,
                "infrastructure_error": False,
            },
            "error": "",
        }

    rollout_dir = Path(str(descriptor["rollout_dir"]))
    evaluation_path = rollout_dir / "evaluation.json"
    try:
        from swe_agent.run.benchmarks.swebench import (
            get_swebench_harness_namespace,
            load_swebench_instances_by_id,
        )
        from swe_agent.run.run_swe_agent import (
            EvaluationRewardConfig,
            evaluate_swebench_instance_patches,
            make_evaluation_payload,
        )

        instances = load_swebench_instances_by_id(
            task["subset"],
            task["split"],
            [task["instance_id"]],
        )
        if not instances:
            raise RuntimeError(
                f"instance {task['instance_id']} not found in "
                f"{task['subset']}/{task['split']}"
            )
        instance = instances[0]
        raw_patch = str(descriptor.get("terminal_patch") or "").rstrip()
        patch = raw_patch + "\n" if raw_patch else ""
        reward_config = EvaluationRewardConfig(
            kind="hard",
            all_pass_reward=1.0,
        )
        rollout_dir.mkdir(parents=True, exist_ok=True)
        if not patch:
            payload = make_evaluation_payload(
                "empty",
                reward_config=reward_config,
            )
        else:
            eval_key = str(rollout_dir)
            payloads = evaluate_swebench_instance_patches(
                instance=instance,
                patches_by_key={eval_key: patch},
                model_name=task["model_name"],
                max_workers=1,
                timeout=int(task.get("gt_eval_timeout", 1800)),
                namespace=get_swebench_harness_namespace(instance),
                work_dir=rollout_dir,
                reward_config=reward_config,
            )
            if not isinstance(payloads, dict) or eval_key not in payloads:
                raise RuntimeError(f"missing evaluation payload for {eval_key}")
            payload = payloads[eval_key]
    except Exception as exc:
        try:
            from swe_agent.run.run_swe_agent import (
                EvaluationRewardConfig,
                make_evaluation_payload,
            )

            payload = make_evaluation_payload(
                "error",
                error=exc,
                reward_config=EvaluationRewardConfig(
                    kind="hard",
                    all_pass_reward=1.0,
                ),
                infrastructure_error=True,
            )
        except Exception:
            payload = {
                "status": "error",
                "metainfo": {"infrastructure_error": True},
            }

    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    status = str(payload.get("status") or "error")
    metainfo = payload.get("metainfo")
    metainfo = metainfo if isinstance(metainfo, dict) else {}
    return {
        **task,
        "bundle": None,
        "validation": {
            **base_validation,
            "status": status,
            "resolved": status == "resolved",
            "infrastructure_error": bool(
                payload.get("infrastructure_error")
                or metainfo.get("infrastructure_error")
            ),
        },
        "error": "",
    }


# ---------------------------------------------------------------------------
# Module-level rollout state (mirrors lanes pattern)
# ---------------------------------------------------------------------------


@dataclass
class _BufferedGroup:
    samples: list[Sample]
    rollout_id: int = 0
    usage_group_id: str = ""


_BUFFER: list[_BufferedGroup] = []
_TASK_INDEX = 0
_WARMUP_DONE = False
_STALE_DROPPED_GROUPS = 0
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
# Reward-distribution counters are updated after a complete group has valid
# training fields but before the dynamic filter runs.  Keeping cumulative
# sufficient statistics lets generate_rollout take an exact per-update delta
# without retaining every group's rewards for the life of the job.
_REWARD_GROUPS_OBSERVED = 0
_REWARD_GROUP_MEAN_SUM = 0.0
_REWARD_GROUP_VARIANCE_SUM = 0.0
_ZERO_VARIANCE_GROUPS = 0
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
_NODE_WORKER_GENERATION = 0
_PENDING: dict[Any, dict[str, Any]] = {}
_REPLAY_PENDING_TASKS: list[dict[str, Any]] = []
_DISPATCH_COUNTER = 0
_USAGE_TRACKER: UsageMetricsTracker | None = None
_USAGE_TRACKER_PATH = ""
_HEARTBEAT_EVENT_STEP = 0
_HEARTBEAT_OUTPUT_ROOT = ""
_SOURCE_BUDGET_EXHAUSTED = False
_SOURCE_BUDGET_ERROR: TrainingInstanceBudgetExhausted | None = None
_SOURCE_VALIDATION_ERROR: TrainingValidationBoundaryReached | None = None
# A generate() call can pause after accepting fewer than rollout_batch_size
# groups at an attempt-based validation boundary.  This state is deliberately
# process-local: train_async ACKs the boundary in the same job and retries the
# same rollout id, which may then refill the preserved groups.  No other
# generate boundary is allowed to retain hidden work.
_VALIDATION_PARTIAL_STATE: dict[str, Any] | None = None
# Terminal validation is fail-closed, but a retry must not regenerate policy
# trajectories that already completed successfully.  Live retry state stays in
# the dedicated AsyncValidationManager and is keyed by the exact
# rollout/source-attempt pair; driver restarts seed it from complete artifacts.
_EVAL_PARTIAL_STATE: dict[str, Any] | None = None


def checkpoint_state_dict(rollout_id: int) -> dict[str, Any]:
    """Return the exact untrained collector state after rollout ``rollout_id``."""
    return {
        "schema_version": COLLECTOR_CHECKPOINT_SCHEMA_VERSION,
        "collector": "naive",
        "checkpoint_rollout_id": int(rollout_id),
        "buffer": serialize_buffer(_BUFFER),
        "pending_tasks": (
            serialize_pending_tasks(_PENDING)
            + copy.deepcopy(_REPLAY_PENDING_TASKS)
        ),
        "task_index": int(_TASK_INDEX),
        "dispatch_counter": int(_DISPATCH_COUNTER),
        "usage_ledger_offset": usage_ledger_offset(
            _ensure_usage_tracking()
        ),
        "random_state": random.getstate(),
        "counters": {
            "stale_dropped_groups": int(_STALE_DROPPED_GROUPS),
            "failed_instances": int(_FAILED_INSTANCES),
            "truncated_oversized": int(_TRUNCATED_OVERSIZED),
            "filter_dropped_groups": int(_FILTER_DROPPED_GROUPS),
            "filter_drop_reasons": dict(_FILTER_DROP_REASONS),
            "infra_dropped_groups": int(_INFRA_DROPPED_GROUPS),
            "infra_drop_reasons": dict(_INFRA_DROP_REASONS),
            "total_groups_attempted": int(_TOTAL_GROUPS_ATTEMPTED),
            "reward_groups_observed": int(_REWARD_GROUPS_OBSERVED),
            "reward_group_mean_sum": float(_REWARD_GROUP_MEAN_SUM),
            "reward_group_variance_sum": float(
                _REWARD_GROUP_VARIANCE_SUM
            ),
            "zero_variance_groups": int(_ZERO_VARIANCE_GROUPS),
        },
        "source_budget_exhausted": bool(_SOURCE_BUDGET_EXHAUSTED),
        "source_budget_error": serialize_budget_error(
            _SOURCE_BUDGET_ERROR
        ),
        "source_validation_error": serialize_validation_error(
            _SOURCE_VALIDATION_ERROR
        ),
        "validation_partial_state": copy.deepcopy(
            _VALIDATION_PARTIAL_STATE
        ),
    }


def load_checkpoint_state_dict(
    state: dict[str, Any],
    rollout_id: int,
) -> None:
    """Restore logical state; live pending tasks are rebound on first generate."""
    global _BUFFER, _TASK_INDEX, _WARMUP_DONE
    global _STALE_DROPPED_GROUPS, _FAILED_INSTANCES
    global _TRUNCATED_OVERSIZED, _FILTER_DROPPED_GROUPS
    global _FILTER_DROP_REASONS, _INFRA_DROPPED_GROUPS
    global _INFRA_DROP_REASONS, _TOTAL_GROUPS_ATTEMPTED
    global _REWARD_GROUPS_OBSERVED, _REWARD_GROUP_MEAN_SUM
    global _REWARD_GROUP_VARIANCE_SUM, _ZERO_VARIANCE_GROUPS
    global _NODE_WORKERS, _NODE_WORKER_IPS
    global _NODE_WORKER_GENERATION, _PENDING
    global _REPLAY_PENDING_TASKS, _DISPATCH_COUNTER
    global _USAGE_TRACKER, _USAGE_TRACKER_PATH
    global _HEARTBEAT_EVENT_STEP, _HEARTBEAT_OUTPUT_ROOT
    global _SOURCE_BUDGET_EXHAUSTED, _SOURCE_BUDGET_ERROR
    global _SOURCE_VALIDATION_ERROR, _VALIDATION_PARTIAL_STATE

    if int(state.get("schema_version", -1)) != (
        COLLECTOR_CHECKPOINT_SCHEMA_VERSION
    ):
        raise RuntimeError(
            "unsupported naive collector checkpoint schema: "
            f"{state.get('schema_version')!r}"
        )
    if state.get("collector") != "naive":
        raise RuntimeError(
            "naive collector cannot restore state for "
            f"{state.get('collector')!r}"
        )
    if int(state.get("checkpoint_rollout_id", -1)) != int(rollout_id):
        raise RuntimeError(
            "naive collector checkpoint id mismatch: "
            f"requested={rollout_id} "
            f"saved={state.get('checkpoint_rollout_id')}"
        )
    if _BUFFER or _PENDING or _REPLAY_PENDING_TASKS:
        raise RuntimeError(
            "refusing to overlay a naive collector checkpoint on live state"
        )

    _BUFFER = deserialize_buffer(
        list(state.get("buffer") or []),
        _BufferedGroup,
    )
    _PENDING = {}
    _REPLAY_PENDING_TASKS = copy.deepcopy(
        list(state.get("pending_tasks") or [])
    )
    _TASK_INDEX = int(state.get("task_index", 0))
    _DISPATCH_COUNTER = int(state.get("dispatch_counter", 0))
    counters = dict(state.get("counters") or {})
    _STALE_DROPPED_GROUPS = int(
        counters.get("stale_dropped_groups", 0)
    )
    _FAILED_INSTANCES = int(counters.get("failed_instances", 0))
    _TRUNCATED_OVERSIZED = int(counters.get("truncated_oversized", 0))
    _FILTER_DROPPED_GROUPS = int(
        counters.get("filter_dropped_groups", 0)
    )
    _FILTER_DROP_REASONS = dict(
        counters.get("filter_drop_reasons") or {}
    )
    _INFRA_DROPPED_GROUPS = int(
        counters.get("infra_dropped_groups", 0)
    )
    _INFRA_DROP_REASONS = dict(
        counters.get("infra_drop_reasons") or {}
    )
    _TOTAL_GROUPS_ATTEMPTED = int(
        counters.get("total_groups_attempted", 0)
    )
    _REWARD_GROUPS_OBSERVED = int(
        counters.get("reward_groups_observed", 0)
    )
    _REWARD_GROUP_MEAN_SUM = float(
        counters.get("reward_group_mean_sum", 0.0)
    )
    _REWARD_GROUP_VARIANCE_SUM = float(
        counters.get("reward_group_variance_sum", 0.0)
    )
    _ZERO_VARIANCE_GROUPS = int(
        counters.get("zero_variance_groups", 0)
    )
    _SOURCE_BUDGET_EXHAUSTED = bool(
        state.get("source_budget_exhausted", False)
    )
    _SOURCE_BUDGET_ERROR = deserialize_budget_error(
        state.get("source_budget_error"),
        TrainingInstanceBudgetExhausted,
    )
    _SOURCE_VALIDATION_ERROR = deserialize_validation_error(
        state.get("source_validation_error"),
        TrainingValidationBoundaryReached,
    )
    _VALIDATION_PARTIAL_STATE = copy.deepcopy(
        state.get("validation_partial_state")
    )
    _restore_usage_tracking_checkpoint(
        state.get("usage_ledger_offset")
    )
    random_state = state.get("random_state")
    if random_state is not None:
        random.setstate(random_state)

    # Runtime resources are always recreated and rebound to the policy
    # checkpoint that was just loaded.
    _WARMUP_DONE = False
    _NODE_WORKERS = []
    _NODE_WORKER_IPS = []
    _NODE_WORKER_GENERATION = 0
    _ENDPOINT_INFLIGHT.clear()
    _USAGE_TRACKER = None
    _USAGE_TRACKER_PATH = ""
    _HEARTBEAT_EVENT_STEP = 0
    _HEARTBEAT_OUTPUT_ROOT = ""


def _mark_invalid_group(reason: str) -> None:
    """Count one expected group as invalid.

    This helper is deliberately separate from usage disposition recording:
    the usage ledger describes why a concrete group was not trained, while
    these counters enforce the optimizer-update conservation equation.
    """
    global _INFRA_DROPPED_GROUPS, _INFRA_DROP_REASONS
    _INFRA_DROPPED_GROUPS += 1
    key = str(reason or "unspecified").split(" ", 1)[0]
    _INFRA_DROP_REASONS[key] = _INFRA_DROP_REASONS.get(key, 0) + 1


def _observe_reward_group(args, samples: list[Sample]) -> None:
    """Accumulate pre-filter reward mean/variance for one valid GRPO group."""
    global _REWARD_GROUPS_OBSERVED
    global _REWARD_GROUP_MEAN_SUM, _REWARD_GROUP_VARIANCE_SUM
    global _ZERO_VARIANCE_GROUPS
    rewards = [float(sample.get_reward_value(args)) for sample in samples]
    if not rewards:
        raise ValueError("cannot observe reward statistics for an empty group")
    mean = sum(rewards) / len(rewards)
    variance = (
        sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
        if len(rewards) > 1
        else 0.0
    )
    _REWARD_GROUPS_OBSERVED += 1
    _REWARD_GROUP_MEAN_SUM += mean
    _REWARD_GROUP_VARIANCE_SUM += variance
    if variance <= 1e-12:
        _ZERO_VARIANCE_GROUPS += 1


def _group_outcome_metrics(
    *,
    attempted: int,
    accepted: int,
    invalid: int,
    dynamic_filtered: int,
    excess: int,
    carried_in: int = 0,
    carried_out: int = 0,
    prefix: str = "",
) -> dict[str, float | int]:
    """Build conserved per-update group counts and attempted-denominator rates.

    The Lane collector historically carries valid over-sampling excess into
    the next rollout cycle.  ``carried_in``/``carried_out`` make that lifecycle
    explicit without reclassifying valid data as dropped.
    """
    accounted = accepted + invalid + dynamic_filtered + excess
    available = attempted + carried_in
    consumed = accounted + carried_out
    if available != consumed:
        raise AssertionError(
            "group outcome conservation failed: "
            f"attempted={attempted} carried_in={carried_in} "
            f"accepted={accepted} invalid={invalid} "
            f"dynamic_filtered={dynamic_filtered} excess={excess} "
            f"carried_out={carried_out}"
        )
    stem = f"swe_agent/{prefix}"
    denominator = attempted if attempted > 0 else 1
    return {
        f"{stem}groups_attempted": attempted,
        f"{stem}groups_accepted": accepted,
        f"{stem}groups_invalid": invalid,
        f"{stem}groups_dynamic_filtered": dynamic_filtered,
        f"{stem}groups_excess": excess,
        f"{stem}groups_carried_in": carried_in,
        f"{stem}groups_carried_out": carried_out,
        f"{stem}groups_accepted_rate": accepted / denominator if attempted else 0.0,
        f"{stem}groups_invalid_rate": invalid / denominator if attempted else 0.0,
        f"{stem}groups_dynamic_filtered_rate": (
            dynamic_filtered / denominator if attempted else 0.0
        ),
        f"{stem}groups_excess_rate": excess / denominator if attempted else 0.0,
    }


def _ensure_usage_tracking() -> str:
    global _USAGE_TRACKER, _USAGE_TRACKER_PATH
    path = str(os.environ.get("RLER_USAGE_LEDGER_PATH") or "").strip()
    configure_usage_ledger(path or None)
    if path and (_USAGE_TRACKER is None or _USAGE_TRACKER_PATH != path):
        _USAGE_TRACKER = UsageMetricsTracker(path)
        _USAGE_TRACKER_PATH = path
    return path


def _restore_usage_tracking_checkpoint(
    historical_end_offset: int | None,
) -> None:
    """Reset the shared tracker around a durable model/data checkpoint."""

    global _USAGE_TRACKER, _USAGE_TRACKER_PATH
    configure_usage_resume_window(historical_end_offset)
    _USAGE_TRACKER = None
    _USAGE_TRACKER_PATH = ""


def _usage_metrics(
    *,
    commit_update: bool = False,
) -> dict[str, float | int]:
    """Return the small approximate token-cost surface logged to W&B."""

    _ensure_usage_tracking()
    if _USAGE_TRACKER is None:
        return {}
    if commit_update:
        metrics = _USAGE_TRACKER.commit_update()
    else:
        metrics = _USAGE_TRACKER.peek()
    retained = {
        "usage/event_step",
        "usage/total_tokens_cumulative",
        "usage/train_tokens_cumulative",
        "usage/validation_tokens_cumulative",
        "usage/qwen_tokens_cumulative",
        "usage/glm_tokens_cumulative",
    }
    return {key: value for key, value in metrics.items() if key in retained}


def _emit_heartbeat(
    *,
    source: str,
    rollout_id: int,
    elapsed_seconds: float,
    pending: int,
    buffered_groups: int,
    submitted: int,
    output_root: Path,
) -> None:
    """Persist a local pulse and mirror it to W&B when this process owns a run."""
    global _HEARTBEAT_EVENT_STEP, _HEARTBEAT_OUTPUT_ROOT
    output_root_key = str(output_root.resolve())
    if _HEARTBEAT_OUTPUT_ROOT != output_root_key:
        # A requeued Slurm job imports this module in a fresh process. Resume
        # the stable W&B heartbeat axis from the durable local ledger before
        # emitting the first new pulse, rather than restarting at one.
        _HEARTBEAT_EVENT_STEP = 0
        heartbeat_path = output_root / "heartbeat.jsonl"
        try:
            if heartbeat_path.exists():
                last_line = ""
                with heartbeat_path.open(encoding="utf-8") as stream:
                    for raw_line in stream:
                        if raw_line.strip():
                            last_line = raw_line
                if last_line:
                    _HEARTBEAT_EVENT_STEP = int(
                        json.loads(last_line).get("event_step", 0) or 0
                    )
        except Exception:
            # A malformed/unreadable last pulse must not block training.
            _HEARTBEAT_EVENT_STEP = 0
            logger.exception(
                "[%s-async] could not restore heartbeat event_step",
                source,
            )
        _HEARTBEAT_OUTPUT_ROOT = output_root_key
    _HEARTBEAT_EVENT_STEP += 1
    payload: dict[str, float | int | str] = {
        "timestamp": time.time(),
        "source": source,
        "rollout_id": int(rollout_id),
        "event_step": _HEARTBEAT_EVENT_STEP,
        "elapsed_seconds": float(elapsed_seconds),
        "pending": int(pending),
        "buffered_groups": int(buffered_groups),
        "submitted": int(submitted),
    }
    try:
        heartbeat_path = output_root / "heartbeat.jsonl"
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        with heartbeat_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        logger.exception("[%s-async] could not persist heartbeat", source)

    try:
        import wandb

        if wandb.run is None:
            return
        metrics: dict[str, float | int] = {
            "heartbeat/event_step": _HEARTBEAT_EVENT_STEP,
            "heartbeat/elapsed_seconds": float(elapsed_seconds),
            "heartbeat/pending": int(pending),
            "heartbeat/buffered_groups": int(buffered_groups),
            "heartbeat/submitted": int(submitted),
        }
        metrics.update(_usage_metrics())
        wandb.log(metrics)
    except Exception:
        # A telemetry failure must never kill an optimizer update. The local
        # heartbeat remains the scheduler-independent source of truth.
        logger.exception("[%s-async] W&B heartbeat failed", source)


def _usage_group_prefix(
    *,
    phase: str,
    rollout_id: int,
    instance_id: str,
    task_index: int,
    dataset_name: str = "",
) -> str:
    return build_usage_group_prefix(
        phase=phase,
        rollout_id=rollout_id,
        instance_id=instance_id,
        task_index=task_index,
        dataset_name=dataset_name,
    )


def _record_usage_disposition(
    group_id: str,
    *,
    disposition: str,
    reason: str = "",
    phase: str = "train",
) -> None:
    if not group_id:
        return
    _ensure_usage_tracking()
    record_group_disposition(
        group_id,
        disposition=disposition,
        reason=reason,
        phase=phase,
    )


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
        return self._submit_with_rebuild(
            _naive_bundle_task,
            task,
            instance_id=task.get("instance_id", "?"),
        )

    def _submit_with_rebuild(
        self,
        function,
        argument,
        *,
        instance_id: str,
    ):
        from concurrent.futures.process import BrokenProcessPool
        try:
            future = self._executor.submit(function, argument)
            return future.result()
        except BrokenProcessPool:
            logger.warning(
                "[NaiveNodeWorker %s] BrokenProcessPool — rebuilding "
                "executor and retrying instance=%s",
                self.name, instance_id,
            )
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._build_executor()
            future = self._executor.submit(function, argument)
            return future.result()

    def submit_validation_policy_batch(
        self,
        tasks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return self._submit_with_rebuild(
            _naive_validation_policy_batch_task,
            tasks,
            instance_id=f"validation-batch[{len(tasks)}]",
        )

    def submit_validation_gt(self, task: dict[str, Any]) -> dict[str, Any]:
        return self._submit_with_rebuild(
            _naive_validation_gt_task,
            task,
            instance_id=task.get("instance_id", "?"),
        )

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
    global _NODE_WORKERS, _NODE_WORKER_IPS, _NODE_WORKER_GENERATION
    if _NODE_WORKERS:
        return
    _NODE_WORKER_GENERATION += 1
    # SWE_AGENT_NAIVE_SKIP_NODE_IPS: csv of IPs/hostnames to exclude from
    # the NodeWorker pool. Used when the actor lives on a dedicated node
    # and we don't want rollout subprocess pressure on it (round-1 OOM at
    # step 23 was on the actor node co-hosting 6 instance workers).
    skip = {
        s.strip()
        for env_name in (
            "SWE_AGENT_NAIVE_SKIP_NODE_IPS",
            "SWE_AGENT_LANES_SKIP_NODE_IPS",
        )
        for s in os.environ.get(env_name, "").split(",")
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
    worker_role = "".join(
        character
        for character in os.environ.get(
            "RLER_NODE_WORKER_ROLE",
            "",
        ).strip().lower()
        if character.isalnum() or character in {"-", "_"}
    )
    role_suffix = f"-{worker_role}" if worker_role else ""
    for n in nodes:
        node_id = n["NodeID"]
        node_name = n.get("NodeName") or n.get("NodeManagerAddress") or node_id[:8]
        worker = _NaiveNodeWorker.options(
            num_cpus=2,
            max_concurrency=actor_concurrency,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False,
            ),
            name=(
                f"naive-node-worker{role_suffix}-{node_name}"
                f"-g{_NODE_WORKER_GENERATION}"
            ),
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


def _node_workers_are_healthy() -> bool:
    if not _NODE_WORKERS:
        return False
    try:
        worker_ips = ray.get(
            [worker.get_ip.remote() for worker in _NODE_WORKERS],
            timeout=15,
        )
    except Exception as exc:
        logger.warning(
            "[validation] node worker health check failed; rebuilding pool: "
            "%s: %s",
            type(exc).__name__,
            exc,
        )
        return False
    return len(worker_ips) == len(_NODE_WORKERS)


def _ensure_validation_node_workers(
    *,
    instance_workers: int,
    verify_existing: bool,
) -> None:
    if verify_existing and _NODE_WORKERS and not _node_workers_are_healthy():
        _shutdown_node_workers()
    if _NODE_WORKERS:
        return

    skip = {
        s.strip()
        for env_name in (
            "SWE_AGENT_NAIVE_SKIP_NODE_IPS",
            "SWE_AGENT_LANES_SKIP_NODE_IPS",
        )
        for s in os.environ.get(env_name, "").split(",")
        if s.strip()
    }
    n_nodes = max(
        1,
        len(
            [
                node
                for node in ray.nodes()
                if node.get("Alive")
                and float(node.get("Resources", {}).get("GPU", 0)) > 0
                and (node.get("NodeManagerAddress") or "") not in skip
                and (node.get("NodeName") or "") not in skip
            ]
        ),
    )
    _spawn_node_workers(
        per_node_concurrency=max(
            1, (instance_workers + n_nodes - 1) // n_nodes
        )
    )


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
        "gt_eval_timeout": _int(
            "SWE_AGENT_NAIVE_GT_EVAL_TIMEOUT", 600
        ),
        "rollout_pool_size": _int("SWE_AGENT_NAIVE_ROLLOUT_POOL_SIZE"),
        "completion_max_tokens": _int("SWE_AGENT_NAIVE_COMPLETION_MAX_TOKENS", 20480),
        "context_length": _int("SWE_AGENT_MODEL_CONTEXT_LENGTH", 128000),
        "policy_temperature": _float("SWE_AGENT_NAIVE_POLICY_TEMPERATURE", 1.0),
        "policy_top_p": _float("SWE_AGENT_NAIVE_POLICY_TOP_P", 0.95),
        "fallback_patch_penalty": _float("SWE_AGENT_NAIVE_FALLBACK_PATCH_PENALTY", 0.5),
        "no_action_patch_penalty": _float("SWE_AGENT_NAIVE_NO_ACTION_PATCH_PENALTY", -0.1),
        "reward_kind": (os.environ.get("SWE_AGENT_NAIVE_REWARD_KIND") or "joint").lower(),
        "joint_alpha": _float("SWE_AGENT_NAIVE_JOINT_ALPHA", 1.0),
        "all_pass_reward": _float("SWE_AGENT_NAIVE_ALL_PASS_REWARD", 1.0),
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


def _dispatch_policy_version(rollout_id: int) -> str | None:
    """Return the committed weight epoch, pausing dispatch during an update."""
    try:
        version = committed_policy_version()
    except PolicyVersionMismatch:
        return None
    return version or f"legacy-rollout-{int(rollout_id)}"


def _policy_endpoints_for_model(args, model_name: str) -> list[tuple[str, int]]:
    engines_map = getattr(args, "sglang_model_engines", None) or {}
    endpoints: list[tuple[str, int]] = list(engines_map.get(model_name, []))
    if not endpoints and len(engines_map) == 1:
        endpoints = list(next(iter(engines_map.values())))
    if endpoints:
        return [(str(host), int(port)) for host, port in endpoints]
    router_ip, router_port = (getattr(args, "sglang_model_routers", None) or {}).get(
        model_name, (args.sglang_router_ip, args.sglang_router_port),
    )
    logger.warning(
        "[naive-async] no per-engine endpoints for model=%s — "
        "falling back to single router %s:%s",
        model_name,
        router_ip,
        router_port,
    )
    return [(str(router_ip), int(router_port))]


def _replay_checkpoint_pending_tasks(
    *,
    args,
    rollout_id: int,
    output_root: Path,
    model_name: str,
) -> int:
    """Redispatch drawn-but-untrained instances under the resumed policy."""
    global _REPLAY_PENDING_TASKS, _DISPATCH_COUNTER
    if not _REPLAY_PENDING_TASKS:
        return 0
    if not _NODE_WORKERS:
        raise RuntimeError(
            "cannot replay naive pending tasks before node workers exist"
        )
    policy_version = _dispatch_policy_version(rollout_id)
    if policy_version is None:
        raise RuntimeError(
            "cannot replay naive pending tasks during a policy transition"
        )
    endpoints = _policy_endpoints_for_model(args, model_name)
    api_key = os.environ.get("SEARCH_SWE_SLIME_API_KEY", "EMPTY")
    replay_tasks = _REPLAY_PENDING_TASKS
    _REPLAY_PENDING_TASKS = []
    for original in replay_tasks:
        task = copy.deepcopy(original)
        old_usage_group_id = str(
            task.get("usage_group_id")
            or (
                f"{task.get('usage_group_prefix')}:g0"
                if task.get("usage_group_prefix")
                else ""
            )
        )
        _record_usage_disposition(
            old_usage_group_id,
            disposition="dropped",
            reason="checkpoint_replay",
        )
        source_group_index = checkpoint_task_source_group_index(task)
        usage_prefix = _usage_group_prefix(
            phase=str(task.get("usage_phase") or "train"),
            rollout_id=int(rollout_id),
            instance_id=str(task["instance_id"]),
            task_index=source_group_index,
            dataset_name=str(task.get("dataset_name") or ""),
        )
        policy_base_urls, endpoints_picked = (
            _pick_least_loaded_endpoints(
                int(task.get("m", 8)),
                endpoints,
            )
        )
        task.update(
            {
                "rollout_id": int(rollout_id),
                "output_root": str(
                    output_root / f"rollout_{int(rollout_id):04d}"
                ),
                "model_name": model_name,
                "policy_base_url": policy_base_urls[0],
                "policy_base_urls": policy_base_urls,
                "_endpoints_picked": endpoints_picked,
                "api_key": api_key,
                "usage_ledger": _ensure_usage_tracking(),
                # The durable ledger prefix already contains one approximate
                # cost for this pending source instance. Re-execution is a
                # checkpoint retry and must not be charged a second time.
                "suppress_usage_accounting": True,
                "policy_version": policy_version,
                "source_group_index": source_group_index,
                "usage_group_prefix": usage_prefix,
                "usage_group_id": f"{usage_prefix}:g0",
            }
        )
        worker = _NODE_WORKERS[
            _DISPATCH_COUNTER % len(_NODE_WORKERS)
        ]
        _DISPATCH_COUNTER += 1
        ref = worker.submit_task.remote(task)
        _PENDING[ref] = task
    logger.info(
        "[naive-async] replayed %d checkpointed pending source instances "
        "under policy_version=%s rollout_id=%d",
        len(replay_tasks),
        policy_version,
        rollout_id,
    )
    return len(replay_tasks)


def _read_validation_records(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        records: list[Any] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid validation JSONL {path}:{line_number}: {exc}"
                    ) from exc
        return records
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("instances", "rows", "samples", "data"):
            if isinstance(payload.get(key), list):
                return list(payload[key])
    raise ValueError(
        f"validation manifest {path} must be JSONL, a JSON list, or contain "
        "one of instances/rows/samples/data"
    )


def _validation_rows_from_args(args) -> list[tuple[str, dict[str, Any]]]:
    """Read logical validation manifests without using them as benchmark data.

    Rows provide only IDs and split metadata.  `_naive_bundle_task` resolves
    each ID through the repository's local Hugging Face snapshot.
    """
    datasets = list(getattr(args, "eval_datasets", None) or [])
    if not datasets:
        raise ValueError(
            "SWE validation requires --eval-prompt-data/--eval-config pointing "
            "at the fixed validation manifest"
        )
    rows: list[tuple[str, dict[str, Any]]] = []
    for dataset_cfg in datasets:
        dataset_name = str(getattr(dataset_cfg, "name", "") or "swebench_validation")
        path = Path(str(getattr(dataset_cfg, "path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"validation manifest does not exist: {path}")
        n_eval = getattr(dataset_cfg, "n_samples_per_eval_prompt", None)
        if n_eval not in (None, 1):
            raise ValueError(
                f"{dataset_name}: validation protocol requires exactly one "
                f"rollout per instance, got n_samples_per_eval_prompt={n_eval}"
            )
        metadata_key = str(getattr(dataset_cfg, "metadata_key", "") or "metadata")
        metadata_overrides = dict(
            getattr(dataset_cfg, "metadata_overrides", None) or {}
        )
        seen: set[str] = set()
        dataset_rows: list[tuple[str, dict[str, Any]]] = []
        for raw in _read_validation_records(path):
            if isinstance(raw, str):
                raw = {"instance_id": raw}
            if not isinstance(raw, dict):
                raise TypeError(
                    f"{dataset_name}: validation rows must be mappings or IDs"
                )
            nested = raw.get(metadata_key)
            metadata = dict(nested) if isinstance(nested, dict) else {}
            for key in (
                "instance_id",
                "subset",
                "split",
                "dataset_split",
                "hf_split",
                "fold_id",
                "fold_partition",
                "partition",
                "repo",
            ):
                if key in raw and key not in metadata:
                    metadata[key] = raw[key]
            metadata.update(metadata_overrides)
            instance_id = str(metadata.get("instance_id") or "").strip()
            if not instance_id:
                raise ValueError(
                    f"{dataset_name}: every validation row needs metadata.instance_id"
                )
            if instance_id in seen:
                raise ValueError(
                    f"{dataset_name}: duplicate validation instance_id={instance_id}"
                )
            seen.add(instance_id)
            logical_partition = (
                metadata.get("fold_partition")
                or metadata.get("partition")
                or "validation"
            )
            dataset_rows.append(
                (
                    dataset_name,
                    {
                        **metadata,
                        "instance_id": instance_id,
                        "subset": str(metadata.get("subset") or "verified"),
                        # The logical 250/50/200 partition must not overwrite
                        # SWE-bench Verified's physical Hugging Face split.
                        "split": str(
                            metadata.get("dataset_split")
                            or metadata.get("hf_split")
                            or metadata.get("split")
                            or "test"
                        ),
                        "fold_partition": str(logical_partition),
                    },
                )
            )
        min_eval = getattr(dataset_cfg, "min_eval_samples", None)
        if min_eval is not None and len(dataset_rows) < int(min_eval):
            raise ValueError(
                f"{dataset_name}: manifest has {len(dataset_rows)} rows, "
                f"below min_eval_samples={min_eval}"
            )
        rows.extend(dataset_rows)
    return rows


def _validation_sample(
    *,
    result: dict[str, Any],
    metadata: dict[str, Any],
    index: int,
) -> tuple[Sample, bool, bool]:
    validation = dict(result.get("validation") or {})
    status = str(validation.get("status") or "error")
    infrastructure_error = bool(validation.get("infrastructure_error"))
    rollout_error = validation.get("rollout_error") or result.get("error")
    stale_policy_aborted = bool(
        validation.get("stale_policy_aborted")
        or "PolicyVersionMismatch" in str(rollout_error or "")
    )
    completed = (
        status in {"resolved", "unresolved", "empty"}
        and not infrastructure_error
        and not rollout_error
    )
    resolved = completed and status == "resolved"

    sample: Sample | None = None
    bundle = result.get("bundle")
    groups = list(getattr(bundle, "policy_groups", None) or [])
    if groups:
        try:
            built, _ = build_rollout_samples(
                groups=[groups[0]],
                include_turn_rewards=False,
            )
            if built:
                sample = built[0]
        except Exception as exc:
            logger.warning(
                "[validation] could not materialize sample for %s: %s",
                metadata["instance_id"],
                exc,
            )
    if sample is None:
        sample = Sample(
            prompt=[],
            tokens=[],
            response="",
            response_length=0,
        )
    sample.index = index
    sample.group_index = None
    sample.reward = 1.0 if resolved else 0.0
    sample.status = Sample.Status.COMPLETED if completed else Sample.Status.FAILED
    sample.metadata = {
        **(sample.metadata or {}),
        **metadata,
        "validation_status": status,
        "validation_completed": completed,
        "validation_resolved": resolved,
        "validation_infrastructure_error": infrastructure_error,
        "validation_rollout_error": str(rollout_error or ""),
        "validation_total_tokens": dict(validation.get("total_tokens") or {}),
        "validation_policy_version": validation.get("policy_version"),
        "validation_stale_policy_aborted": stale_policy_aborted,
    }
    truncated = "trunc" in str(validation.get("rollout_status") or "").lower()
    return sample, completed, truncated


def _persisted_validation_entry(
    *,
    output_root: Path,
    rollout_id: int,
    dataset_name: str,
    metadata: dict[str, Any],
    index: int,
    policy_version: str,
) -> tuple[str, Sample, bool, bool] | None:
    """Restore one artifact-complete validation result after driver restart.

    Validation is deliberately asynchronous, so a Slurm wall-time boundary
    can arrive after most terminal rollouts have already completed.  Ray actor
    memory does not survive that boundary.  Reuse the earliest complete result
    for an instance so a later recovery attempt cannot silently replace the
    policy snapshot that the validation cursor was meant to measure.
    """

    instance_root = (
        output_root
        / "validation"
        / f"rollout_{rollout_id:04d}"
        / dataset_name.replace("/", "__")
        / str(metadata["instance_id"])
    )
    if not instance_root.is_dir():
        return None

    for evaluation_path in sorted(instance_root.rglob("evaluation.json")):
        run_root = evaluation_path.parent
        messages_path = run_root / "messages.json"
        patch_path = run_root / "model_patch.json"
        # NaiveSearchRunner stores rollout artifacts below
        # <task-run>/<instance>/<rollout>, while the worker records the pinned
        # policy once at <task-run>/validation_policy.json.
        policy_path = run_root.parent.parent / "validation_policy.json"
        if (
            not messages_path.is_file()
            or not patch_path.is_file()
            or not policy_path.is_file()
        ):
            continue
        try:
            evaluation = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
            messages = json.loads(messages_path.read_text(encoding="utf-8"))
            patch = json.loads(patch_path.read_text(encoding="utf-8"))
            policy_metadata = json.loads(
                policy_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if (
            not isinstance(evaluation, dict)
            or not isinstance(messages, dict)
            or not isinstance(patch, dict)
            or not isinstance(policy_metadata, dict)
        ):
            continue
        if str(policy_metadata.get("policy_version") or "") != str(
            policy_version
        ):
            continue

        status = str(evaluation.get("status") or "error")
        metainfo = evaluation.get("metainfo")
        metainfo = metainfo if isinstance(metainfo, dict) else {}
        infrastructure_error = bool(
            evaluation.get("infrastructure_error")
            or metainfo.get("infrastructure_error")
        )
        if (
            status not in {"resolved", "unresolved", "empty"}
            or infrastructure_error
        ):
            continue

        resolved = status == "resolved"
        sample = Sample(
            prompt=[],
            tokens=[],
            response="",
            response_length=0,
        )
        sample.index = index
        sample.group_index = None
        sample.reward = 1.0 if resolved else 0.0
        sample.status = Sample.Status.COMPLETED
        sample.metadata = {
            **metadata,
            "validation_status": status,
            "validation_completed": True,
            "validation_resolved": resolved,
            "validation_infrastructure_error": False,
            "validation_rollout_error": "",
            "validation_total_tokens": {},
            "validation_persisted": True,
            "validation_artifact": str(evaluation_path),
            "validation_policy_version": str(policy_version),
            "validation_stale_policy_aborted": False,
        }
        return dataset_name, sample, True, False
    return None


def _restore_persisted_validation_entries(
    *,
    rows: list[tuple[str, dict[str, Any]]],
    output_root: Path,
    rollout_id: int,
    policy_version: str,
) -> dict[int, tuple[str, Sample, bool, bool]]:
    restored: dict[int, tuple[str, Sample, bool, bool]] = {}
    for index, (dataset_name, metadata) in enumerate(rows):
        entry = _persisted_validation_entry(
            output_root=output_root,
            rollout_id=rollout_id,
            dataset_name=dataset_name,
            metadata=metadata,
            index=index,
            policy_version=policy_version,
        )
        if entry is not None:
            restored[index] = entry
    return restored


def generate_validation_rollout(
    args,
    rollout_id: int,
    *,
    output_root: Path,
    model_name: str,
) -> RolloutFnEvalOutput:
    """One policy-only terminal rollout and binary GT evaluation per val ID."""
    global _TASK_INDEX, _DISPATCH_COUNTER, _EVAL_PARTIAL_STATE
    rows = _validation_rows_from_args(args)
    if not rows:
        raise ValueError("validation manifest is empty")
    policy_version = str(
        getattr(args, "eval_policy_version", "") or ""
    ).strip()
    if not policy_version:
        policy_version = (
            _dispatch_policy_version(rollout_id)
            or f"legacy-rollout-{int(rollout_id)}"
        )
    eval_key = (
        int(rollout_id),
        int(getattr(args, "eval_instance_attempt", -1) or -1),
        policy_version,
    )
    row_signature = tuple(
        (dataset_name, str(metadata["instance_id"]))
        for dataset_name, metadata in rows
    )
    if (
        _EVAL_PARTIAL_STATE is not None
        and (
            _EVAL_PARTIAL_STATE.get("key") != eval_key
            or _EVAL_PARTIAL_STATE.get("row_signature") != row_signature
        )
    ):
        logger.warning(
            "[validation] discarding incomplete resume state for key=%s "
            "before starting key=%s",
            _EVAL_PARTIAL_STATE.get("key"),
            eval_key,
        )
        _EVAL_PARTIAL_STATE = None
    if _EVAL_PARTIAL_STATE is None:
        restored = _restore_persisted_validation_entries(
            rows=rows,
            output_root=output_root,
            rollout_id=rollout_id,
            policy_version=policy_version,
        )
        _EVAL_PARTIAL_STATE = {
            "key": eval_key,
            "row_signature": row_signature,
            "completed": restored,
            "calls": 0,
        }
        if restored:
            logger.warning(
                "[validation] restored persisted artifacts key=%s "
                "completed=%d missing=%d",
                eval_key,
                len(restored),
                len(rows) - len(restored),
            )
    _EVAL_PARTIAL_STATE["calls"] += 1
    completed_entries: dict[
        int, tuple[str, Sample, bool, bool]
    ] = _EVAL_PARTIAL_STATE["completed"]
    missing_indices = [
        index for index in range(len(rows)) if index not in completed_entries
    ]
    if _EVAL_PARTIAL_STATE["calls"] > 1:
        logger.warning(
            "[validation] missing-only resume key=%s completed=%d missing=%d",
            eval_key,
            len(completed_entries),
            len(missing_indices),
        )

    policy_records: list[
        tuple[
            int,
            str,
            dict[str, Any],
            tuple[str, int],
            str,
            dict[str, Any],
        ]
    ] = []
    stale_before_dispatch = bool(
        getattr(args, "eval_policy_stale_before_dispatch", False)
    )
    dispatch_indices = (
        [] if stale_before_dispatch else list(missing_indices)
    )
    if dispatch_indices:
        policy_concurrency = max(
            1,
            int(
                os.environ.get(
                    "SWE_AGENT_VALIDATION_INSTANCE_WORKERS",
                    "8",
                )
            ),
        )
        process_workers = min(
            policy_concurrency,
            max(
                1,
                int(
                    os.environ.get(
                        "SWE_AGENT_VALIDATION_PROCESS_WORKERS",
                        "8",
                    )
                ),
            ),
        )
        _ensure_validation_node_workers(
            instance_workers=process_workers,
            verify_existing=_EVAL_PARTIAL_STATE["calls"] > 1,
        )
        endpoints = _policy_endpoints_for_model(args, model_name)
        api_key = os.environ.get("SEARCH_SWE_SLIME_API_KEY", "EMPTY")
    for index in dispatch_indices:
        dataset_name, metadata = rows[index]
        policy_urls, picked = _pick_least_loaded_endpoints(1, endpoints)
        usage_prefix = _usage_group_prefix(
            phase="validation",
            rollout_id=rollout_id,
            instance_id=metadata["instance_id"],
            task_index=_TASK_INDEX,
            dataset_name=dataset_name,
        )
        task = {
            "index": _TASK_INDEX,
            "rollout_id": rollout_id,
            "instance_id": metadata["instance_id"],
            "subset": metadata["subset"],
            "split": metadata["split"],
            "output_root": str(
                output_root
                / "validation"
                / f"rollout_{rollout_id:04d}"
                / dataset_name.replace("/", "__")
            ),
            "model_name": model_name,
            "policy_base_url": policy_urls[0],
            "policy_base_urls": policy_urls,
            "api_key": api_key,
            "m": 1,
            "step_limit": int(
                os.environ.get("SWE_AGENT_VALIDATION_STEP_LIMIT", "120")
            ),
            "gt_eval_workers": 1,
            "gt_eval_timeout": int(
                os.environ.get(
                    "SWE_AGENT_VALIDATION_GT_EVAL_TIMEOUT", "1800"
                )
            ),
            "rollout_pool_size": 1,
            "completion_max_tokens": int(
                os.environ.get(
                    "SWE_AGENT_VALIDATION_COMPLETION_MAX_TOKENS", "20480"
                )
            ),
            "context_length": int(
                os.environ.get("SWE_AGENT_MODEL_CONTEXT_LENGTH", "128000")
            ),
            "policy_temperature": float(
                os.environ.get("SWE_AGENT_VALIDATION_TEMPERATURE", "1.0")
            ),
            "policy_top_p": float(
                os.environ.get("SWE_AGENT_VALIDATION_TOP_P", "0.95")
            ),
            "fallback_patch_penalty": 1.0,
            "no_action_patch_penalty": 0.0,
            "reward_kind": "hard",
            "joint_alpha": 1.0,
            "all_pass_reward": 1.0,
            "validation": True,
            "defer_gt_evaluation": True,
            "usage_ledger": _ensure_usage_tracking(),
            "usage_phase": "validation",
            "usage_group_prefix": usage_prefix,
            "usage_group_id": f"{usage_prefix}:g0",
            "policy_version": policy_version,
        }
        _TASK_INDEX += 1
        policy_records.append(
            (
                index,
                dataset_name,
                metadata,
                picked[0],
                task["usage_group_id"],
                task,
            )
        )

    # Match baseline's proven process x thread shape: a small bounded process
    # pool owns all heavyweight imports, while each process carries several
    # independent single-session validation rollouts as threads.
    policy_results: list[
        tuple[int, str, dict[str, Any], str, dict[str, Any], dict[str, Any]]
    ] = []
    if policy_records:
        n_batches = min(process_workers, len(policy_records))
        batches: list[list[Any]] = [[] for _ in range(n_batches)]
        for offset, record in enumerate(policy_records):
            batches[offset % n_batches].append(record)
        pending_batches: list[tuple[list[Any], Any]] = []
        for batch in batches:
            worker = _NODE_WORKERS[_DISPATCH_COUNTER % len(_NODE_WORKERS)]
            _DISPATCH_COUNTER += 1
            ref = worker.submit_validation_policy_batch.remote(
                [record[-1] for record in batch]
            )
            pending_batches.append((batch, ref))
        for batch, ref in pending_batches:
            try:
                batch_results = ray.get(ref)
                if len(batch_results) != len(batch):
                    raise RuntimeError(
                        "validation policy batch returned "
                        f"{len(batch_results)} results for {len(batch)} tasks"
                    )
            except Exception as exc:
                batch_results = [
                    {
                        "bundle": None,
                        "validation": None,
                        "validation_policy": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    for _ in batch
                ]
            finally:
                _release_endpoints([record[3] for record in batch])
            for record, result in zip(batch, batch_results):
                index, dataset_name, metadata, _, usage_group_id, task = record
                policy_results.append(
                    (
                        index,
                        dataset_name,
                        metadata,
                        usage_group_id,
                        task,
                        result,
                    )
                )

    # Policy generation is complete before evaluator work enters the process
    # pool, so slow SWE-bench harnesses cannot reduce policy concurrency.
    gt_pending: list[
        tuple[int, str, dict[str, Any], str, Any]
    ] = []
    direct_results: list[
        tuple[int, str, dict[str, Any], str, dict[str, Any]]
    ] = []
    for (
        index,
        dataset_name,
        metadata,
        usage_group_id,
        task,
        result,
    ) in policy_results:
        descriptor = result.get("validation_policy")
        if not isinstance(descriptor, dict):
            direct_results.append(
                (index, dataset_name, metadata, usage_group_id, result)
            )
            continue
        gt_task = {**task, "validation_policy": descriptor}
        worker = _NODE_WORKERS[_DISPATCH_COUNTER % len(_NODE_WORKERS)]
        _DISPATCH_COUNTER += 1
        gt_pending.append(
            (
                index,
                dataset_name,
                metadata,
                usage_group_id,
                worker.submit_validation_gt.remote(gt_task),
            )
        )

    attempt_entries: dict[int, tuple[str, Sample, bool, bool]] = {}
    if stale_before_dispatch:
        for index in missing_indices:
            dataset_name, metadata = rows[index]
            sample, completed, truncated = _validation_sample(
                result={
                    "bundle": None,
                    "error": (
                        "PolicyVersionMismatch: validation checkpoint "
                        "was stale before policy dispatch"
                    ),
                    "validation": {
                        "status": "error",
                        "infrastructure_error": False,
                        "rollout_error": (
                            "PolicyVersionMismatch: validation checkpoint "
                            "was stale before policy dispatch"
                        ),
                        "policy_version": policy_version,
                        "stale_policy_aborted": True,
                    },
                },
                metadata=metadata,
                index=index,
            )
            attempt_entries[index] = (
                dataset_name,
                sample,
                completed,
                truncated,
            )
    final_results = list(direct_results)
    for index, dataset_name, metadata, usage_group_id, ref in gt_pending:
        try:
            result = ray.get(ref)
        except Exception as exc:
            result = {
                "bundle": None,
                "validation": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        final_results.append(
            (index, dataset_name, metadata, usage_group_id, result)
        )
    for index, dataset_name, metadata, usage_group_id, result in final_results:
        sample, completed, truncated = _validation_sample(
            result=result,
            metadata=metadata,
            index=index,
        )
        _record_usage_disposition(
            str(result.get("usage_group_id") or usage_group_id),
            disposition="accepted" if completed else "invalid",
            reason="" if completed else "validation_error",
            phase="validation",
        )
        entry = (dataset_name, sample, completed, truncated)
        attempt_entries[index] = entry
        if completed:
            completed_entries[index] = entry

    combined_entries = dict(completed_entries)
    combined_entries.update(
        {
            index: entry
            for index, entry in attempt_entries.items()
            if index not in combined_entries
        }
    )
    by_dataset: dict[str, list[tuple[int, Sample, bool, bool]]] = {}
    for index, (dataset_name, sample, completed, truncated) in sorted(
        combined_entries.items()
    ):
        by_dataset.setdefault(dataset_name, []).append(
            (index, sample, completed, truncated)
        )

    data: dict[str, dict[str, Any]] = {}
    metrics: dict[str, Any] = {}
    total_attempted = total_completed = total_resolved = 0
    total_stale_policy_aborted = 0
    for dataset_name, entries in by_dataset.items():
        entries.sort(key=lambda item: item[0])
        samples = [item[1] for item in entries]
        completed_samples = [
            item[1] for item in entries if bool(item[2])
        ]
        completed_truncated = [
            bool(item[3]) for item in entries if bool(item[2])
        ]
        completed = sum(bool(item[2]) for item in entries)
        resolved = sum(
            float(sample.reward or 0.0) >= 0.5
            for sample in completed_samples
        )
        stale_policy_aborted = sum(
            bool(
                (sample.metadata or {}).get(
                    "validation_stale_policy_aborted",
                    False,
                )
            )
            for sample in samples
        )
        attempted = len(samples)
        errors = attempted - completed
        if completed_samples:
            data[dataset_name] = {
                "rewards": [
                    float(sample.reward or 0.0)
                    for sample in completed_samples
                ],
                "truncated": completed_truncated,
                "samples": completed_samples,
            }
        prefix = f"eval/{dataset_name}"
        metrics.update(
            {
                f"{prefix}/attempted": attempted,
                f"{prefix}/completed": completed,
                f"{prefix}/errors": errors,
                f"{prefix}/resolved": resolved,
                f"{prefix}/resolved_rate": (
                    resolved / completed if completed else 0.0
                ),
                f"{prefix}/coverage": (
                    completed / attempted if attempted else 0.0
                ),
                f"{prefix}/stale_policy_aborted": stale_policy_aborted,
            }
        )
        total_attempted += attempted
        total_completed += completed
        total_resolved += resolved
        total_stale_policy_aborted += stale_policy_aborted
    metrics.update(
        {
            "eval/attempted": total_attempted,
            "eval/completed": total_completed,
            "eval/errors": total_attempted - total_completed,
            "eval/resolved": total_resolved,
            "eval/resolved_rate": (
                total_resolved / total_completed if total_completed else 0.0
            ),
            "eval/coverage": (
                total_completed / total_attempted
                if total_attempted
                else 0.0
            ),
            "eval/stale_policy_aborted": total_stale_policy_aborted,
            "eval/incomplete": int(total_completed != total_attempted),
        }
    )
    metrics.update(_usage_metrics())
    output = RolloutFnEvalOutput(data=data, metrics=metrics)
    if total_completed == len(rows):
        _EVAL_PARTIAL_STATE = None
    return output


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
    global _SOURCE_BUDGET_EXHAUSTED, _SOURCE_BUDGET_ERROR
    global _SOURCE_VALIDATION_ERROR
    submitted = 0
    naive_values = _naive_values_from_env()
    if _SOURCE_VALIDATION_ERROR is not None:
        scheduled = int(
            getattr(
                data_buffer,
                "last_validation_scheduled_attempt",
                getattr(data_buffer, "last_validation_attempt", 0),
            )
            or 0
        )
        boundary = int(_SOURCE_VALIDATION_ERROR.boundary or 0)
        if scheduled >= boundary:
            _SOURCE_VALIDATION_ERROR = None

    # Per-engine endpoints discovered at engine init in slime/ray/rollout.py.
    # Each entry is (host, port) for one sglang engine; the policy dispatcher
    # picks the least-loaded one per trial, then pins all turns of that trial
    # to the chosen endpoint (sticky per trial -> KV cache reuse within the
    # trial, spread across trials of an instance).
    endpoints = _policy_endpoints_for_model(args, model_name)

    api_key = os.environ.get("SEARCH_SWE_SLIME_API_KEY", "EMPTY")

    # Keep the CPU/SGLang pipeline full across optimizer batches.  The
    # rollout-id stale guard, not an artificial batch-local drain, bounds how
    # long completed or in-flight work may survive.
    while len(_PENDING) < max_pending:
        if _SOURCE_BUDGET_EXHAUSTED or _SOURCE_VALIDATION_ERROR is not None:
            break
        policy_version = _dispatch_policy_version(rollout_id)
        if policy_version is None:
            break
        try:
            prompt_groups = data_buffer.get_samples(1)
        except TrainingValidationBoundaryReached as exc:
            _SOURCE_VALIDATION_ERROR = exc
            logger.info("[naive-async] %s", exc)
            break
        except TrainingInstanceBudgetExhausted as exc:
            _SOURCE_BUDGET_EXHAUSTED = True
            _SOURCE_BUDGET_ERROR = exc
            logger.info("[naive-async] %s", exc)
            break
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

        usage_prefix = _usage_group_prefix(
            phase="train",
            rollout_id=rollout_id,
            instance_id=instance_id,
            # The source cursor is checkpointed and therefore stable across
            # Slurm requeues.  _TASK_INDEX is process-local and restarts at
            # zero, so using it here makes replayed logical groups
            # indistinguishable from new source attempts in the usage ledger.
            task_index=int(prompt_group[0].group_index),
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
            "usage_ledger": _ensure_usage_tracking(),
            "usage_phase": "train",
            "usage_group_prefix": usage_prefix,
            "usage_group_id": f"{usage_prefix}:g0",
            "source_group_index": int(
                prompt_group[0].group_index
            ),
            "policy_version": policy_version,
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
    current_rollout_id: int | None = None,
) -> int:
    """Reap Ray Futures whose subprocess fully returned. For each completed
    bundle, expand its policy_groups into Samples and append to _BUFFER."""
    global _FAILED_INSTANCES, _TRUNCATED_OVERSIZED
    global _FILTER_DROPPED_GROUPS, _FILTER_DROP_REASONS
    global _INFRA_DROPPED_GROUPS, _INFRA_DROP_REASONS, _TOTAL_GROUPS_ATTEMPTED
    global _STALE_DROPPED_GROUPS
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
        ready, _ = ray.wait(
            pending_refs,
            num_returns=1,
            timeout=float(os.environ.get("RLER_COLLECTOR_POLL_SECONDS", "5")),
        )
    else:
        ready, _ = ray.wait(pending_refs, num_returns=len(pending_refs), timeout=0)
    for ref in ready:
        task = _PENDING.pop(ref)
        source_rollout_id = int(
            task.get(
                "rollout_id",
                0 if current_rollout_id is None else current_rollout_id,
            )
        )
        effective_rollout_id = (
            source_rollout_id
            if current_rollout_id is None
            else int(current_rollout_id)
        )
        policy_stale_lag = (
            None
            if current_rollout_id is None
            else checkpoint_policy_stale_lag(
                str(task.get("policy_version") or ""),
                consumer_rollout_id=effective_rollout_id,
            )
        )
        usage_group_id = str(task.get("usage_group_id") or "")
        # The bundle is no longer in-flight on any engine — release the
        # M endpoint slots we reserved at dispatch, regardless of outcome.
        # Tuples come back from Ray as lists; coerce to tuple for dict key.
        _release_endpoints([
            tuple(ep) for ep in task.get("_endpoints_picked", [])
        ])
        if policy_stale_lag is not None and policy_stale_lag < 0:
            raise RuntimeError(
                "naive collector observed policy weights from the future: "
                f"consumer={effective_rollout_id} "
                f"policy={task.get('policy_version')!r}"
            )
        if (
            source_rollout_id < effective_rollout_id - 1
            or (
                policy_stale_lag is not None
                and policy_stale_lag > 1
            )
        ):
            # Restore the pre-diff stale=1 contract exactly. Stale work still
            # counts in the token ledger, but is not an on-policy attempted
            # group for this optimizer batch.
            try:
                ray.get(ref)
            except Exception as exc:
                _raise_fatal_ray_infrastructure_error(
                    exc,
                    collector="naive",
                    instance_id=task.get("instance_id"),
                )
            _STALE_DROPPED_GROUPS += 1
            _record_usage_disposition(
                usage_group_id,
                disposition="dropped",
                reason="stale",
            )
            del ref
            continue
        try:
            result = ray.get(ref)
        except Exception as exc:
            _raise_fatal_ray_infrastructure_error(
                exc,
                collector="naive",
                instance_id=task.get("instance_id"),
            )
            _TOTAL_GROUPS_ATTEMPTED += 1
            _FAILED_INSTANCES += 1
            _mark_invalid_group("ray_task_error")
            _record_usage_disposition(
                usage_group_id,
                disposition="invalid",
                reason="ray_task_error",
            )
            logger.warning(
                "[naive-async] instance %s failed: %s",
                task.get("instance_id"), str(exc)[:200],
            )
            del ref
            continue
        # A returned worker result is one attempted model group even when the
        # runner later reports an application/bundle error. A lost Ray node is
        # infrastructure recovery and is intentionally excluded.
        _TOTAL_GROUPS_ATTEMPTED += 1
        if result.get("error"):
            _FAILED_INSTANCES += 1
            invalid_reason = (
                "policy_version_mismatch"
                if "PolicyVersionMismatch" in str(result["error"])
                else "runner_error"
            )
            _mark_invalid_group(invalid_reason)
            _record_usage_disposition(
                usage_group_id,
                disposition="invalid",
                reason=invalid_reason,
            )
            logger.warning(
                "[naive-async] instance %s failed: %s",
                result.get("instance_id"), result["error"][:200],
            )
            del result, ref
            continue
        if result.get("policy_version") != task.get("policy_version"):
            _mark_invalid_group("policy_version_metadata_mismatch")
            _record_usage_disposition(
                usage_group_id,
                disposition="invalid",
                reason="policy_version_metadata_mismatch",
            )
            del result, ref
            continue
        bundle = result.get("bundle")
        if bundle is None:
            _mark_invalid_group("missing_bundle")
            _record_usage_disposition(
                usage_group_id,
                disposition="invalid",
                reason="missing_bundle",
            )
            del result, ref
            continue
        # Naive baseline is policy-only. target=rubric is invalid here.
        groups = (
            bundle.policy_groups if target == "policy" else bundle.rubric_groups
        )
        if len(groups) != 1:
            drop_reason = (bundle.metadata or {}).get("group_dropped_reason")
            invalid_reason = "invalid_branch" if drop_reason else "wrong_group_count"
            _mark_invalid_group(str(drop_reason or invalid_reason))
            _record_usage_disposition(
                usage_group_id,
                disposition="invalid",
                reason=invalid_reason,
            )
            del result, bundle, groups, ref
            continue
        for group in groups:
            samples, truncated_here = build_rollout_samples(
                groups=[group],
                include_turn_rewards=False,
                max_sample_tokens=max_sample_tokens,
            )
            _TRUNCATED_OVERSIZED += truncated_here
            if len(samples) != int(task["m"]):
                _mark_invalid_group("wrong_sample_count")
                _record_usage_disposition(
                    usage_group_id,
                    disposition="invalid",
                    reason="wrong_sample_count",
                )
                continue
            for sample in samples:
                policy_version = str(task["policy_version"])
                sample.weight_versions = [policy_version]
                sample.metadata = {
                    **(sample.metadata or {}),
                    "policy_version": policy_version,
                    "source_rollout_id": source_rollout_id,
                    "instance_id": task["instance_id"],
                }
            _observe_reward_group(args, samples)
            if dynamic_filter is not None and samples:
                filter_out = call_dynamic_filter(dynamic_filter, args, samples)
                if not filter_out.keep:
                    _FILTER_DROPPED_GROUPS += 1
                    reason = filter_out.reason or "unspecified"
                    _FILTER_DROP_REASONS[reason] = _FILTER_DROP_REASONS.get(reason, 0) + 1
                    _record_usage_disposition(
                        usage_group_id,
                        disposition="filtered",
                        reason=(
                            "zero_variance"
                            if reason.startswith("zero_std")
                            else reason
                        ),
                    )
                    continue
            _BUFFER.append(
                _BufferedGroup(
                    samples=samples,
                    rollout_id=source_rollout_id,
                    usage_group_id=usage_group_id,
                )
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


def _drop_stale_buffer(current_rollout_id: int) -> None:
    global _STALE_DROPPED_GROUPS
    kept: list[_BufferedGroup] = []
    for group in _BUFFER:
        policy_versions = {
            str((sample.metadata or {}).get("policy_version") or "")
            for sample in group.samples
        }
        if len(policy_versions) > 1:
            raise RuntimeError(
                "buffered naive group mixes policy versions: "
                f"{sorted(policy_versions)}"
            )
        policy_version = (
            next(iter(policy_versions)) if policy_versions else ""
        )
        policy_stale_lag = checkpoint_policy_stale_lag(
            policy_version,
            consumer_rollout_id=current_rollout_id,
        )
        if policy_stale_lag is not None and policy_stale_lag < 0:
            raise RuntimeError(
                "buffered naive group has policy weights from the future: "
                f"consumer={current_rollout_id} "
                f"policy={policy_version!r}"
            )
        if (
            group.rollout_id >= int(current_rollout_id) - 1
            and (
                policy_stale_lag is None
                or policy_stale_lag <= 1
            )
        ):
            kept.append(group)
            continue
        _STALE_DROPPED_GROUPS += 1
        _record_usage_disposition(
            group.usage_group_id,
            disposition="dropped",
            reason="stale",
        )
    _BUFFER[:] = kept


def _drop_incomplete_source_budget_buffer(
    *, reason: str = "source_budget_partial_update"
) -> None:
    """Mark valid but untrainable tail groups dropped before terminal stop."""
    groups = list(_BUFFER)
    _BUFFER.clear()
    for group in groups:
        _record_usage_disposition(
            group.usage_group_id,
            disposition="dropped",
            reason=reason,
        )


def _pending_validation_boundary(
    data_buffer,
) -> TrainingValidationBoundaryReached | None:
    """Return an exact attempted-instance boundary awaiting validation."""
    progress_fn = getattr(data_buffer, "training_progress", None)
    if not callable(progress_fn):
        return None
    progress = progress_fn()
    interval_value = progress.get("eval_instance_interval")
    if interval_value is None:
        return None
    interval = int(interval_value)
    if interval <= 0:
        raise ValueError(
            f"eval_instance_interval must be positive, got {interval}"
        )
    attempted = int(progress.get("attempted_instances", 0) or 0)
    scheduled = int(
        progress.get(
            "last_validation_scheduled_attempt",
            progress.get("last_validation_attempt", 0),
        )
        or 0
    )
    boundary = (attempted // interval) * interval
    if boundary <= 0 or boundary <= scheduled:
        return None
    return TrainingValidationBoundaryReached(
        attempted_instances=attempted,
        boundary=boundary,
    )


def _validation_partial_resume_state(
    *,
    data_buffer,
    rollout_id: int,
) -> dict[str, Any] | None:
    """Return metrics state for a same-rollout validation retry.

    Ordinary buffers and pending tasks are intentionally allowed across
    generate calls; stale=1 is the lifecycle guard.
    """
    state = _VALIDATION_PARTIAL_STATE
    if state is None:
        return None
    if int(state["rollout_id"]) != int(rollout_id):
        raise RuntimeError(
            "Naive validation partial must retry the same rollout id: "
            f"preserved={state['rollout_id']} requested={rollout_id}"
        )
    boundary = int(state["boundary"])
    source_boundary = int(
        getattr(_SOURCE_VALIDATION_ERROR, "boundary", 0) or 0
    )
    if source_boundary != boundary:
        raise RuntimeError(
            "Naive validation partial lost its source-boundary guard: "
            f"preserved={boundary} source={source_boundary}"
        )
    scheduled = int(
        getattr(
            data_buffer,
            "last_validation_scheduled_attempt",
            getattr(data_buffer, "last_validation_attempt", 0),
        )
        or 0
    )
    if scheduled < boundary:
        raise RuntimeError(
            "Naive validation partial cannot resume before validation is "
            f"scheduled: scheduled={scheduled} boundary={boundary}"
        )
    checkpoint_pending_count = (
        len(_PENDING) + len(_REPLAY_PENDING_TASKS)
    )
    if (
        int(state["group_count"]) != len(_BUFFER)
        or int(state["pending_count"]) != checkpoint_pending_count
    ):
        raise RuntimeError(
            "Naive validation partial changed while paused: "
            f"groups={state['group_count']}->{len(_BUFFER)} "
            f"pending={state['pending_count']}->{checkpoint_pending_count}"
        )
    return state


# ---------------------------------------------------------------------------
# Slime rollout entry
# ---------------------------------------------------------------------------


def generate_rollout(args, rollout_id: int, data_buffer, evaluation: bool = False):
    """Slime rollout entry — called once per training rollout cycle."""
    global _WARMUP_DONE, _SOURCE_VALIDATION_ERROR
    global _VALIDATION_PARTIAL_STATE
    output_root = Path(
        os.environ.get(
            "SWE_AGENT_NAIVE_OUTPUT_ROOT",
            "/workspace/rler/agent/outputs/naive_outputs/train_async_naive",
        )
    )
    model_name = os.environ.get("SWE_AGENT_GRPO_MODEL_NAME") or "Qwen/Qwen3.5-9B"
    if evaluation:
        return generate_validation_rollout(
            args,
            rollout_id,
            output_root=output_root,
            model_name=model_name,
        )

    _drop_stale_buffer(rollout_id)
    resume_state = _validation_partial_resume_state(
        data_buffer=data_buffer,
        rollout_id=rollout_id,
    )
    started = (
        float(resume_state["started"])
        if resume_state is not None
        else time.perf_counter()
    )
    target = os.environ.get("SWE_AGENT_GRPO_TARGET", "policy")
    if resume_state is not None:
        attempted_at_start = int(resume_state["attempted_at_start"])
        invalid_at_start = int(resume_state["invalid_at_start"])
        filtered_at_start = int(resume_state["filtered_at_start"])
        stale_at_start = int(resume_state["stale_at_start"])
        carried_in = int(resume_state["carried_in"])
        reward_stats_at_start = tuple(resume_state["reward_stats_at_start"])
    else:
        carried_in = len(_BUFFER)
        attempted_at_start = _TOTAL_GROUPS_ATTEMPTED
        invalid_at_start = _INFRA_DROPPED_GROUPS
        filtered_at_start = _FILTER_DROPPED_GROUPS
        stale_at_start = _STALE_DROPPED_GROUPS
        reward_stats_at_start = (
            _REWARD_GROUPS_OBSERVED,
            _REWARD_GROUP_MEAN_SUM,
            _REWARD_GROUP_VARIANCE_SUM,
            _ZERO_VARIANCE_GROUPS,
        )
    if _SOURCE_VALIDATION_ERROR is not None:
        scheduled = int(
            getattr(
                data_buffer,
                "last_validation_scheduled_attempt",
                getattr(data_buffer, "last_validation_attempt", 0),
            )
            or 0
        )
        if scheduled >= int(_SOURCE_VALIDATION_ERROR.boundary or 0):
            _SOURCE_VALIDATION_ERROR = None

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
    replayed_pending = _replay_checkpoint_pending_tasks(
        args=args,
        rollout_id=rollout_id,
        output_root=output_root,
        model_name=model_name,
    )

    logger.info(
        "[naive-async] generate_rollout START rollout_id=%d target=%s "
        "min_ready_groups=%d max_pending=%d buffer_at_entry=%d pending_at_entry=%d",
        rollout_id, target, min_ready_groups, max_pending,
        len(_BUFFER), len(_PENDING),
    )

    submitted = (
        int(resume_state["submitted"])
        if resume_state is not None
        else 0
    ) + replayed_pending

    def _preserve_and_raise_validation(
        exc: TrainingValidationBoundaryReached,
    ) -> None:
        global _VALIDATION_PARTIAL_STATE
        preserved_count = len(_BUFFER)
        preserved_pending = len(_PENDING)
        exc.preserved_partial = (
            preserved_count > 0 or preserved_pending > 0
        )
        exc.preserved_group_count = preserved_count
        exc.preserved_pending_count = preserved_pending
        exc.preserved_group_kinds = (
            {"root": preserved_count} if preserved_count else {}
        )
        _VALIDATION_PARTIAL_STATE = {
            "rollout_id": int(rollout_id),
            "boundary": int(exc.boundary or 0),
            "group_count": preserved_count,
            "pending_count": preserved_pending,
            "started": started,
            "attempted_at_start": attempted_at_start,
            "invalid_at_start": invalid_at_start,
            "filtered_at_start": filtered_at_start,
            "stale_at_start": stale_at_start,
            "carried_in": carried_in,
            "reward_stats_at_start": reward_stats_at_start,
            "submitted": submitted,
        }
        raise exc

    if _SOURCE_VALIDATION_ERROR is not None:
        _preserve_and_raise_validation(_SOURCE_VALIDATION_ERROR)

    _harvest_ready(
        args=args,
        target=target,
        current_rollout_id=rollout_id,
        block=False,
    )
    submitted += _submit_until_full(
        args=args, data_buffer=data_buffer, rollout_id=rollout_id,
        output_root=output_root, model_name=model_name, max_pending=max_pending,
        target_groups=min_ready_groups,
    )
    if _SOURCE_VALIDATION_ERROR is not None:
        _preserve_and_raise_validation(_SOURCE_VALIDATION_ERROR)
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
            args=args,
            target=target,
            current_rollout_id=rollout_id,
            block=bool(_PENDING),
        )
        submitted += _submit_until_full(
            args=args, data_buffer=data_buffer, rollout_id=rollout_id,
            output_root=output_root, model_name=model_name, max_pending=max_pending,
            target_groups=min_ready_groups,
        )
        if _SOURCE_VALIDATION_ERROR is not None:
            _preserve_and_raise_validation(_SOURCE_VALIDATION_ERROR)
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
            _emit_heartbeat(
                source="naive",
                rollout_id=rollout_id,
                elapsed_seconds=time.perf_counter() - wait_started,
                pending=len(_PENDING),
                buffered_groups=len(_BUFFER),
                submitted=submitted,
                output_root=output_root,
            )
        if not _PENDING and len(_BUFFER) < min_ready_groups:
            if _SOURCE_BUDGET_EXHAUSTED:
                _VALIDATION_PARTIAL_STATE = None
                _drop_incomplete_source_budget_buffer()
                if _SOURCE_BUDGET_ERROR is None:
                    raise AssertionError(
                        "source budget is exhausted without its terminal error"
                    )
                raise _SOURCE_BUDGET_ERROR
            raise RuntimeError(
                f"Naive async produced no {target} samples and has no pending instance."
            )

    # As with Lane collection, attempt N can be the group that exactly fills
    # the optimizer batch.  Enforce validation before returning that batch,
    # even though the collector does not need to request attempt N+1.
    pending_boundary = _pending_validation_boundary(data_buffer)
    if pending_boundary is not None:
        _SOURCE_VALIDATION_ERROR = pending_boundary
        _preserve_and_raise_validation(pending_boundary)

    selected_groups = _pop_groups(min_ready_groups)
    _VALIDATION_PARTIAL_STATE = None
    submitted += _submit_until_full(
        args=args,
        data_buffer=data_buffer,
        rollout_id=rollout_id,
        output_root=output_root,
        model_name=model_name,
        max_pending=max_pending,
        target_groups=min_ready_groups,
    )
    # A test double, or a very fast local dispatcher, may synchronously place
    # newly prefetched groups in the buffer.  Count carry only after that
    # refill so every attempted group remains visible in the conservation
    # equation.  Merely pending tasks have not produced an attempted group yet.
    carried_out = len(_BUFFER)
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
    filter_dropped_step = _FILTER_DROPPED_GROUPS - filtered_at_start
    infra_dropped_step = _INFRA_DROPPED_GROUPS - invalid_at_start
    stale_dropped_step = _STALE_DROPPED_GROUPS - stale_at_start
    attempted_step = _TOTAL_GROUPS_ATTEMPTED - attempted_at_start
    n_groups_kept = len(selected_groups)
    outcome_metrics = _group_outcome_metrics(
        attempted=attempted_step,
        accepted=n_groups_kept,
        invalid=infra_dropped_step,
        dynamic_filtered=filter_dropped_step,
        excess=0,
        carried_in=carried_in,
        carried_out=carried_out,
    )
    filter_drop_rate_step = outcome_metrics[
        "swe_agent/groups_dynamic_filtered_rate"
    ]
    infra_drop_rate_step = (
        infra_dropped_step / attempted_step if attempted_step > 0 else 0.0
    )
    reward_groups_step = (
        _REWARD_GROUPS_OBSERVED - reward_stats_at_start[0]
    )
    reward_mean_sum_step = (
        _REWARD_GROUP_MEAN_SUM - reward_stats_at_start[1]
    )
    reward_variance_sum_step = (
        _REWARD_GROUP_VARIANCE_SUM - reward_stats_at_start[2]
    )
    zero_variance_groups_step = (
        _ZERO_VARIANCE_GROUPS - reward_stats_at_start[3]
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

    source_progress_fn = getattr(data_buffer, "training_progress", None)
    source_progress = (
        source_progress_fn() if source_progress_fn is not None else {}
    )
    source_attempted = int(
        source_progress.get(
            "attempted_instances",
            getattr(data_buffer, "sample_group_index", 0),
        )
    )
    source_budget = source_progress.get("instance_budget")
    source_budget_remaining = (
        max(0, int(source_budget) - source_attempted)
        if source_budget is not None
        else -1
    )

    output = RolloutFnTrainOutput(
        samples=samples,
        metrics={
            "swe_agent/target": target,
            "swe_agent/samples": len(samples),
            "swe_agent/groups": len(selected_groups),
            "swe_agent/pending_instances": len(_PENDING),
            "swe_agent/buffer_groups": len(_BUFFER),
            "swe_agent/submitted_instances": submitted,
            "swe_agent/failed_instances_total": _FAILED_INSTANCES,
            "swe_agent/stale_dropped_groups_total": (
                _STALE_DROPPED_GROUPS
            ),
            "swe_agent/truncated_oversized_samples_total": _TRUNCATED_OVERSIZED,
            "swe_agent/filter_dropped_groups_total": _FILTER_DROPPED_GROUPS,
            "swe_agent/infra_dropped_groups_total": _INFRA_DROPPED_GROUPS,
            "swe_agent/total_groups_attempted_total": _TOTAL_GROUPS_ATTEMPTED,
            # Source-instance attempts are the experiment's epoch/validation
            # axis. Failed and dynamically filtered groups are included.
            "swe_agent/train_instances_attempted": source_attempted,
            "swe_agent/train_epoch": float(
                source_progress.get("epoch", 0.0)
            ),
            "swe_agent/train_instance_budget_remaining": (
                source_budget_remaining
            ),
            "swe_agent/last_validation_attempt": int(
                source_progress.get("last_validation_attempt", 0)
            ),
            "swe_agent/last_validation_scheduled_attempt": int(
                source_progress.get(
                    "last_validation_scheduled_attempt",
                    source_progress.get("last_validation_attempt", 0),
                )
            ),
            # --- per-step drop deltas (easier to chart than diff of totals) ---
            "swe_agent/stale_dropped_groups": stale_dropped_step,
            "swe_agent/dynamic_filter_dropped_groups": filter_dropped_step,
            "swe_agent/dynamic_filter_drop_rate": filter_drop_rate_step,
            "swe_agent/infra_dropped_groups": infra_dropped_step,
            "swe_agent/groups_attempted": attempted_step,
            "swe_agent/infra_drop_rate": infra_drop_rate_step,
            **outcome_metrics,
            # Includes every valid-reward group before dynamic filtering.
            "swe_agent/reward_groups_observed": reward_groups_step,
            "swe_agent/reward_group_mean": (
                reward_mean_sum_step / reward_groups_step
                if reward_groups_step else 0.0
            ),
            "swe_agent/reward_group_variance_mean": (
                reward_variance_sum_step / reward_groups_step
                if reward_groups_step else 0.0
            ),
            "swe_agent/zero_variance_group_rate": (
                zero_variance_groups_step / reward_groups_step
                if reward_groups_step else 0.0
            ),
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
    # A valid group is not accepted for training merely because harvesting
    # placed it in a partial buffer. Record acceptance only after the complete
    # optimizer batch has passed every fail-closed check and is ready to
    # return to Slime. Validation/budget tails therefore remain pending until
    # they are either selected here or explicitly disposition-dropped.
    for group in selected_groups:
        _record_usage_disposition(
            group.usage_group_id,
            disposition="accepted",
        )
    output.metrics.update(_usage_metrics(commit_update=True))
    return output
