#!/usr/bin/env python3
"""Async rollout function backed by TrajectorySearchParallelRunner.

Drop-in for collect_grpo_rollout_async.generate_rollout — same signature,
same return type. Uses lane-based search to produce per-instance
GRPOExportBundle.policy_groups (one ExportGroup per fork-group = per
mid_cp). Atomic mode only — no streaming queue.

Selects target='policy'; Lane C is reward-only and is never trained.

ENV CONTRACT (set by slime/train_agent/run/grpo_async_lanes.py to match
the SLURM script's tunables):
  SWE_AGENT_LANES_M                  int    forks per mid_cp (default 8)
  SWE_AGENT_LANES_STEPS_PER_ROUND    int    asst turns between mid_cps (default 20)
  SWE_AGENT_LANES_STEP_LIMIT         int    hard cap per Lane B branch (default 120)
  SWE_AGENT_LANES_INSTANCE_WORKERS   int    concurrent instance subprocesses (default 8)
  SWE_AGENT_LANES_GT_EVAL_WORKERS    int    per-instance GT eval threads (default 8)
  SWE_AGENT_LANES_COMPLETION_MAX_TOKENS int per-call max_new_tokens (default 20480)
  SWE_AGENT_LANES_RUBRIC_MAX_TOKENS   int    Lane-C rubric max tokens (default 20480)
  SWE_AGENT_LANES_JUDGE_MAX_TOKENS    int    Lane-C judge max tokens (default 20480)
  SWE_AGENT_LANES_POLICY_TEMPERATURE float  Lane A sampling temp (default 1.0)
  SWE_AGENT_LANES_POLICY_TOP_P       float  default 0.95
  SWE_AGENT_LANES_LANE_B_TEMPERATURE float  Lane B sampling temp (default 1.0)
  SWE_AGENT_LANES_LANE_B_TOP_P       float  default 0.95
  SWE_AGENT_LANES_FALLBACK_PATCH_PENALTY float default 0.5
  SWE_AGENT_LANES_OUTPUT_ROOT        str    where to write per-instance run dirs
  SWE_AGENT_LANES_SKIP_NODE_IPS      str    csv of Ray node IPs/hostnames to skip
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.data_source import (
    TrainingInstanceBudgetExhausted,
    TrainingValidationBoundaryReached,
)
from slime.rollout.filter_hub.base_types import call_dynamic_filter
from slime.utils.misc import load_function
from slime.utils.types import Sample

# Reuse v0's tokenizer cache + rollout-sample assembler — they're independent
# of the search topology. build_rollout_samples consumes ExportGroup -> Sample.
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
from train_agent.collect_naive_rollout_async import (
    _emit_heartbeat,
    _ensure_usage_tracking,
    _group_outcome_metrics,
    _raise_fatal_ray_infrastructure_error,
    _record_usage_disposition,
    _restore_usage_tracking_checkpoint,
    _usage_group_prefix,
    _usage_metrics,
)
from train_agent.serving.sglang_chat_service import start_slime_policy_route_warmup
from swe_agent.exceptions import PolicyVersionMismatch
from swe_agent.policy_version import (
    checkpoint_policy_stale_lag,
    committed_policy_version,
)
from swe_agent.usage import usage_ledger_offset


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
        from swe_agent.usage import configure_usage_ledger, usage_context

        configure_usage_ledger(task.get("usage_ledger") or None)
        os.environ["SEARCH_SWE_SLIME_API_BASE"] = task["policy_base_url"]
        os.environ["SEARCH_SWE_SLIME_API_KEY"] = task["policy_api_key"]
        # Imports inside the subprocess so env-var-dependent module init runs
        # AFTER we've set the slime API base.
        from swe_agent.backend import SWEAgentRolloutBackend
        from swe_agent.lane_to_grpo_bundle import instance_record_to_bundle
        from swe_agent.run.benchmarks.swebench import (
            build_swebench_config,
            get_swebench_docker_image_name,
            get_swebench_harness_namespace,
            get_swebench_singularity_image_name,
            load_swebench_instances_by_id,
            select_container_environment_class,
        )
        from swe_agent.run.run_swe_agent import SWE_AGENT_TEXTBASED_CONFIG
        from swe_agent.run.search_swe_agent import DEFAULT_FROZEN_EXPERIENCE_BANK
        from swe_agent.rubric_bank import ExperienceRubricBank
        from swe_agent.trajectory_search_parallel import (
            ParallelSearchConfig,
            TrajectorySearchParallelRunner,
        )
        from swe_agent.utils.serialize import recursive_merge

        instance_id = task["instance_id"]
        subset = task["subset"]
        split = task["split"]
        model_name = task["model_name"]
        policy_base_url = task["policy_base_url"]
        # Per-fork-group endpoint pool. Fallback to single-URL list if the
        # task dispatcher didn't supply one (old launcher / external caller).
        policy_base_urls = task.get("policy_base_urls") or [policy_base_url]
        rubric_base_url = task["rubric_base_url"]
        policy_api_key = task["policy_api_key"]
        rubric_api_key = task["rubric_api_key"]
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
            "cwd": instance.get("swebench_workdir") or "/testbed",
        }
        if instance.get("expected_output_json"):
            environment_overrides["dataset_name"] = "r2egym"
        overrides = {
            "agent": {"step_limit": task["step_limit"]},
            "model": {
                "context_length": int(task.get("context_length", 128000)),
                "model_kwargs": {
                    "api_base": api_base,
                    "api_key": policy_api_key,
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
            raise ValueError(f"unsupported lanes reward_kind={reward_kind!r}")
        cfg = ParallelSearchConfig(
            m=task["m"],
            k=task["steps_per_round"],
            step_limit=task["step_limit"],
            gt_eval_workers=task.get("gt_eval_workers", 8),
            lane_b_pool_size=task.get("lane_b_pool_size"),
            keep_images=False,
            policy_temperature=task.get("policy_temperature", 1.0),
            policy_top_p=task.get("policy_top_p", 0.95),
            lane_b_temperature=task.get("lane_b_temperature", 1.0),
            lane_b_top_p=task.get("lane_b_top_p", 0.95),
            fallback_patch_penalty=task.get("fallback_patch_penalty", 0.5),
            no_action_patch_penalty=task.get("no_action_patch_penalty", -0.1),
            disable_rubric=task.get("disable_rubric", False),
            reward_kind=reward_kind,
            joint_alpha=task.get("joint_alpha", 1.0),
            all_pass_reward=task.get("all_pass_reward", 1.0),
            topology=task.get("topology", "depth1"),
            p=task.get("beam_parents", 2),
            terminal_rollout=task.get("terminal_rollout", False),
            rubric_max_tokens=task.get("rubric_max_tokens", 20480),
            judge_max_tokens=task.get("judge_max_tokens", 20480),
        )
        run_dir = (
            output_root / instance_id
            / time.strftime("%Y%m%d-%H%M%S")
            / f"task-{task['index']:06d}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        experience_bank = ExperienceRubricBank(
            bank_path=Path(
                task.get("experience_bank") or DEFAULT_FROZEN_EXPERIENCE_BANK
            ),
            scope="siblings",
        )
        if not experience_bank.is_frozen:
            raise ValueError(
                "Lane C requires the frozen siblings experience checkpoint "
                f"directory, got {experience_bank.bank_path}"
            )
        runner = TrajectorySearchParallelRunner(
            instance=instance,
            backend=backend,
            run_dir=run_dir,
            policy_model_name=model_name,
            policy_version=task.get("policy_version"),
            enforce_policy_version=False,
            rubric_model_name=task.get("rubric_model_name") or model_name,
            judge_model_name=task.get("judge_model_name") or model_name,
            config=cfg,
            harness_namespace=get_swebench_harness_namespace(instance),
            policy_base_url=policy_base_url,
            rubric_base_url=rubric_base_url,
            policy_api_key=policy_api_key,
            rubric_api_key=rubric_api_key,
            policy_base_urls=policy_base_urls,
            experience_banks={"siblings": experience_bank},
            usage_group_prefix=task.get("usage_group_prefix") or None,
        )
        with usage_context(
            phase=task.get("usage_phase", "train"),
            group_id=task.get("usage_group_prefix", ""),
            suppress_accounting=bool(
                task.get("suppress_usage_accounting")
            ),
        ):
            record = asyncio.run(runner.run())
        # steps_per_round is read from record.config inside the converter;
        # task["steps_per_round"] is honored as an explicit override.
        bundle = instance_record_to_bundle(
            record, steps_per_round=task.get("steps_per_round"),
            gt_only_reward=task.get("disable_rubric", False),
        )
        result = {
            "index": task["index"],
            "instance_id": instance_id,
            "bundle": bundle,
            "error": "",
            "usage_group_prefix": task.get("usage_group_prefix", ""),
            "policy_version": task.get("policy_version"),
        }
    except Exception as exc:
        result = {
            "index": task.get("index", -1),
            "instance_id": task.get("instance_id", "?"),
            "bundle": None,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            "usage_group_prefix": task.get("usage_group_prefix", ""),
            "policy_version": task.get("policy_version"),
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
    samples: list[Sample]
    rollout_id: int = 0
    group_kind: str = "root"
    usage_group_id: str = ""


_BUFFER: list[_BufferedGroup] = []
_TASK_INDEX = 0
_WARMUP_DONE = False
_STALE_DROPPED_GROUPS = 0
_STALE_DROPPED_BY_KIND: dict[str, int] = {}
_FAILED_INSTANCES = 0
_TRUNCATED_OVERSIZED = 0
_FILTER_DROPPED_GROUPS = 0
_FILTER_DROP_REASONS: dict[str, int] = {}
_FILTER_DROPPED_BY_KIND: dict[str, int] = {}
_EXCESS_DROPPED_GROUPS = 0
_EXCESS_DROPPED_BY_KIND: dict[str, int] = {}
_TOTAL_GROUPS_ATTEMPTED = 0
_GROUPS_ATTEMPTED_BY_KIND: dict[str, int] = {}
_INVALID_DROPPED_GROUPS = 0
_INVALID_DROPPED_BY_KIND: dict[str, int] = {}
_INVALID_DROP_REASONS: dict[str, int] = {}
_REWARD_GROUPS_OBSERVED_BY_KIND: dict[str, int] = {}
_REWARD_GROUP_MEAN_SUM_BY_KIND: dict[str, float] = {}
_REWARD_GROUP_VARIANCE_SUM_BY_KIND: dict[str, float] = {}
_ZERO_VARIANCE_GROUPS_BY_KIND: dict[str, int] = {}
_SOURCE_BUDGET_EXHAUSTED = False
_SOURCE_BUDGET_ERROR: TrainingInstanceBudgetExhausted | None = None
_SOURCE_VALIDATION_ERROR: TrainingValidationBoundaryReached | None = None
# Validation pauses retain extra resume metadata because the same rollout id
# must continue after ACK.  Ordinary valid over-sampling excess remains in
# _BUFFER across rollout cycles, matching the pre-change Lane collector.
_VALIDATION_PARTIAL_STATE: dict[str, Any] | None = None

_NODE_WORKERS: list[Any] = []          # list[ray.actor.ActorHandle]
_NODE_WORKER_IPS: list[str] = []
_PENDING: dict[Any, dict[str, Any]] = {}   # ObjectRef -> task dict
_REPLAY_PENDING_TASKS: list[dict[str, Any]] = []
_DISPATCH_COUNTER = 0

# Rolling estimate of groups produced per completed instance. Updated in
# _harvest_ready as bundles return; used by _submit_until_full to decide
# whether the in-flight set already covers the over-sampling group target.
# Initialized to one group per instance; the observed depth-2 yield refines it.
_OBSERVED_GROUPS_TOTAL = 0
_OBSERVED_INSTANCES = 0
_DEFAULT_EST_GROUPS = 1.0


def checkpoint_state_dict(rollout_id: int) -> dict[str, Any]:
    """Return every untrained Lane collector datum at this save point."""
    return {
        "schema_version": COLLECTOR_CHECKPOINT_SCHEMA_VERSION,
        "collector": "lanes",
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
        "counters": {
            "stale_dropped_groups": int(_STALE_DROPPED_GROUPS),
            "stale_dropped_by_kind": dict(_STALE_DROPPED_BY_KIND),
            "failed_instances": int(_FAILED_INSTANCES),
            "truncated_oversized": int(_TRUNCATED_OVERSIZED),
            "filter_dropped_groups": int(_FILTER_DROPPED_GROUPS),
            "filter_drop_reasons": dict(_FILTER_DROP_REASONS),
            "filter_dropped_by_kind": dict(
                _FILTER_DROPPED_BY_KIND
            ),
            "excess_dropped_groups": int(_EXCESS_DROPPED_GROUPS),
            "excess_dropped_by_kind": dict(
                _EXCESS_DROPPED_BY_KIND
            ),
            "total_groups_attempted": int(_TOTAL_GROUPS_ATTEMPTED),
            "groups_attempted_by_kind": dict(
                _GROUPS_ATTEMPTED_BY_KIND
            ),
            "invalid_dropped_groups": int(
                _INVALID_DROPPED_GROUPS
            ),
            "invalid_dropped_by_kind": dict(
                _INVALID_DROPPED_BY_KIND
            ),
            "invalid_drop_reasons": dict(_INVALID_DROP_REASONS),
            "reward_groups_observed_by_kind": dict(
                _REWARD_GROUPS_OBSERVED_BY_KIND
            ),
            "reward_group_mean_sum_by_kind": dict(
                _REWARD_GROUP_MEAN_SUM_BY_KIND
            ),
            "reward_group_variance_sum_by_kind": dict(
                _REWARD_GROUP_VARIANCE_SUM_BY_KIND
            ),
            "zero_variance_groups_by_kind": dict(
                _ZERO_VARIANCE_GROUPS_BY_KIND
            ),
            "observed_groups_total": int(_OBSERVED_GROUPS_TOTAL),
            "observed_instances": int(_OBSERVED_INSTANCES),
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
    """Restore logical state and defer live task rebinding to generate()."""
    global _BUFFER, _TASK_INDEX, _WARMUP_DONE
    global _STALE_DROPPED_GROUPS, _STALE_DROPPED_BY_KIND
    global _FAILED_INSTANCES, _TRUNCATED_OVERSIZED
    global _FILTER_DROPPED_GROUPS, _FILTER_DROP_REASONS
    global _FILTER_DROPPED_BY_KIND, _EXCESS_DROPPED_GROUPS
    global _EXCESS_DROPPED_BY_KIND, _TOTAL_GROUPS_ATTEMPTED
    global _GROUPS_ATTEMPTED_BY_KIND, _INVALID_DROPPED_GROUPS
    global _INVALID_DROPPED_BY_KIND, _INVALID_DROP_REASONS
    global _REWARD_GROUPS_OBSERVED_BY_KIND
    global _REWARD_GROUP_MEAN_SUM_BY_KIND
    global _REWARD_GROUP_VARIANCE_SUM_BY_KIND
    global _ZERO_VARIANCE_GROUPS_BY_KIND
    global _SOURCE_BUDGET_EXHAUSTED, _SOURCE_BUDGET_ERROR
    global _SOURCE_VALIDATION_ERROR, _VALIDATION_PARTIAL_STATE
    global _NODE_WORKERS, _NODE_WORKER_IPS, _PENDING
    global _REPLAY_PENDING_TASKS, _DISPATCH_COUNTER
    global _OBSERVED_GROUPS_TOTAL, _OBSERVED_INSTANCES

    if int(state.get("schema_version", -1)) != (
        COLLECTOR_CHECKPOINT_SCHEMA_VERSION
    ):
        raise RuntimeError(
            "unsupported Lane collector checkpoint schema: "
            f"{state.get('schema_version')!r}"
        )
    if state.get("collector") != "lanes":
        raise RuntimeError(
            "Lane collector cannot restore state for "
            f"{state.get('collector')!r}"
        )
    if int(state.get("checkpoint_rollout_id", -1)) != int(rollout_id):
        raise RuntimeError(
            "Lane collector checkpoint id mismatch: "
            f"requested={rollout_id} "
            f"saved={state.get('checkpoint_rollout_id')}"
        )
    if _BUFFER or _PENDING or _REPLAY_PENDING_TASKS:
        raise RuntimeError(
            "refusing to overlay a Lane collector checkpoint on live state"
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
    _STALE_DROPPED_BY_KIND = dict(
        counters.get("stale_dropped_by_kind") or {}
    )
    _FAILED_INSTANCES = int(counters.get("failed_instances", 0))
    _TRUNCATED_OVERSIZED = int(counters.get("truncated_oversized", 0))
    _FILTER_DROPPED_GROUPS = int(
        counters.get("filter_dropped_groups", 0)
    )
    _FILTER_DROP_REASONS = dict(
        counters.get("filter_drop_reasons") or {}
    )
    _FILTER_DROPPED_BY_KIND = dict(
        counters.get("filter_dropped_by_kind") or {}
    )
    _EXCESS_DROPPED_GROUPS = int(
        counters.get("excess_dropped_groups", 0)
    )
    _EXCESS_DROPPED_BY_KIND = dict(
        counters.get("excess_dropped_by_kind") or {}
    )
    _TOTAL_GROUPS_ATTEMPTED = int(
        counters.get("total_groups_attempted", 0)
    )
    _GROUPS_ATTEMPTED_BY_KIND = dict(
        counters.get("groups_attempted_by_kind") or {}
    )
    _INVALID_DROPPED_GROUPS = int(
        counters.get("invalid_dropped_groups", 0)
    )
    _INVALID_DROPPED_BY_KIND = dict(
        counters.get("invalid_dropped_by_kind") or {}
    )
    _INVALID_DROP_REASONS = dict(
        counters.get("invalid_drop_reasons") or {}
    )
    _REWARD_GROUPS_OBSERVED_BY_KIND = dict(
        counters.get("reward_groups_observed_by_kind") or {}
    )
    _REWARD_GROUP_MEAN_SUM_BY_KIND = dict(
        counters.get("reward_group_mean_sum_by_kind") or {}
    )
    _REWARD_GROUP_VARIANCE_SUM_BY_KIND = dict(
        counters.get("reward_group_variance_sum_by_kind") or {}
    )
    _ZERO_VARIANCE_GROUPS_BY_KIND = dict(
        counters.get("zero_variance_groups_by_kind") or {}
    )
    _OBSERVED_GROUPS_TOTAL = int(
        counters.get("observed_groups_total", 0)
    )
    _OBSERVED_INSTANCES = int(counters.get("observed_instances", 0))
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

    _WARMUP_DONE = False
    _NODE_WORKERS = []
    _NODE_WORKER_IPS = []
    _ENDPOINT_LOAD.clear()


def _expected_usage_group_specs(task: dict[str, Any]) -> dict[str, str]:
    """Map each expected usage id to its conserved root/beam group kind."""
    prefix = str(task.get("usage_group_prefix") or "")
    if not prefix:
        return {}
    specs = {f"{prefix}:g0": "root"}
    if task.get("topology") == "depth2":
        specs[f"{prefix}:g1"] = "beam"
    return specs


def _increment_kind_counter(counter: dict[str, int], kind: str) -> None:
    counter[kind] = counter.get(kind, 0) + 1


def _mark_expected_attempt(kind: str) -> None:
    global _TOTAL_GROUPS_ATTEMPTED
    _TOTAL_GROUPS_ATTEMPTED += 1
    _increment_kind_counter(_GROUPS_ATTEMPTED_BY_KIND, kind)


def _mark_expected_stale(kind: str) -> None:
    global _STALE_DROPPED_GROUPS
    _STALE_DROPPED_GROUPS += 1
    _increment_kind_counter(_STALE_DROPPED_BY_KIND, kind)


def _mark_expected_drop(kind: str, reason: str, *, filtered: bool) -> None:
    global _FILTER_DROPPED_GROUPS, _INVALID_DROPPED_GROUPS
    if filtered:
        _FILTER_DROPPED_GROUPS += 1
        _increment_kind_counter(_FILTER_DROPPED_BY_KIND, kind)
        key = str(reason or "unspecified")
        _FILTER_DROP_REASONS[key] = _FILTER_DROP_REASONS.get(key, 0) + 1
        return
    _INVALID_DROPPED_GROUPS += 1
    _increment_kind_counter(_INVALID_DROPPED_BY_KIND, kind)
    key = str(reason or "unspecified").split(" ", 1)[0]
    _INVALID_DROP_REASONS[key] = _INVALID_DROP_REASONS.get(key, 0) + 1


def _observe_reward_group(args, samples: list[Sample], *, kind: str) -> None:
    """Accumulate reward statistics before the dynamic filter can drop them."""
    rewards = [float(sample.get_reward_value(args)) for sample in samples]
    if not rewards:
        raise ValueError("cannot observe reward statistics for an empty group")
    mean = sum(rewards) / len(rewards)
    variance = (
        sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
        if len(rewards) > 1
        else 0.0
    )
    _REWARD_GROUPS_OBSERVED_BY_KIND[kind] = (
        _REWARD_GROUPS_OBSERVED_BY_KIND.get(kind, 0) + 1
    )
    _REWARD_GROUP_MEAN_SUM_BY_KIND[kind] = (
        _REWARD_GROUP_MEAN_SUM_BY_KIND.get(kind, 0.0) + mean
    )
    _REWARD_GROUP_VARIANCE_SUM_BY_KIND[kind] = (
        _REWARD_GROUP_VARIANCE_SUM_BY_KIND.get(kind, 0.0) + variance
    )
    if variance <= 1e-12:
        _ZERO_VARIANCE_GROUPS_BY_KIND[kind] = (
            _ZERO_VARIANCE_GROUPS_BY_KIND.get(kind, 0) + 1
        )


def _record_task_disposition(
    task: dict[str, Any],
    *,
    disposition: str,
    reason: str,
) -> None:
    for group_id in _expected_usage_group_specs(task):
        _record_usage_disposition(
            group_id,
            disposition=disposition,
            reason=reason,
        )


@ray.remote(num_cpus=2)
class _LanesNodeWorker:
    """Per-physical-node executor for v1 lane bundle tasks.

    Identical pattern to v0's _PDSNodeWorker but atomic-only."""

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
    skip = {
        s.strip()
        for s in os.environ.get("SWE_AGENT_LANES_SKIP_NODE_IPS", "").split(",")
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
                f"[lanes-async] SKIP node {node_name or node_ip} "
                f"(matched SWE_AGENT_LANES_SKIP_NODE_IPS)"
            )
            continue
        nodes.append(n)
    if not nodes:
        raise RuntimeError("No GPU-bearing Ray nodes found for Lanes worker spawn")
    actor_concurrency = per_node_concurrency + 4
    logger.info(
        f"[lanes-async] spawning {len(nodes)} node workers "
        f"(skipped={len(skip)}), "
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

    def _bool(name, default):
        v = os.environ.get(name, "")
        if not v:
            return default
        return v.strip().lower() in ("1", "true", "yes", "on")

    return {
        "m": _int("SWE_AGENT_LANES_M", 8),
        "steps_per_round": _int("SWE_AGENT_LANES_STEPS_PER_ROUND", 20),
        "step_limit": _int("SWE_AGENT_LANES_STEP_LIMIT", 120),
        "gt_eval_workers": _int("SWE_AGENT_LANES_GT_EVAL_WORKERS", 8),
        "lane_b_pool_size": _int("SWE_AGENT_LANES_LANE_B_POOL_SIZE"),
        "completion_max_tokens": _int("SWE_AGENT_LANES_COMPLETION_MAX_TOKENS", 20480),
        "context_length": _int("SWE_AGENT_MODEL_CONTEXT_LENGTH", 128000),
        "rubric_max_tokens": _int("SWE_AGENT_LANES_RUBRIC_MAX_TOKENS", 20480),
        "judge_max_tokens": _int("SWE_AGENT_LANES_JUDGE_MAX_TOKENS", 20480),
        "policy_temperature": _float("SWE_AGENT_LANES_POLICY_TEMPERATURE", 1.0),
        "policy_top_p": _float("SWE_AGENT_LANES_POLICY_TOP_P", 0.95),
        "lane_b_temperature": _float("SWE_AGENT_LANES_LANE_B_TEMPERATURE", 1.0),
        "lane_b_top_p": _float("SWE_AGENT_LANES_LANE_B_TOP_P", 0.95),
        "fallback_patch_penalty": _float("SWE_AGENT_LANES_FALLBACK_PATCH_PENALTY", 0.5),
        "no_action_patch_penalty": _float("SWE_AGENT_LANES_NO_ACTION_PATCH_PENALTY", -0.1),
        # GT-only training: skip rubric/judge in trajectory_search_parallel
        # and use branch.gt_score as the reward in the bundler.
        "disable_rubric": _bool("SWE_AGENT_LANES_DISABLE_RUBRIC", False),
        # Shared evaluator reward kind, identical to naive in GT-only mode.
        "reward_kind": (os.environ.get("SWE_AGENT_LANES_REWARD_KIND") or "joint").lower(),
        "joint_alpha": _float("SWE_AGENT_LANES_JOINT_ALPHA", 1.0),
        "all_pass_reward": _float("SWE_AGENT_LANES_ALL_PASS_REWARD", 1.0),
        "topology": (
            os.environ.get("SWE_AGENT_LANES_TOPOLOGY") or "depth1"
        ).strip().lower(),
        "beam_parents": _int("SWE_AGENT_LANES_BEAM_PARENTS", 2),
        # Training must stop at the judged prefix. Validation uses the
        # policy-only naive evaluator path below and never enables Lane C.
        "terminal_rollout": _bool("SWE_AGENT_LANES_TERMINAL_ROLLOUT", False),
    }


# Per-endpoint in-flight counters for least-loaded routing. Lock-protected
# because _submit_until_full runs on the rollout coordinator thread but
# completion (which releases the counter) happens in _harvest_ready —
# they may interleave with future async tweaks.
_ENDPOINT_LOAD: dict[str, int] = {}
_ENDPOINT_LOAD_LOCK = threading.Lock()


def _hosted_lane_c_settings() -> tuple[str, str, str, str]:
    """Return the single hosted Lane-C route, never a local policy URL."""
    default_lane_c_model = "openai/azure/openai/gpt-5.6-luna"
    rubric_model = (
        os.environ.get("SWE_AGENT_LANES_RUBRIC_MODEL")
        or default_lane_c_model
    ).strip()
    judge_model = (
        os.environ.get("SWE_AGENT_LANES_JUDGE_MODEL")
        or default_lane_c_model
    ).strip()
    allowed_model_routes = {
        # GLM remains an explicit experiment route alongside Luna.
        "nvidia/zai-org/glm-5.2",
        "openai/azure/zai-org/glm-5.2",
        # The outer openai/ selects LiteLLM's OpenAI-compatible adapter; the
        # NVIDIA gateway receives azure/openai/gpt-5.6-luna as the model id.
        "openai/azure/openai/gpt-5.6-luna",
    }
    if (
        rubric_model != judge_model
        or rubric_model not in allowed_model_routes
    ):
        raise ValueError(
            "Current early-prediction experiments require one identical "
            "approved NVIDIA-gateway model for every Lane C call; "
            f"got rubric={rubric_model!r}, judge={judge_model!r}"
        )
    api_base = (
        os.environ.get("SWE_AGENT_LANES_RUBRIC_API_BASE")
        or os.environ.get("LITELLM_API_BASE")
        or os.environ.get("NVIDIA_API_BASE")
        or "https://inference-api.nvidia.com/v1"
    ).strip().rstrip("/")
    if api_base.endswith("/chat/completions"):
        api_base = api_base[: -len("/chat/completions")]
    lowered = api_base.lower()
    if not lowered.startswith("https://") or any(
        local in lowered for local in ("127.0.0.1", "localhost", "0.0.0.0")
    ):
        raise ValueError(
            "Lane C requires the hosted NVIDIA HTTPS endpoint, "
            f"got {api_base!r}"
        )
    api_key = next(
        (
            value.strip()
            for name in (
                "SWE_AGENT_LANES_RUBRIC_API_KEY",
                "LITELLM_API_KEY",
                "NVIDIA_API_KEY",
                "NVIDIA_NIM_API_KEY",
            )
            if (value := os.environ.get(name)) and value.strip()
        ),
        "",
    )
    if not api_key:
        raise ValueError(
            "Lane C hosted-model API key is missing; export "
            "LITELLM_API_KEY or NVIDIA_API_KEY from exp/config.md"
        )
    return rubric_model, judge_model, api_base, api_key


def _policy_urls_for_model(args, model_name: str) -> list[str]:
    api_host = os.environ.get("SWE_AGENT_LANES_API_HOST", "http://127.0.0.1")
    explicit_urls = [
        f"{api_host.rstrip('/')}:{int(port)}"
        for port in os.environ.get(
            "SWE_AGENT_LANES_POLICY_PORTS",
            "",
        ).split(",")
        if port
    ]
    if explicit_urls:
        return explicit_urls

    engines_map = getattr(args, "sglang_model_engines", None) or {}
    endpoints = list(engines_map.get(model_name, []))
    if not endpoints and len(engines_map) == 1:
        endpoints = list(next(iter(engines_map.values())))
    if not endpoints:
        raise RuntimeError(
            f"No per-engine SGLang endpoints found for model={model_name!r}; "
            "lane async requires args.sglang_model_engines or explicit "
            "SWE_AGENT_LANES_POLICY_PORTS."
        )
    return [f"http://{host}:{int(port)}" for host, port in endpoints]


def _pick_least_loaded_urls(candidates: list[str], n: int) -> list[str]:
    """Pick n endpoint URLs by least-loaded with pre-increment."""
    picks: list[str] = []
    with _ENDPOINT_LOAD_LOCK:
        for c in candidates:
            _ENDPOINT_LOAD.setdefault(c, 0)
        for _ in range(max(1, n)):
            url = min(candidates, key=lambda c: _ENDPOINT_LOAD[c])
            _ENDPOINT_LOAD[url] += 1
            picks.append(url)
    return picks


def _release_endpoints(urls: list[str]) -> None:
    with _ENDPOINT_LOAD_LOCK:
        for url in urls:
            if url in _ENDPOINT_LOAD:
                _ENDPOINT_LOAD[url] = max(0, _ENDPOINT_LOAD[url] - 1)


def _dispatch_policy_version() -> str | None:
    """Return the committed weight epoch, pausing dispatch during an update."""
    try:
        return committed_policy_version()
    except PolicyVersionMismatch:
        return None


def _replay_checkpoint_pending_tasks(
    *,
    args,
    rollout_id: int,
    output_root: Path,
    model_name: str,
) -> int:
    """Rebind checkpointed pending instances to live policy/Lane-C routes."""
    global _REPLAY_PENDING_TASKS, _DISPATCH_COUNTER
    if not _REPLAY_PENDING_TASKS:
        return 0
    if not _NODE_WORKERS:
        raise RuntimeError(
            "cannot replay Lane pending tasks before node workers exist"
        )
    policy_version = _dispatch_policy_version()
    if policy_version is None:
        raise RuntimeError(
            "cannot replay Lane pending tasks during a policy transition"
        )
    policy_urls = _policy_urls_for_model(args, model_name)
    rubric_model, judge_model, rubric_base_url, rubric_api_key = (
        _hosted_lane_c_settings()
    )
    policy_api_key = os.environ.get(
        "SEARCH_SWE_SLIME_API_KEY",
        "EMPTY",
    )
    replay_tasks = _REPLAY_PENDING_TASKS
    _REPLAY_PENDING_TASKS = []
    for original in replay_tasks:
        task = copy.deepcopy(original)
        _record_task_disposition(
            task,
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
        n_urls = (
            1 + int(task.get("beam_parents", 2))
            if task.get("topology") == "depth2"
            else 1
        )
        policy_base_urls = _pick_least_loaded_urls(policy_urls, n=n_urls)
        task.update(
            {
                "rollout_id": int(rollout_id),
                "output_root": str(
                    output_root / f"rollout_{int(rollout_id):04d}"
                ),
                "model_name": model_name,
                "rubric_model_name": rubric_model,
                "judge_model_name": judge_model,
                "policy_base_url": policy_base_urls[0],
                "policy_base_urls": policy_base_urls,
                "_pinned_endpoints": list(policy_base_urls),
                "rubric_base_url": rubric_base_url,
                "policy_api_key": policy_api_key,
                "rubric_api_key": rubric_api_key,
                "experience_bank": os.environ.get(
                    "SWE_AGENT_LANES_EXPERIENCE_BANK",
                    str(task.get("experience_bank") or ""),
                ),
                "usage_ledger": _ensure_usage_tracking(),
                # The durable prefix already contains one approximate cost
                # for this pending source instance. Do not charge its
                # checkpoint re-execution a second time.
                "suppress_usage_accounting": True,
                "policy_version": policy_version,
                "source_group_index": source_group_index,
                "usage_group_prefix": usage_prefix,
            }
        )
        task.pop("usage_group_id", None)
        worker = _NODE_WORKERS[
            _DISPATCH_COUNTER % len(_NODE_WORKERS)
        ]
        _DISPATCH_COUNTER += 1
        ref = worker.submit_task.remote(task)
        _PENDING[ref] = task
    logger.info(
        "[lanes-async] replayed %d checkpointed pending source instances "
        "under policy_version=%s rollout_id=%d",
        len(replay_tasks),
        policy_version,
        rollout_id,
    )
    return len(replay_tasks)


def _submit_until_full(
    *,
    args,
    data_buffer,
    rollout_id: int,
    output_root: Path,
    model_name: str,
    max_pending: int,
    over_sampling_groups: int,
    est_groups_per_instance: float,
) -> int:
    """Dispatch instance tasks subject to two caps:
      - max_pending: hard concurrency cap on in-flight Ray Futures
      - over_sampling_groups: stop dispatching once
        (buffer + pending * est_groups_per_instance) reaches this many groups

    The second cap keeps depth-2's variable one-or-two group yield from
    dispatching avoidable extra instances while preserving FIFO ordering.
    """
    global _TASK_INDEX, _DISPATCH_COUNTER
    global _SOURCE_BUDGET_EXHAUSTED, _SOURCE_BUDGET_ERROR
    global _SOURCE_VALIDATION_ERROR
    submitted = 0
    lanes_values = _lanes_values_from_env()
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
    policy_urls = _policy_urls_for_model(args, model_name)
    rubric_model, judge_model, rubric_base_url, rubric_api_key = (
        _hosted_lane_c_settings()
    )
    policy_api_key = os.environ.get("SEARCH_SWE_SLIME_API_KEY", "EMPTY")
    # Root rollout uses one endpoint. Depth-2 reserves one additional endpoint
    # per independent Lane-A parent for its possible continuation group.
    n_urls_per_instance = (
        1 + int(lanes_values["beam_parents"])
        if lanes_values["topology"] == "depth2"
        else 1
    )

    while True:
        if len(_PENDING) >= max_pending:
            break
        # Stop dispatching once buffer + in-flight estimates cover the
        # over-sampling target. This preserves the pre-change FIFO collector
        # semantics; depth2 root and beam groups are not reweighted after
        # invalid/zero-variance filtering.
        est_pending_groups = len(_PENDING) * est_groups_per_instance
        if len(_BUFFER) + est_pending_groups >= over_sampling_groups:
            break
        if _SOURCE_BUDGET_EXHAUSTED or _SOURCE_VALIDATION_ERROR is not None:
            break
        policy_version = _dispatch_policy_version()
        if policy_version is None:
            break
        try:
            prompt_groups = data_buffer.get_samples(1)
        except TrainingValidationBoundaryReached as exc:
            _SOURCE_VALIDATION_ERROR = exc
            logger.info("[lanes-async] %s", exc)
            break
        except TrainingInstanceBudgetExhausted as exc:
            _SOURCE_BUDGET_EXHAUSTED = True
            _SOURCE_BUDGET_ERROR = exc
            logger.info("[lanes-async] %s", exc)
            break
        if not prompt_groups:
            break
        (prompt_group,) = prompt_groups
        # slime expands n_samples_per_prompt by duplicating the prompt; the
        # runner itself creates the M=8 sibling group, so we just take the
        # first duplicate. n_samples_per_prompt MUST equal SWE_AGENT_LANES_M.
        metadata = prompt_group[0].metadata
        instance_id = metadata["instance_id"]
        # Per-fork-group endpoint pool. Pick N URLs by least-loaded with
        # pre-increment, so an instance whose runner
        # eventually spawns N fork groups gets N distinct (or repeated, when
        # ports < N) endpoints. The runner round-robins groups across this
        # pool. All picks are released together in _harvest_ready.
        policy_base_urls = _pick_least_loaded_urls(policy_urls, n=n_urls_per_instance)
        policy_base_url = policy_base_urls[0]
        usage_prefix = _usage_group_prefix(
            phase="train",
            rollout_id=rollout_id,
            instance_id=instance_id,
            # Use the checkpointed source-attempt cursor for ledger identity.
            # _TASK_INDEX is process-local and resets after a requeue.
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
            "rubric_model_name": rubric_model,
            "judge_model_name": judge_model,
            "policy_base_url": policy_base_url,
            "policy_base_urls": policy_base_urls,
            "rubric_base_url": rubric_base_url,
            "policy_api_key": policy_api_key,
            "rubric_api_key": rubric_api_key,
            "experience_bank": os.environ.get(
                "SWE_AGENT_LANES_EXPERIENCE_BANK", ""
            ),
            "usage_ledger": _ensure_usage_tracking(),
            "usage_phase": "train",
            "usage_group_prefix": usage_prefix,
            "source_group_index": int(
                prompt_group[0].group_index
            ),
            "policy_version": policy_version,
            **lanes_values,
        }
        # Stash for release on completion (every picked URL, even repeats).
        task["_pinned_endpoints"] = list(policy_base_urls)
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
    global _FILTER_DROPPED_BY_KIND
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
        expected_specs = _expected_usage_group_specs(task)
        expected_usage_ids = set(expected_specs)
        accounted_usage_ids: set[str] = set()

        def mark_invalid(usage_group_id: str, reason: str) -> None:
            if (
                usage_group_id not in expected_specs
                or usage_group_id in accounted_usage_ids
            ):
                return
            accounted_usage_ids.add(usage_group_id)
            _mark_expected_drop(
                expected_specs[usage_group_id],
                reason,
                filtered=False,
            )

        def mark_filtered(usage_group_id: str, reason: str) -> None:
            if (
                usage_group_id not in expected_specs
                or usage_group_id in accounted_usage_ids
            ):
                return
            accounted_usage_ids.add(usage_group_id)
            _mark_expected_drop(
                expected_specs[usage_group_id],
                reason,
                filtered=True,
            )

        # Release this task's pinned endpoint(s) from the least-loaded
        # counter — must happen regardless of success/error so the
        # counter stays correct.
        _release_endpoints(list(task.get("_pinned_endpoints", [])))
        if policy_stale_lag is not None and policy_stale_lag < 0:
            raise RuntimeError(
                "Lane collector observed policy weights from the future: "
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
            try:
                ray.get(ref)
            except Exception as exc:
                _raise_fatal_ray_infrastructure_error(
                    exc,
                    collector="lanes",
                    instance_id=task.get("instance_id"),
                )
            for expected_kind in expected_specs.values():
                _mark_expected_stale(expected_kind)
            _record_task_disposition(
                task,
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
                collector="lanes",
                instance_id=task.get("instance_id"),
            )
            for expected_kind in expected_specs.values():
                _mark_expected_attempt(expected_kind)
            _FAILED_INSTANCES += 1
            for expected_group_id in expected_usage_ids:
                mark_invalid(expected_group_id, "ray_task_error")
            _record_task_disposition(
                task,
                disposition="invalid",
                reason="ray_task_error",
            )
            logger.warning(
                "[lanes-async] instance %s failed: %s",
                task.get("instance_id"), str(exc)[:200],
            )
            del ref
            continue
        # Returned application results count as attempts; node/worker loss is
        # recovery overhead and must not advance the experiment denominator.
        for expected_kind in expected_specs.values():
            _mark_expected_attempt(expected_kind)
        if result.get("error"):
            _FAILED_INSTANCES += 1
            invalid_reason = (
                "policy_version_mismatch"
                if "PolicyVersionMismatch" in str(result["error"])
                else "runner_error"
            )
            for expected_group_id in expected_usage_ids:
                mark_invalid(expected_group_id, invalid_reason)
            _record_task_disposition(
                task,
                disposition="invalid",
                reason=invalid_reason,
            )
            logger.warning(
                "[lanes-async] instance %s failed: %s",
                result.get("instance_id"), result["error"][:200],
            )
            del result, ref
            continue
        if result.get("policy_version") != task.get("policy_version"):
            for expected_group_id in expected_usage_ids:
                mark_invalid(
                    expected_group_id,
                    "policy_version_metadata_mismatch",
                )
            _record_task_disposition(
                task,
                disposition="invalid",
                reason="policy_version_metadata_mismatch",
            )
            del result, ref
            continue
        bundle = result.get("bundle")
        if bundle is None:
            for expected_group_id in expected_usage_ids:
                mark_invalid(expected_group_id, "missing_bundle")
            _record_task_disposition(
                task,
                disposition="invalid",
                reason="missing_bundle",
            )
            del result, ref
            continue
        # Target selection: v1 first pass only generates policy_groups.
        groups = (
            bundle.policy_groups if target == "policy" else bundle.rubric_groups
        )
        bundle_metadata = getattr(bundle, "metadata", None) or {}
        skipped_group_reasons = dict(
            bundle_metadata.get("skipped_group_reasons") or {}
        )
        # Refine the rolling groups-per-instance estimate from every
        # successfully harvested bundle.
        global _OBSERVED_GROUPS_TOTAL, _OBSERVED_INSTANCES
        _OBSERVED_GROUPS_TOTAL += len(groups)
        _OBSERVED_INSTANCES += 1
        resolved_groups: list[tuple[Any, str, dict[str, Any]]] = []
        usage_id_counts: dict[str, int] = {}
        for group in groups:
            metadata = getattr(group, "metadata", None) or {}
            try:
                group_index = int(metadata["group_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "lane ExportGroup for "
                    f"{task.get('instance_id')} is missing a valid "
                    "metadata.group_index"
                ) from exc
            usage_group_id = (
                f"{task.get('usage_group_prefix', '')}:g{group_index}"
            )
            resolved_groups.append((group, usage_group_id, metadata))
            usage_id_counts[usage_group_id] = (
                usage_id_counts.get(usage_group_id, 0) + 1
            )
        for group, usage_group_id, metadata in resolved_groups:
            if usage_id_counts[usage_group_id] > 1:
                raise RuntimeError(
                    "Lane runner returned duplicate GRPO groups for "
                    f"{usage_group_id}"
                )
            if usage_group_id not in expected_usage_ids:
                raise RuntimeError(
                    "Lane runner returned an unexpected GRPO group: "
                    f"{usage_group_id}; expected={sorted(expected_usage_ids)}"
                )
            group_kind = str(metadata.get("group_kind", "root")).strip().lower()
            if group_kind not in {"root", "beam"}:
                raise RuntimeError(
                    f"Lane runner returned invalid group_kind={group_kind!r} "
                    f"for {usage_group_id}"
                )
            group_index = int(metadata["group_index"])
            expected_kind = "root" if group_index == 0 else "beam"
            if group_kind != expected_kind:
                raise RuntimeError(
                    "Lane group kind/index invariant failed for "
                    f"{usage_group_id}: kind={group_kind} index={group_index}"
                )
            samples, truncated_here = build_rollout_samples(
                groups=[group],
                include_turn_rewards=False,
                max_sample_tokens=max_sample_tokens,
            )
            _TRUNCATED_OVERSIZED += truncated_here
            if len(samples) != int(task["m"]):
                mark_invalid(usage_group_id, "wrong_sample_count")
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
            _observe_reward_group(args, samples, kind=group_kind)
            if dynamic_filter is not None and samples:
                filter_out = call_dynamic_filter(dynamic_filter, args, samples)
                if not filter_out.keep:
                    reason = filter_out.reason or "unspecified"
                    mark_filtered(usage_group_id, reason)
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
            accounted_usage_ids.add(usage_group_id)
            _BUFFER.append(
                _BufferedGroup(
                    samples=samples,
                    rollout_id=source_rollout_id,
                    group_kind=group_kind,
                    usage_group_id=usage_group_id,
                )
            )
            harvested += 1
        for raw_group_index, raw_reason in skipped_group_reasons.items():
            try:
                skipped_group_index = int(raw_group_index)
            except (TypeError, ValueError):
                raise RuntimeError(
                    "Lane runner returned an invalid skipped group index: "
                    f"{raw_group_index!r}"
                )
            skipped_usage_id = (
                f"{task.get('usage_group_prefix', '')}:g{skipped_group_index}"
            )
            if skipped_usage_id not in expected_usage_ids:
                raise RuntimeError(
                    "Lane runner returned an unexpected skipped group: "
                    f"{skipped_usage_id}; expected={sorted(expected_usage_ids)}"
                )
            if skipped_usage_id in accounted_usage_ids:
                raise RuntimeError(
                    "Lane runner both exported and skipped the same group: "
                    f"{skipped_usage_id}"
                )
            reason = f"topology_skip:{str(raw_reason or 'unspecified')}"
            mark_filtered(skipped_usage_id, reason)
            _record_usage_disposition(
                skipped_usage_id,
                disposition="filtered",
                reason=reason,
            )
        for missing_group_id in expected_usage_ids - accounted_usage_ids:
            mark_invalid(missing_group_id, "missing_group")
            _record_usage_disposition(
                missing_group_id,
                disposition="invalid",
                reason="missing_group",
            )
        del result, bundle, groups, ref
    return harvested


def _buffer_kind_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for group in _BUFFER:
        counts[group.group_kind] = counts.get(group.group_kind, 0) + 1
    return counts


def _drop_stale_buffer(current_rollout_id: int) -> None:
    kept: list[_BufferedGroup] = []
    for group in _BUFFER:
        policy_versions = {
            str((sample.metadata or {}).get("policy_version") or "")
            for sample in group.samples
        }
        if len(policy_versions) > 1:
            raise RuntimeError(
                "buffered Lane group mixes policy versions: "
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
                "buffered Lane group has policy weights from the future: "
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
        _mark_expected_stale(group.group_kind)
        _record_usage_disposition(
            group.usage_group_id,
            disposition="dropped",
            reason="stale",
        )
    _BUFFER[:] = kept


def _pop_groups(group_count: int) -> list[_BufferedGroup]:
    groups = _BUFFER[:group_count]
    del _BUFFER[:group_count]
    return groups


def _discard_excess_buffer(*, reason: str = "excess_after_quota") -> int:
    """Drop a terminal partial batch that can no longer form an update."""
    global _EXCESS_DROPPED_GROUPS
    excess_groups = list(_BUFFER)
    _BUFFER.clear()
    for group in excess_groups:
        _EXCESS_DROPPED_GROUPS += 1
        _increment_kind_counter(
            _EXCESS_DROPPED_BY_KIND,
            group.group_kind,
        )
        _record_usage_disposition(
            group.usage_group_id,
            disposition="dropped",
            reason=reason,
        )
    return len(excess_groups)


def _pending_validation_boundary(
    data_buffer,
) -> TrainingValidationBoundaryReached | None:
    """Return the source boundary that must run before consuming carry-over.

    Lane over-sampling intentionally carries valid excess groups into the
    following optimizer batch.  A quota-ready carry buffer can therefore
    bypass ``get_samples()``, which is where RolloutDataSource normally raises
    its validation control signal.  Read the durable source cursor directly so
    validation still occurs exactly at the configured attempted-instance axis.
    """
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
    """Validate and return attempt-boundary resume state when one exists."""
    state = _VALIDATION_PARTIAL_STATE
    if state is None:
        # Ordinary valid excess and in-flight work carry into the next
        # rollout cycle; stale=1 bounds their lifetime.
        return None
    if int(state["rollout_id"]) != int(rollout_id):
        raise RuntimeError(
            "Lane validation partial must retry the same rollout id: "
            f"preserved={state['rollout_id']} requested={rollout_id}"
        )
    boundary = int(state["boundary"])
    source_boundary = int(
        getattr(_SOURCE_VALIDATION_ERROR, "boundary", 0) or 0
    )
    if source_boundary != boundary:
        raise RuntimeError(
            "Lane validation partial lost its source-boundary guard: "
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
            "Lane validation partial cannot resume before validation is "
            f"scheduled: scheduled={scheduled} boundary={boundary}"
        )
    current_kinds = _buffer_kind_counts()
    checkpoint_pending_count = (
        len(_PENDING) + len(_REPLAY_PENDING_TASKS)
    )
    if (
        int(state["group_count"]) != len(_BUFFER)
        or dict(state["group_kinds"]) != current_kinds
        or int(state["pending_count"]) != checkpoint_pending_count
    ):
        raise RuntimeError(
            "Lane validation partial changed while paused: "
            f"groups={state['group_kinds']}->{current_kinds} "
            f"pending={state['pending_count']}->{checkpoint_pending_count}"
        )
    return state


# ---------------------------------------------------------------------------
# Slime rollout entry
# ---------------------------------------------------------------------------


def generate_rollout(args, rollout_id: int, data_buffer, evaluation: bool = False):
    """Slime rollout entry — called once per training rollout cycle."""
    global _WARMUP_DONE, _EXCESS_DROPPED_GROUPS
    global _SOURCE_VALIDATION_ERROR, _VALIDATION_PARTIAL_STATE
    output_root = Path(
        os.environ.get(
            "SWE_AGENT_LANES_OUTPUT_ROOT",
            "/workspace/rler/agent/outputs/lane_outputs/train_async_lanes",
        )
    )
    model_name = os.environ.get("SWE_AGENT_GRPO_MODEL_NAME") or "Qwen/Qwen3.5-9B"
    if evaluation:
        # Validation deliberately bypasses Lane C: one root-to-terminal policy
        # rollout per manifest instance followed by the binary GT evaluator.
        from train_agent.collect_naive_rollout_async import (
            generate_validation_rollout,
        )

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
        attempted_by_kind_at_start = dict(
            resume_state["attempted_by_kind_at_start"]
        )
        invalid_at_start = int(resume_state["invalid_at_start"])
        invalid_by_kind_at_start = dict(
            resume_state["invalid_by_kind_at_start"]
        )
        filtered_at_start = int(resume_state["filtered_at_start"])
        filtered_by_kind_at_start = dict(
            resume_state["filtered_by_kind_at_start"]
        )
        reward_count_at_start = dict(
            resume_state["reward_count_at_start"]
        )
        reward_mean_sum_at_start = dict(
            resume_state["reward_mean_sum_at_start"]
        )
        reward_variance_sum_at_start = dict(
            resume_state["reward_variance_sum_at_start"]
        )
        zero_variance_at_start = dict(
            resume_state["zero_variance_at_start"]
        )
        stale_at_start = int(resume_state["stale_at_start"])
        stale_by_kind_at_start = dict(
            resume_state["stale_by_kind_at_start"]
        )
        carried_in = int(resume_state["carried_in"])
        carried_in_by_kind = dict(resume_state["carried_in_by_kind"])
    else:
        carried_in = len(_BUFFER)
        carried_in_by_kind = _buffer_kind_counts()
        attempted_at_start = _TOTAL_GROUPS_ATTEMPTED
        attempted_by_kind_at_start = dict(_GROUPS_ATTEMPTED_BY_KIND)
        invalid_at_start = _INVALID_DROPPED_GROUPS
        invalid_by_kind_at_start = dict(_INVALID_DROPPED_BY_KIND)
        filtered_at_start = _FILTER_DROPPED_GROUPS
        filtered_by_kind_at_start = dict(_FILTER_DROPPED_BY_KIND)
        stale_at_start = _STALE_DROPPED_GROUPS
        stale_by_kind_at_start = dict(_STALE_DROPPED_BY_KIND)
        reward_count_at_start = dict(_REWARD_GROUPS_OBSERVED_BY_KIND)
        reward_mean_sum_at_start = dict(_REWARD_GROUP_MEAN_SUM_BY_KIND)
        reward_variance_sum_at_start = dict(
            _REWARD_GROUP_VARIANCE_SUM_BY_KIND
        )
        zero_variance_at_start = dict(_ZERO_VARIANCE_GROUPS_BY_KIND)

    if (
        _SOURCE_VALIDATION_ERROR is not None
        and int(
            getattr(
                data_buffer,
                "last_validation_scheduled_attempt",
                getattr(data_buffer, "last_validation_attempt", 0),
            )
            or 0
        )
        >= int(_SOURCE_VALIDATION_ERROR.boundary or 0)
    ):
        _SOURCE_VALIDATION_ERROR = None

    lanes_values = _lanes_values_from_env()
    if lanes_values["terminal_rollout"]:
        raise ValueError(
            "Lane training must use terminal_rollout=false; terminal policy "
            "rollouts are reserved for evaluation=True"
        )

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

    instance_workers = max(1, int(os.environ.get("SWE_AGENT_LANES_INSTANCE_WORKERS", "8")))
    max_pending = max(1, int(
        os.environ.get("SWE_AGENT_LANES_MAX_PENDING", str(instance_workers))
    ))
    target_groups = int(args.rollout_batch_size)
    if target_groups <= 0:
        raise ValueError(f"rollout_batch_size must be positive, got {target_groups}")
    over_sampling_groups = int(args.over_sampling_batch_size)
    timeout = float(os.environ.get("SWE_AGENT_LANES_WAIT_TIMEOUT", "10800"))
    def _target_ready() -> bool:
        return len(_BUFFER) >= target_groups

    if not _NODE_WORKERS:
        skip = {
            s.strip()
            for s in os.environ.get("SWE_AGENT_LANES_SKIP_NODE_IPS", "").split(",")
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
        "[lanes-async] generate_rollout START rollout_id=%d target=%s "
        "target_groups=%d max_pending=%d buffer_at_entry=%d pending_at_entry=%d",
        rollout_id, target, target_groups, max_pending,
        len(_BUFFER), len(_PENDING),
    )

    def _est_groups() -> float:
        if _OBSERVED_INSTANCES <= 0:
            return (
                2.0
                if lanes_values.get("topology") == "depth2"
                else _DEFAULT_EST_GROUPS
            )
        return max(1.0, _OBSERVED_GROUPS_TOTAL / _OBSERVED_INSTANCES)

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
        preserved_kinds = _buffer_kind_counts()
        exc.preserved_partial = (
            preserved_count > 0 or preserved_pending > 0
        )
        exc.preserved_group_count = preserved_count
        exc.preserved_pending_count = preserved_pending
        exc.preserved_group_kinds = preserved_kinds
        _VALIDATION_PARTIAL_STATE = {
            "rollout_id": int(rollout_id),
            "boundary": int(exc.boundary or 0),
            "group_count": preserved_count,
            "group_kinds": preserved_kinds,
            "pending_count": preserved_pending,
            "started": started,
            "attempted_at_start": attempted_at_start,
            "attempted_by_kind_at_start": attempted_by_kind_at_start,
            "invalid_at_start": invalid_at_start,
            "invalid_by_kind_at_start": invalid_by_kind_at_start,
            "filtered_at_start": filtered_at_start,
            "filtered_by_kind_at_start": filtered_by_kind_at_start,
            "stale_at_start": stale_at_start,
            "stale_by_kind_at_start": stale_by_kind_at_start,
            "reward_count_at_start": reward_count_at_start,
            "reward_mean_sum_at_start": reward_mean_sum_at_start,
            "reward_variance_sum_at_start": reward_variance_sum_at_start,
            "zero_variance_at_start": zero_variance_at_start,
            "carried_in": carried_in,
            "carried_in_by_kind": carried_in_by_kind,
            "submitted": submitted,
        }
        raise exc

    if _SOURCE_VALIDATION_ERROR is not None:
        _preserve_and_raise_validation(_SOURCE_VALIDATION_ERROR)
    pending_boundary = _pending_validation_boundary(data_buffer)
    if pending_boundary is not None:
        _SOURCE_VALIDATION_ERROR = pending_boundary
        _preserve_and_raise_validation(pending_boundary)

    _harvest_ready(
        args=args,
        target=target,
        current_rollout_id=rollout_id,
        block=False,
    )
    if not _target_ready():
        submitted += _submit_until_full(
            args=args, data_buffer=data_buffer, rollout_id=rollout_id,
            output_root=output_root, model_name=model_name, max_pending=max_pending,
            over_sampling_groups=over_sampling_groups,
            est_groups_per_instance=_est_groups(),
        )
    if _SOURCE_VALIDATION_ERROR is not None:
        _preserve_and_raise_validation(_SOURCE_VALIDATION_ERROR)
    logger.info(
        "[lanes-async] post-init submitted=%d buffer=%d pending=%d "
        "over_sampling_groups=%d est_groups_per_instance=%.2f",
        submitted, len(_BUFFER), len(_PENDING),
        over_sampling_groups, _est_groups(),
    )

    wait_started = time.perf_counter()
    last_pulse = wait_started
    while not _target_ready():
        if time.perf_counter() - wait_started > timeout:
            raise RuntimeError(
                f"Lanes async timed out waiting for samples: "
                f"buffer={len(_BUFFER)} kinds={_buffer_kind_counts()} "
                f"pending={len(_PENDING)}"
            )
        _harvest_ready(
            args=args,
            target=target,
            current_rollout_id=rollout_id,
            block=bool(_PENDING),
        )
        if not _target_ready():
            submitted += _submit_until_full(
                args=args, data_buffer=data_buffer, rollout_id=rollout_id,
                output_root=output_root, model_name=model_name, max_pending=max_pending,
                over_sampling_groups=over_sampling_groups,
                est_groups_per_instance=_est_groups(),
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
                "[lanes-async] WAITING rollout_id=%d elapsed=%.0fs "
                "buffer=%d/%d pending=%d submitted_total=%d",
                rollout_id, time.perf_counter() - wait_started,
                len(_BUFFER), target_groups, len(_PENDING), submitted,
            )
            logger.info("[lanes-mem] %s", " | ".join(mems))
            _emit_heartbeat(
                source="lanes",
                rollout_id=rollout_id,
                elapsed_seconds=time.perf_counter() - wait_started,
                pending=len(_PENDING),
                buffered_groups=len(_BUFFER),
                submitted=submitted,
                output_root=output_root,
            )
        if not _PENDING and not _target_ready():
            if _SOURCE_BUDGET_EXHAUSTED:
                _VALIDATION_PARTIAL_STATE = None
                _discard_excess_buffer(reason="source_budget_partial_update")
                if _SOURCE_BUDGET_ERROR is None:
                    raise AssertionError(
                        "source budget is exhausted without its terminal error"
                    )
                raise _SOURCE_BUDGET_ERROR
            raise RuntimeError(
                f"Lanes async did not satisfy {target} quota "
                f"{{'all': {target_groups}}}; "
                f"buffer={_buffer_kind_counts()} and no pending instance."
            )

    # The final required group may itself be source attempt 100, 200, ...
    # In that case no subsequent get_samples(1) call exists to surface the
    # boundary, so check the durable cursor again before this batch can train.
    pending_boundary = _pending_validation_boundary(data_buffer)
    if pending_boundary is not None:
        _SOURCE_VALIDATION_ERROR = pending_boundary
        _preserve_and_raise_validation(pending_boundary)

    selected_groups = _pop_groups(target_groups)
    excess_dropped = 0
    excess_kind_counts: dict[str, int] = {}
    _VALIDATION_PARTIAL_STATE = None
    submitted += _submit_until_full(
        args=args,
        data_buffer=data_buffer,
        rollout_id=rollout_id,
        output_root=output_root,
        model_name=model_name,
        max_pending=max_pending,
        over_sampling_groups=over_sampling_groups,
        est_groups_per_instance=_est_groups(),
    )
    # Include any synchronously completed post-pop prefetch in the next
    # cycle's carry accounting.  Pending tasks are not attempted groups yet.
    carried_out_by_kind = _buffer_kind_counts()
    carried_out = len(_BUFFER)
    samples = [sample for group in selected_groups for sample in group.samples]
    selected_kind_counts: dict[str, int] = {}
    for group in selected_groups:
        selected_kind_counts[group.group_kind] = (
            selected_kind_counts.get(group.group_kind, 0) + 1
        )
    for group_index, group in enumerate(selected_groups):
        for sample in group.samples:
            sample.group_index = group_index
    for index, sample in enumerate(samples):
        sample.index = index

    # --- per-batch metrics aggregated from sample.metadata + sample fields ---
    # Stashed by lane_to_grpo_bundle._build_branch_sample:
    #   n_continuation_steps, n_parent_steps, n_full_trace_steps,
    #   raw_gt_score, terminated_early, and instance_id
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
    # Fraction of samples right-truncated to the per-sample token budget.
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

    # --- patch + eval ratios over post-filter batch (parity with naive) -----
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
    for s in real_samples:
        if s.metadata and s.metadata.get("n_full_trace_steps") is not None:
            turns_for_avg.append(int(s.metadata.get("n_full_trace_steps") or 0))

    attempted_step = _TOTAL_GROUPS_ATTEMPTED - attempted_at_start
    invalid_step = _INVALID_DROPPED_GROUPS - invalid_at_start
    filtered_step = _FILTER_DROPPED_GROUPS - filtered_at_start
    stale_step = _STALE_DROPPED_GROUPS - stale_at_start
    stale_step_by_kind = {
        kind: (
            _STALE_DROPPED_BY_KIND.get(kind, 0)
            - stale_by_kind_at_start.get(kind, 0)
        )
        for kind in ("root", "beam")
    }
    accepted_step = len(selected_groups)
    overall_outcome_metrics = _group_outcome_metrics(
        attempted=attempted_step,
        accepted=accepted_step,
        invalid=invalid_step,
        dynamic_filtered=filtered_step,
        excess=excess_dropped,
        carried_in=carried_in,
        carried_out=carried_out,
    )

    kind_outcome_metrics: dict[str, float | int] = {}
    for kind in ("root", "beam"):
        kind_attempted = (
            _GROUPS_ATTEMPTED_BY_KIND.get(kind, 0)
            - attempted_by_kind_at_start.get(kind, 0)
        )
        kind_invalid = (
            _INVALID_DROPPED_BY_KIND.get(kind, 0)
            - invalid_by_kind_at_start.get(kind, 0)
        )
        kind_filtered = (
            _FILTER_DROPPED_BY_KIND.get(kind, 0)
            - filtered_by_kind_at_start.get(kind, 0)
        )
        kind_accepted = selected_kind_counts.get(kind, 0)
        kind_excess = excess_kind_counts.get(kind, 0)
        kind_outcome_metrics.update(
            _group_outcome_metrics(
                attempted=kind_attempted,
                accepted=kind_accepted,
                invalid=kind_invalid,
                dynamic_filtered=kind_filtered,
                excess=kind_excess,
                carried_in=carried_in_by_kind.get(kind, 0),
                carried_out=carried_out_by_kind.get(kind, 0),
                prefix=f"{kind}_",
            )
        )

    reward_metrics: dict[str, float | int] = {}
    overall_reward_count = 0
    overall_reward_mean_sum = 0.0
    overall_reward_variance_sum = 0.0
    overall_zero_variance = 0
    for kind in ("root", "beam"):
        count = (
            _REWARD_GROUPS_OBSERVED_BY_KIND.get(kind, 0)
            - reward_count_at_start.get(kind, 0)
        )
        mean_sum = (
            _REWARD_GROUP_MEAN_SUM_BY_KIND.get(kind, 0.0)
            - reward_mean_sum_at_start.get(kind, 0.0)
        )
        variance_sum = (
            _REWARD_GROUP_VARIANCE_SUM_BY_KIND.get(kind, 0.0)
            - reward_variance_sum_at_start.get(kind, 0.0)
        )
        zero_count = (
            _ZERO_VARIANCE_GROUPS_BY_KIND.get(kind, 0)
            - zero_variance_at_start.get(kind, 0)
        )
        reward_metrics.update(
            {
                f"swe_agent/{kind}_reward_groups_observed": count,
                f"swe_agent/{kind}_reward_group_mean": (
                    mean_sum / count if count else 0.0
                ),
                f"swe_agent/{kind}_reward_group_variance_mean": (
                    variance_sum / count if count else 0.0
                ),
                f"swe_agent/{kind}_zero_variance_group_rate": (
                    zero_count / count if count else 0.0
                ),
            }
        )
        overall_reward_count += count
        overall_reward_mean_sum += mean_sum
        overall_reward_variance_sum += variance_sum
        overall_zero_variance += zero_count
    reward_metrics.update(
        {
            "swe_agent/reward_groups_observed": overall_reward_count,
            "swe_agent/reward_group_mean": (
                overall_reward_mean_sum / overall_reward_count
                if overall_reward_count else 0.0
            ),
            "swe_agent/reward_group_variance_mean": (
                overall_reward_variance_sum / overall_reward_count
                if overall_reward_count else 0.0
            ),
            "swe_agent/zero_variance_group_rate": (
                overall_zero_variance / overall_reward_count
                if overall_reward_count else 0.0
            ),
        }
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
            "swe_agent/excess_dropped_groups_total": _EXCESS_DROPPED_GROUPS,
            "swe_agent/invalid_dropped_groups_total": _INVALID_DROPPED_GROUPS,
            "swe_agent/total_groups_attempted_total": _TOTAL_GROUPS_ATTEMPTED,
            # This is the exact source-attempt axis used for epoch and
            # validation scheduling, independent of accepted optimizer groups.
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
            "swe_agent/stale_dropped_groups": stale_step,
            "swe_agent/excess_dropped_groups": excess_dropped,
            "swe_agent/root_groups": selected_kind_counts.get("root", 0),
            "swe_agent/beam_groups": selected_kind_counts.get("beam", 0),
            "swe_agent/root_filter_dropped_groups_total": (
                _FILTER_DROPPED_BY_KIND.get("root", 0)
            ),
            "swe_agent/beam_filter_dropped_groups_total": (
                _FILTER_DROPPED_BY_KIND.get("beam", 0)
            ),
            "swe_agent/root_invalid_dropped_groups_total": (
                _INVALID_DROPPED_BY_KIND.get("root", 0)
            ),
            "swe_agent/beam_invalid_dropped_groups_total": (
                _INVALID_DROPPED_BY_KIND.get("beam", 0)
            ),
            "swe_agent/root_excess_dropped_groups_total": (
                _EXCESS_DROPPED_BY_KIND.get("root", 0)
            ),
            "swe_agent/beam_excess_dropped_groups_total": (
                _EXCESS_DROPPED_BY_KIND.get("beam", 0)
            ),
            "swe_agent/root_stale_dropped_groups_total": (
                _STALE_DROPPED_BY_KIND.get("root", 0)
            ),
            "swe_agent/beam_stale_dropped_groups_total": (
                _STALE_DROPPED_BY_KIND.get("beam", 0)
            ),
            "swe_agent/root_stale_dropped_groups": (
                stale_step_by_kind["root"]
            ),
            "swe_agent/beam_stale_dropped_groups": (
                stale_step_by_kind["beam"]
            ),
            **overall_outcome_metrics,
            **kind_outcome_metrics,
            **reward_metrics,
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
            "swe_agent/rollout_logprob_mean": _safe_mean(rollout_lps),
            "swe_agent/rollout_logprob_token_count": len(rollout_lps),
            # --- patch + eval ratios over post-filter batch ---
            "swe_agent/ratio_formal_submit": (formal_submit / n_real) if n_real else 0.0,
            "swe_agent/ratio_informal_submit": (informal_submit / n_real) if n_real else 0.0,
            "swe_agent/ratio_zero_patch": (zero_patch / n_real) if n_real else 0.0,
            "swe_agent/avg_turns": _safe_mean([float(x) for x in turns_for_avg]),
        },
    )
    # Defer accepted until this exact root/beam group is selected into the
    # fully formed optimizer batch. In particular, a validation-boundary
    # partial or a fail-closed refill does not claim that untrained tokens
    # participated in an update.
    for group in selected_groups:
        _record_usage_disposition(
            group.usage_group_id,
            disposition="accepted",
        )
    output.metrics.update(_usage_metrics(commit_update=True))
    return output
