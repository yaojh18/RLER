#!/usr/bin/env python3
from __future__ import annotations

import atexit
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.utils.types import Sample

from train_agent.collect_grpo_rollout import (
    _collect_grpo_bundle_task,
    _rollout_tokenizer,
    build_rollout_samples,
)
from train_agent.serving.sglang_chat_service import start_slime_policy_route_warmup


@dataclass
class BufferedGroup:
    rollout_id: int
    samples: list[Sample]


_EXECUTOR: ProcessPoolExecutor | None = None
_EXECUTOR_WORKERS = 0
_PENDING: dict[Future, dict[str, Any]] = {}
_BUFFER: list[BufferedGroup] = []
_TASK_INDEX = 0
_WARMUP_DONE = False
_STALE_DROPPED_GROUPS = 0
_FAILED_INSTANCES = 0


def _shutdown_executor() -> None:
    global _EXECUTOR
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _EXECUTOR = None


atexit.register(_shutdown_executor)


def _executor(max_workers: int) -> ProcessPoolExecutor:
    global _EXECUTOR, _EXECUTOR_WORKERS
    if _EXECUTOR is None or _EXECUTOR_WORKERS != max_workers:
        _shutdown_executor()
        _EXECUTOR = ProcessPoolExecutor(max_workers=max_workers)
        _EXECUTOR_WORKERS = max_workers
    return _EXECUTOR


def _search_values_from_env() -> dict[str, int | None]:
    values: dict[str, int | None] = {}
    for env_name, arg_name in (
        ("SWE_AGENT_GRPO_M", "m"),
        ("SWE_AGENT_GRPO_N", "n"),
        ("SWE_AGENT_GRPO_K", "k"),
        ("SWE_AGENT_GRPO_P", "p"),
        ("SWE_AGENT_GRPO_MAX_ROUNDS", "max_rounds"),
        ("SWE_AGENT_GRPO_STEP_LIMIT", "step_limit"),
    ):
        value = os.environ[env_name]
        values[arg_name] = int(value) if value else None
    return values


def _submit_until_full(
    *,
    args,
    data_buffer,
    rollout_id: int,
    output_root: Path,
    model_name: str,
    max_pending: int,
    executor: ProcessPoolExecutor,
) -> int:
    global _TASK_INDEX
    submitted = 0
    search_values = _search_values_from_env()
    router_ip, router_port = (getattr(args, "sglang_model_routers", None) or {}).get(
        model_name,
        (args.sglang_router_ip, args.sglang_router_port),
    )
    slime_api_base = f"http://{router_ip}:{router_port}"
    slime_api_key = os.environ.get("SEARCH_SWE_SLIME_API_KEY", "EMPTY")
    while len(_PENDING) < max_pending:
        prompt_groups = data_buffer.get_samples(1)
        if not prompt_groups:
            break
        (prompt_group,) = prompt_groups
        if len(prompt_group) != 1:
            raise RuntimeError(
                f"SWE-agent rollout expects one slime sample per prompt group; got {len(prompt_group)}. "
                "Keep --n-samples-per-prompt 1 because SWE search creates GRPO groups internally."
            )
        metadata = prompt_group[0].metadata
        task = {
            "index": _TASK_INDEX,
            "rollout_id": rollout_id,
            "instance_id": metadata["instance_id"],
            "subset": metadata["subset"],
            "split": metadata["split"],
            "output_root": output_root / f"rollout_{rollout_id:04d}",
            "model_name": model_name,
            "workers": int(os.environ["SWE_AGENT_GRPO_WORKERS"]),
            "search_values": search_values,
            "slime_api_base": slime_api_base,
            "slime_api_key": slime_api_key,
        }
        _TASK_INDEX += 1
        _PENDING[executor.submit(_collect_grpo_bundle_task, task)] = task
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
    global _FAILED_INSTANCES, _STALE_DROPPED_GROUPS
    if not _PENDING:
        return 0
    ready = {future for future in _PENDING if future.done()}
    if not ready and block:
        ready, _ = wait(_PENDING, return_when=FIRST_COMPLETED)
    harvested = 0
    for future in ready:
        task = _PENDING.pop(future)
        result = future.result()
        if result["error"]:
            _FAILED_INSTANCES += 1
            continue
        bundle = result["bundle"]
        groups = bundle.policy_groups if target == "policy" else bundle.rubric_groups
        if task["rollout_id"] < current_rollout_id - 1:
            _STALE_DROPPED_GROUPS += len(groups)
            continue
        for group in groups:
            samples = build_rollout_samples(
                groups=[group],
                tokenizer=tokenizer,
                loss_mask_type=getattr(args, "loss_mask_type", "qwen3_5"),
                include_turn_rewards=target == "rubric",
            )
            policy_version = f"rollout-{task['rollout_id']}"
            for sample in samples:
                sample.weight_versions = [policy_version]
                sample.metadata = {
                    **(sample.metadata or {}),
                    "policy_version": policy_version,
                    "source_rollout_id": task["rollout_id"],
                    "instance_id": task["instance_id"],
                }
            _BUFFER.append(BufferedGroup(rollout_id=task["rollout_id"], samples=samples))
            harvested += 1
    return harvested


def _drop_stale_buffer(current_rollout_id: int) -> None:
    global _STALE_DROPPED_GROUPS
    kept = [group for group in _BUFFER if group.rollout_id >= current_rollout_id - 1]
    _STALE_DROPPED_GROUPS += len(_BUFFER) - len(kept)
    _BUFFER[:] = kept


def _pop_groups(min_groups: int) -> list[BufferedGroup]:
    groups = _BUFFER[:min_groups]
    del _BUFFER[:min_groups]
    return groups


def generate_rollout(args, rollout_id: int, data_buffer, evaluation: bool = False):
    global _WARMUP_DONE
    if evaluation:
        return RolloutFnEvalOutput(data={}, metrics={})

    started = time.perf_counter()
    target = os.environ.get("SWE_AGENT_GRPO_TARGET", "policy")
    output_root = Path(os.environ.get("SWE_AGENT_GRPO_OUTPUT_ROOT", "/workspace/rler/agent/outputs/search_outputs/train_async_grpo"))
    model_name = os.environ.get("SWE_AGENT_GRPO_MODEL_NAME") or "Qwen/Qwen3.5-9B"
    router_ip, router_port = (getattr(args, "sglang_model_routers", None) or {}).get(
        model_name,
        (args.sglang_router_ip, args.sglang_router_port),
    )
    if not _WARMUP_DONE:
        start_slime_policy_route_warmup(
            base_url=f"http://{router_ip}:{router_port}",
            api_key=os.environ.get("SEARCH_SWE_SLIME_API_KEY", "EMPTY"),
            model_name=model_name,
            requests=max(1, int(args.rollout_num_gpus or 1) // int(args.rollout_num_gpus_per_engine or 1)),
        )
        _WARMUP_DONE = True

    tokenizer = _rollout_tokenizer(args)
    instance_workers = max(1, int(os.environ["SWE_AGENT_GRPO_INSTANCE_WORKERS"]))
    max_pending = max(1, int(os.environ.get("SWE_AGENT_GRPO_ASYNC_MAX_PENDING", str(instance_workers))))
    min_ready_groups = max(1, int(os.environ.get("SWE_AGENT_GRPO_ASYNC_MIN_READY_GROUPS", str(args.rollout_batch_size))))
    timeout = float(os.environ.get("SWE_AGENT_GRPO_ASYNC_WAIT_TIMEOUT", "7200"))
    executor = _executor(instance_workers)

    _drop_stale_buffer(rollout_id)
    _harvest_ready(args=args, target=target, tokenizer=tokenizer, current_rollout_id=rollout_id, block=False)
    submitted = _submit_until_full(
        args=args,
        data_buffer=data_buffer,
        rollout_id=rollout_id,
        output_root=output_root,
        model_name=model_name,
        max_pending=max_pending,
        executor=executor,
    )

    wait_started = time.perf_counter()
    while len(_BUFFER) < min_ready_groups:
        if time.perf_counter() - wait_started > timeout:
            raise RuntimeError(
                f"SWE-agent async GRPO timed out waiting for samples: buffer={len(_BUFFER)} pending={len(_PENDING)}"
            )
        _harvest_ready(args=args, target=target, tokenizer=tokenizer, current_rollout_id=rollout_id, block=bool(_PENDING))
        submitted += _submit_until_full(
            args=args,
            data_buffer=data_buffer,
            rollout_id=rollout_id,
            output_root=output_root,
            model_name=model_name,
            max_pending=max_pending,
            executor=executor,
        )
        if not _PENDING and len(_BUFFER) < min_ready_groups:
            raise RuntimeError(f"SWE-agent async GRPO produced no {target} samples and has no pending instance.")

    selected_groups = _pop_groups(min_ready_groups)
    _submit_until_full(
        args=args,
        data_buffer=data_buffer,
        rollout_id=rollout_id,
        output_root=output_root,
        model_name=model_name,
        max_pending=max_pending,
        executor=executor,
    )
    samples = [sample for group in selected_groups for sample in group.samples]
    for group_index, group in enumerate(selected_groups):
        for sample in group.samples:
            sample.group_index = group_index
    for index, sample in enumerate(samples):
        sample.index = index

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
            "swe_agent/wait_seconds": time.perf_counter() - wait_started,
            "swe_agent/seconds": time.perf_counter() - started,
        },
    )
