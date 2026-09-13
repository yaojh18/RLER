#!/usr/bin/env python3
"""Run one complete frozen-split evaluation using existing rollout/evaluator code.

This experiment-only entrypoint deliberately bypasses the training launcher:
the configured SGLang endpoints are already live on the same node, and no Megatron
actor is initialized. Infrastructure failures are retried in place; model
outcomes such as empty patches and context/completion limits are retained.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import signal
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_agent.collect_naive_rollout_async import (
    _naive_bundle_task,
    _naive_validation_gt_task,
)


class _EvaluationDeadlineExceeded(BaseException):
    """Escape evaluator-internal setup loops at the frozen wall-clock limit."""


def _bounded_naive_validation_gt_task(task: dict[str, Any]) -> dict[str, Any]:
    """Apply the 600s GT deadline to setup plus test execution.

    The underlying evaluator passes ``gt_eval_timeout`` to the test command,
    but repository dependency setup can happen before that command and can
    otherwise retry indefinitely.  A deadline here covers the complete worker
    call.  Deadline expiry is a valid unresolved evaluation outcome, not an
    infrastructure failure that should trigger another identical retry.
    """

    timeout = min(max(1, int(task.get("gt_eval_timeout", 600))), 600)
    descriptor = dict(task.get("validation_policy") or {})

    def deadline_handler(_signum: int, _frame: Any) -> None:
        raise _EvaluationDeadlineExceeded()

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, deadline_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return _naive_validation_gt_task(task)
    except _EvaluationDeadlineExceeded:
        return {
            **task,
            "bundle": None,
            "validation": {
                "rollout_error": descriptor.get("rollout_error"),
                "terminated_early": bool(descriptor.get("terminated_early")),
                "rollout_status": str(descriptor.get("rollout_status") or "error"),
                "total_tokens": dict(descriptor.get("total_tokens") or {}),
                "policy_version": descriptor.get("policy_version"),
                "observed_policy_start": descriptor.get("observed_policy_start"),
                "observed_policy_end": descriptor.get("observed_policy_end"),
                "status": "timeout",
                "resolved": False,
                "infrastructure_error": False,
                "timeout_seconds": timeout,
            },
            "error": "",
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def validation_rows(path: Path, samples_per_instance: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        metadata = dict(row.get("metadata") or {})
        base_rows.append(metadata)
    for sample_index in range(samples_per_instance):
        for instance_offset, metadata in enumerate(base_rows):
            records.append(
                {
                    **metadata,
                    "index": sample_index * len(base_rows) + instance_offset,
                    "instance_id": str(metadata["instance_id"]),
                    "sample_index": sample_index,
                    "subset": str(metadata.get("subset") or "verified"),
                    "split": str(metadata.get("dataset_split") or metadata.get("split") or "test"),
                }
            )
    return records


def policy_task(
    row: dict[str, Any],
    *,
    attempt: int,
    endpoints: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    endpoint = endpoints[(int(row["index"]) + attempt) % len(endpoints)]
    usage_group_id = (
        f"validation:r{args.rollout_id}:i{row['instance_id']}:"
        f"s{row['sample_index']}:a{attempt}"
    )
    return {
        "index": int(row["index"]),
        "rollout_id": args.rollout_id,
        "instance_id": row["instance_id"],
        "sample_index": int(row["sample_index"]),
        "subset": row["subset"],
        "split": row["split"],
        "output_root": str(args.output_root / "validation" / f"rollout_{args.rollout_id:04d}"),
        "model_name": args.model_name,
        "policy_base_url": endpoint,
        "policy_base_urls": [endpoint],
        "api_key": "EMPTY",
        "m": 1,
        "step_limit": args.step_limit,
        "gt_eval_workers": 1,
        "gt_eval_timeout": args.gt_eval_timeout,
        "rollout_pool_size": 1,
        "completion_max_tokens": args.completion_max_tokens,
        "context_length": args.context_length,
        "policy_temperature": args.temperature,
        "policy_top_p": args.top_p,
        "fallback_patch_penalty": 1.0,
        "no_action_patch_penalty": 0.0,
        "reward_kind": "hard",
        "joint_alpha": 1.0,
        "all_pass_reward": 1.0,
        "validation": True,
        "defer_gt_evaluation": True,
        "usage_ledger": str(args.usage_ledger),
        "usage_phase": "validation",
        "usage_group_prefix": usage_group_id,
        "usage_group_id": usage_group_id,
        "policy_version": args.policy_version,
        "variance_resample_attempt": attempt,
    }


def fixed_endpoint_batches(tasks: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    by_endpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_endpoint[str(task["policy_base_url"])].append(task)
    endpoints = sorted(by_endpoint)
    slots = {endpoint: 1 for endpoint in endpoints}
    remaining = max(0, workers - len(endpoints))
    for offset in range(remaining):
        slots[endpoints[offset % len(endpoints)]] += 1
    batches: list[list[dict[str, Any]]] = []
    for endpoint in endpoints:
        values = by_endpoint[endpoint]
        endpoint_slots = min(slots[endpoint], len(values))
        for slot in range(endpoint_slots):
            batch = values[slot::endpoint_slots]
            if batch:
                batches.append(batch)
    return batches


def run_bounded_policy_batch(
    tasks: list[dict[str, Any]], max_live_tasks: int
) -> list[dict[str, Any]]:
    """Consume one frozen endpoint batch with bounded sliding concurrency."""

    if not tasks:
        return []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max_live_tasks, len(tasks)),
        thread_name_prefix="bounded-validation-policy",
    ) as executor:
        return list(executor.map(_naive_bundle_task, tasks))


def run_policy(tasks: list[dict[str, Any]], workers: int) -> dict[int, dict[str, Any]]:
    if not tasks:
        return {}
    context = multiprocessing.get_context("spawn")
    results: dict[int, dict[str, Any]] = {}
    # The original collector batch helper starts one thread per batch item. The
    # historical 50-instance x 4-sample validation therefore ran 200 live
    # agents across eight endpoint batches, but an 800-row test set quadrupled
    # that footprint and exhausted host memory. Keep the same eight frozen
    # endpoint batches while limiting each batch to 25 sliding threads, so a
    # completed task is replenished immediately and total live agents stay at
    # or below the proven 200-row validation footprint.
    batches = fixed_endpoint_batches(tasks, workers)
    batch_concurrency = max(1, 200 // len(batches))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(batches), mp_context=context
    ) as executor:
        future_to_batch = {
            executor.submit(run_bounded_policy_batch, batch, batch_concurrency): batch
            for batch in batches
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                batch_results = future.result()
            except Exception as exc:
                batch_results = [
                    {
                        "index": task["index"],
                        "instance_id": task["instance_id"],
                        "validation_policy": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    for task in batch
                ]
            for result in batch_results:
                results[int(result["index"])] = result
    return results


def run_evaluations(
    tasks: list[dict[str, Any]], workers: int
) -> dict[int, dict[str, Any]]:
    if not tasks:
        return {}
    context = multiprocessing.get_context("spawn")
    results: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(workers, len(tasks)), mp_context=context
    ) as executor:
        future_to_task = {
            executor.submit(_bounded_naive_validation_gt_task, task): task
            for task in tasks
        }
        for future, task in future_to_task.items():
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "index": task["index"],
                    "instance_id": task["instance_id"],
                    "validation": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results[int(task["index"])] = result
    return results


def valid_evaluation(result: dict[str, Any] | None) -> bool:
    validation = (result or {}).get("validation")
    return isinstance(validation, dict) and not bool(validation.get("infrastructure_error"))


def load_reusable_policies(
    root: Path,
    rows: dict[int, dict[str, Any]],
    *,
    rollout_id: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, int]]:
    """Recover completed policy descriptors after a Slurm interruption."""
    if not root.is_dir():
        return {}, {}
    candidates: dict[int, list[tuple[int, float, dict[str, Any]]]] = defaultdict(list)
    marker_pattern = f"validation/rollout_{rollout_id:04d}/**/validation_policy.json"
    for marker_path in sorted(root.glob(marker_pattern)):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            index = int(marker["index"])
            descriptor = marker.get("validation_policy")
            if index not in rows or not isinstance(descriptor, dict):
                continue
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        retry = int(marker.get("attempt", 0))
        candidates[index].append(
            (
                retry,
                marker_path.stat().st_mtime,
                {
                    "index": index,
                    "instance_id": str(rows[index]["instance_id"]),
                    "validation_policy": descriptor,
                    "error": "",
                },
            )
        )

    results: dict[int, dict[str, Any]] = {}
    attempts: dict[int, int] = {}
    for index, values in candidates.items():
        retry, _, result = max(values, key=lambda value: (value[0], value[1]))
        results[index] = result
        attempts[index] = retry + 1
    return results, attempts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--rollout-id", type=int, default=0)
    parser.add_argument("--samples-per-instance", type=int, default=4)
    parser.add_argument("--step-limit", type=int, default=250)
    parser.add_argument("--context-length", type=int, default=128000)
    parser.add_argument("--completion-max-tokens", type=int, default=10240)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--gt-eval-timeout", type=int, default=600)
    parser.add_argument("--policy-workers", type=int, default=8)
    parser.add_argument("--gt-eval-workers", type=int, default=16)
    parser.add_argument("--infrastructure-retries", type=int, default=3)
    parser.add_argument(
        "--reuse-policy-root",
        type=Path,
        help="Reuse collector-complete policy artifacts from an interrupted run.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.usage_ledger = args.output_root / "usage.jsonl"
    rows = validation_rows(args.manifest, args.samples_per_instance)
    by_index = {int(row["index"]): row for row in rows}
    started = time.time()

    policy_results: dict[int, dict[str, Any]] = {}
    policy_attempts: dict[int, int] = {}
    if args.reuse_policy_root is not None:
        policy_results, policy_attempts = load_reusable_policies(
            args.reuse_policy_root,
            by_index,
            rollout_id=args.rollout_id,
        )
    pending = [index for index in sorted(by_index) if index not in policy_results]
    for attempt in range(args.infrastructure_retries + 1):
        if not pending:
            break
        tasks = [
            policy_task(by_index[index], attempt=attempt, endpoints=args.endpoint, args=args)
            for index in pending
        ]
        observed = run_policy(tasks, args.policy_workers)
        for index in pending:
            policy_results[index] = observed.get(index, {})
            policy_attempts[index] = attempt + 1
        pending = [
            index
            for index in pending
            if not isinstance(
                (policy_results.get(index) or {}).get("validation_policy"),
                dict,
            )
        ]

    evaluation_results: dict[int, dict[str, Any]] = {}
    evaluation_attempts: dict[int, int] = {}
    evaluable = [
        index
        for index in sorted(by_index)
        if isinstance(
            (policy_results.get(index) or {}).get("validation_policy"),
            dict,
        )
    ]
    pending_eval = evaluable
    for attempt in range(args.infrastructure_retries + 1):
        if not pending_eval:
            break
        tasks = []
        for index in pending_eval:
            base = policy_task(
                by_index[index],
                attempt=max(0, policy_attempts.get(index, 1) - 1),
                endpoints=args.endpoint,
                args=args,
            )
            tasks.append(
                {
                    **base,
                    "validation_policy": policy_results[index]["validation_policy"],
                }
            )
        observed = run_evaluations(tasks, args.gt_eval_workers)
        for index in pending_eval:
            evaluation_results[index] = observed.get(index, {})
            evaluation_attempts[index] = attempt + 1
        pending_eval = [
            index for index in pending_eval if not valid_evaluation(evaluation_results.get(index))
        ]

    policy_records = []
    evaluation_records = []
    for index in sorted(by_index):
        row = by_index[index]
        policy = policy_results.get(index, {})
        policy_records.append(
            {
                "index": row["index"],
                "instance_id": row["instance_id"],
                "sample_index": row["sample_index"],
                "error": str(policy.get("error") or ""),
                "validation_policy": policy.get("validation_policy"),
                "attempts": policy_attempts.get(index, 0),
            }
        )
        evaluation = evaluation_results.get(index, {})
        evaluation_records.append(
            {
                "index": row["index"],
                "instance_id": row["instance_id"],
                "sample_index": row["sample_index"],
                "error": str(evaluation.get("error") or ""),
                "validation": dict(evaluation.get("validation") or {}),
                "attempts": evaluation_attempts.get(index, 0),
            }
        )
    atomic_json(args.output_root / "policy_results.json", policy_records)
    atomic_json(args.output_root / "evaluation_results.json", evaluation_records)

    status_counts: Counter[str] = Counter()
    rollout_error_counts: Counter[str] = Counter()
    resolved = 0
    complete = 0
    per_instance: dict[str, dict[str, Any]] = {}
    for record in evaluation_records:
        validation = record["validation"]
        valid = valid_evaluation({"validation": validation})
        if valid:
            complete += 1
            status = str(validation.get("status") or "error")
            status_counts[status] += 1
            resolved += int(status == "resolved")
            rollout_error = str(validation.get("rollout_error") or "")
            if rollout_error:
                rollout_error_counts[rollout_error.splitlines()[0][:240]] += 1
        entry = per_instance.setdefault(
            record["instance_id"], {"planned": 0, "completed": 0, "resolved": 0}
        )
        entry["planned"] += 1
        entry["completed"] += int(valid)
        entry["resolved"] += int(valid and str(validation.get("status")) == "resolved")
    for entry in per_instance.values():
        entry["resolved_rate"] = entry["resolved"] / entry["completed"] if entry["completed"] else 0.0

    summary = {
        "schema_version": "standalone_checkpoint_evaluation.v1",
        "status": "complete" if complete == len(rows) else "incomplete",
        "policy_version": args.policy_version,
        "model_name": args.model_name,
        "config": {
            "instances": len({row["instance_id"] for row in rows}),
            "samples_per_instance": args.samples_per_instance,
            "planned": len(rows),
            "task_indices": sorted(by_index),
            "step_limit": args.step_limit,
            "context_length": args.context_length,
            "completion_max_tokens": args.completion_max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "gt_eval_timeout": args.gt_eval_timeout,
        },
        "completed": complete,
        "infrastructure_errors": len(rows) - complete,
        "resolved": resolved,
        "resolved_rate_completed": resolved / complete if complete else 0.0,
        "resolved_rate_planned": resolved / len(rows) if rows else 0.0,
        "status_counts": dict(status_counts),
        "rollout_error_counts": dict(rollout_error_counts),
        "policy_retry_histogram": dict(Counter(policy_attempts.values())),
        "evaluation_retry_histogram": dict(Counter(evaluation_attempts.values())),
        "per_instance": per_instance,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.output_root / "summary.json", summary)
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
