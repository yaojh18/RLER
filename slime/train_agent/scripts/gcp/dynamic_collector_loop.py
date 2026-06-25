"""Dynamic-scheduling collector loop.

Each collector process runs this in a long loop:
  1. flock-pop next instance_id from a shared queue.txt
  2. run collect_teacher_student_export_parallel on it (WORKERS=1 internally)
  3. append result to a shared done.jsonl
  4. repeat until queue empty

Replaces static round-robin sharding. No instance pre-assignment; each
collector grabs the next free one when it finishes, so a collector that
gets stuck on a hard instance doesn't block easy instances on its shard.

Atomicity: POSIX fcntl.LOCK_EX on the queue file. The whole pop is
serialized across collectors; processing happens AFTER unlock so we
maximize parallelism between collectors.

Usage (inside baked slime container):
    python3 dynamic_collector_loop.py \\
        --queue-file <art>/queue.txt \\
        --done-file <art>/done.jsonl \\
        --teacher-base-url http://metavmds1-a4-135:8888 \\
        --output-root <art>/search_outputs \\
        --collector-id col-NN \\
        --search-m 8 --search-n 2 [...]
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path


def teacher_alive(base_url: str, timeout: float = 5.0) -> bool:
    """Return True iff teacher /health responds 200 within `timeout` seconds."""
    url = base_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


def pop_next_instance(queue_file: Path) -> str | None:
    """Atomically pop the FIRST line from queue_file under fcntl.LOCK_EX.
    Returns the instance_id, or None if queue is empty."""
    if not queue_file.exists():
        return None
    # Open r+ so we can rewrite. flock blocks until we get exclusive access.
    with open(queue_file, "r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            lines = f.readlines()
            if not lines:
                return None
            head = lines[0].strip()
            f.seek(0)
            f.writelines(lines[1:])
            f.truncate()
            return head or None
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def append_done(done_file: Path, record: dict) -> None:
    """Atomically append a single JSON line to done_file."""
    with open(done_file, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--queue-file", required=True, type=Path)
    p.add_argument("--done-file", required=True, type=Path)
    p.add_argument("--teacher-base-url", required=True)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--collector-id", default="?")
    p.add_argument("--rler-root", default="/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/RLER-zsh-trim")
    # search args
    p.add_argument("--search-m", type=int, default=8)
    p.add_argument("--search-n", type=int, default=2)
    p.add_argument("--lanes-max-mid-cps", type=int, default=6)
    p.add_argument("--lanes-steps-per-round", type=int, default=20)
    p.add_argument("--search-step-limit", type=int, default=120)
    p.add_argument("--gt-eval-workers", type=int, default=8)
    p.add_argument("--rubric-bank-strategy", default="score")
    p.add_argument("--subset", default="rebench_v2")
    p.add_argument("--split", default="train")
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.rler_root) / "agent"))
    sys.path.insert(0, str(Path(args.rler_root) / "slime"))
    from train_agent.collect_sft_rollout import collect_teacher_student_export_parallel  # noqa

    n_done = 0
    n_err = 0
    t_start = time.time()
    while True:
        iid = pop_next_instance(args.queue_file)
        if iid is None:
            elapsed = time.time() - t_start
            print(
                f"[{args.collector_id}] queue empty, exiting "
                f"(n_done={n_done} n_err={n_err} elapsed={elapsed:.0f}s)",
                flush=True,
            )
            return 0
        t0 = time.time()
        print(f"[{args.collector_id}] start {iid}", flush=True)
        try:
            bundle = collect_teacher_student_export_parallel(
                instance_id=iid,
                output_root=args.output_root,
                teacher_model_name="openai/deepseek-v4-pro",
                teacher_base_url=args.teacher_base_url,
                teacher_api_key="EMPTY",
                subset=args.subset,
                split=args.split,
                m=args.search_m,
                n=args.search_n,
                max_mid_cps=args.lanes_max_mid_cps,
                steps_per_round=args.lanes_steps_per_round,
                step_limit=args.search_step_limit,
                gt_eval_workers=args.gt_eval_workers,
                rubric_bank_strategy=args.rubric_bank_strategy,
            )
            res = {
                "instance_id": iid,
                "status": "ok",
                "collector_id": args.collector_id,
                "teacher_base_url": args.teacher_base_url,
                "policy_samples": len(bundle.policy_samples),
                "rubric_samples": len(bundle.rubric_samples),
                "accepted_groups": len(bundle.accepted_group_ids),
                "run_dir": bundle.run_dir,
                "seconds": time.time() - t0,
                "timestamp_done": time.time(),
            }
            n_done += 1
        except Exception as exc:
            res = {
                "instance_id": iid,
                "status": "error",
                "collector_id": args.collector_id,
                "teacher_base_url": args.teacher_base_url,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[:4000],
                "seconds": time.time() - t0,
                "timestamp_done": time.time(),
            }
            n_err += 1

        # Silent-failure guard: when collect_teacher_student_export_parallel
        # returns an empty bundle, ping teacher /health. If the teacher is
        # dead we requeue the instance and exit; otherwise the empty bundle
        # is genuine (instance had zero passing branches). Without this guard
        # a dead teacher cascades into hundreds of fake status="ok" rows
        # because the underlying agent code swallows ConnectionError and
        # returns an empty trajectory.
        is_empty = (
            res.get("status") == "ok"
            and res.get("policy_samples", 0) == 0
            and res.get("rubric_samples", 0) == 0
            and res.get("accepted_groups", 0) == 0
        )
        if is_empty and not teacher_alive(args.teacher_base_url):
            res["status"] = "error"
            res["error"] = "teacher_unreachable_after_empty_bundle"
            n_done -= 1
            n_err += 1
            try:
                with open(args.queue_file, "r+") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        existing = f.read()
                        f.seek(0)
                        f.write(iid + "\n" + existing)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                print(
                    f"[{args.collector_id}] teacher {args.teacher_base_url} "
                    f"unreachable, requeued {iid}, exiting",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[{args.collector_id}] teacher dead AND requeue failed "
                    f"({type(exc).__name__}: {exc}); {iid} lost",
                    flush=True,
                )
            append_done(args.done_file, res)
            return 2

        append_done(args.done_file, res)
        print(
            f"[{args.collector_id}] done {iid} status={res['status']} "
            f"pol={res.get('policy_samples',0)} rub={res.get('rubric_samples',0)} "
            f"sec={res['seconds']:.0f} (this_col total: {n_done} ok + {n_err} err)",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
