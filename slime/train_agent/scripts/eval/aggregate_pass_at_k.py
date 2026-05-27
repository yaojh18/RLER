#!/usr/bin/env python3
"""Aggregate pass@1 / pass@2 / pass@4 across one or more model run dirs.

Each model run dir is expected to contain attempt-{1..N}/nebius-eval/eval_report.json
(produced by slime/train_agent/scripts/eval/eval_pass_at_4_*.slurm).

Usage:
  python aggregate_pass_at_k.py /mnt/lustre/.../<jid>-evp4-9b-<tag> [<jid>-evp4-9b-<tag2>] ...
"""

from __future__ import annotations

import glob
import json
import os
import sys


def load_attempts(run_dir: str) -> list[dict[str, bool]]:
    attempts: list[dict[str, bool]] = []
    for d in sorted(glob.glob(os.path.join(run_dir, "attempt-*"))):
        rep = os.path.join(d, "nebius-eval", "eval_report.json")
        if not os.path.exists(rep):
            print(f"  ! missing {rep}")
            continue
        try:
            r = json.load(open(rep))
        except Exception as exc:
            print(f"  ! could not load {rep}: {exc}")
            continue
        attempts.append(
            {it["instance_id"]: bool(it.get("passed_match")) for it in r.get("items", [])}
        )
    return attempts


def pass_at_k(attempts: list[dict[str, bool]], k: int) -> tuple[int, int]:
    if k > len(attempts):
        return -1, -1
    iids = sorted(set().union(*[a.keys() for a in attempts]))
    passed = sum(
        1 for iid in iids if any(a.get(iid, False) for a in attempts[:k])
    )
    return passed, len(iids)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    rows = []
    for run_dir in argv:
        tag = os.path.basename(run_dir.rstrip("/"))
        print(f"\n=== {tag} ===")
        attempts = load_attempts(run_dir)
        if not attempts:
            print("  (no attempts loaded)")
            continue
        # per-attempt pass@1
        per_attempt = []
        for i, a in enumerate(attempts, start=1):
            n = len(a)
            p = sum(1 for v in a.values() if v)
            print(f"  attempt-{i}: {p}/{n} = {100.0*p/n if n else 0:.1f}%")
            per_attempt.append((p, n))
        # pass@k
        pks = {}
        for k in (1, 2, 4):
            p, n = pass_at_k(attempts, k)
            if p < 0:
                continue
            pks[k] = (p, n)
            print(f"  pass@{k}: {p}/{n} = {100.0*p/n:.1f}%")
        rows.append((tag, pks))

    if rows:
        print("\n=== summary ===")
        print(f"{'model':<40}{'pass@1':>10}{'pass@2':>10}{'pass@4':>10}")
        for tag, pks in rows:
            cells = []
            for k in (1, 2, 4):
                if k in pks:
                    p, n = pks[k]
                    cells.append(f"{100.0*p/n:.1f}%" if n else "n/a")
                else:
                    cells.append("n/a")
            print(f"{tag:<40}{cells[0]:>10}{cells[1]:>10}{cells[2]:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
