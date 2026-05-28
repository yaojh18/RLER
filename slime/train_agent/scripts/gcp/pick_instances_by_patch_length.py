#!/usr/bin/env python3
"""Pick swe-rebench-v2 train instances by gold-patch length.

Usage:
    python3 pick_instances_by_patch_length.py --offset 30 --limit 30
    # → prints 30 instance_ids whose gold patches sort to positions [30, 60)
    #   in ascending-length order (i.e. "next batch after the easy 30").

The "30 easy" set was offset=0 limit=30. This script picks the next slice
so we don't re-collect the same instances.
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    from datasets import load_from_disk, load_dataset
except ImportError:
    print("ERROR: pip install datasets", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        default="/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/datasets/v2_python1k_split_train",
        help="path to save_to_disk dataset (or HF id)",
    )
    p.add_argument("--split", default="train")
    p.add_argument("--offset", type=int, default=30)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--order", choices=["asc", "desc"], default="asc",
                   help="asc=easier first (shortest patches); desc=harder first")
    args = p.parse_args()

    if os.path.isdir(args.dataset):
        ds = load_from_disk(args.dataset)[args.split]
    else:
        ds = load_dataset(args.dataset, split=args.split)

    pairs = [(str(r["instance_id"]), len(r.get("patch") or "")) for r in ds]
    pairs.sort(key=lambda p: p[1], reverse=(args.order == "desc"))
    slice_ = pairs[args.offset : args.offset + args.limit]
    if len(slice_) < args.limit:
        print(
            f"WARN: only {len(slice_)} instances at offset={args.offset} (dataset has {len(pairs)})",
            file=sys.stderr,
        )
    print(" ".join(iid for iid, _ in slice_))
    # also dump min/max/median patch length for context
    if slice_:
        lengths = sorted(l for _, l in slice_)
        print(
            f"# slice patch_len: min={lengths[0]} median={lengths[len(lengths)//2]} max={lengths[-1]}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
