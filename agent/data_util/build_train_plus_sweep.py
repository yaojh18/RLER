"""Assemble the final training dataset: base train + sweep middle band.

Steps:
  1. Read the eval split and collect the set of repos to exclude.
  2. Read the middle-band TSV; keep rows whose source_shard starts with the
     configured prefix (default 'pysweep').
  3. Load all sweep DatasetDict shards, concat, filter to those ids, dedup.
  4. Concat with the base train split.
  5. Drop any row whose `repo` appears in the eval split.
  6. Shuffle with a fixed seed and save as a DatasetDict-on-disk.

See agent/data_util/README.md for the full pipeline.
"""
import argparse
import os
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_from_disk


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base",
        default="/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench",
    )
    p.add_argument("--train", default=None, help="Default: <base>/datasets/v2_python1k_split_train")
    p.add_argument("--eval", default=None, help="Default: <base>/datasets/v2_python1k_split_eval")
    p.add_argument("--sweep-root", default=None, help="Default: <base>/datasets/v2_python_sweep_shards")
    p.add_argument("--tsv", default=None, help="Default: <base>/datasets/v2_python_pass1to3of4.tsv")
    p.add_argument("--out", default=None, help="Default: <base>/datasets/v2_python_train_plus_sweep_middle")
    p.add_argument("--num-sweep-shards", type=int, default=16)
    p.add_argument("--source-prefix", default="pysweep",
                   help="Only TSV rows whose source_shard starts with this prefix contribute sweep ids.")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--num-proc", type=int, default=8)
    args = p.parse_args()

    base = Path(args.base)
    train_path = Path(args.train) if args.train else base / "datasets" / "v2_python1k_split_train"
    eval_path = Path(args.eval) if args.eval else base / "datasets" / "v2_python1k_split_eval"
    sweep_root = Path(args.sweep_root) if args.sweep_root else base / "datasets" / "v2_python_sweep_shards"
    tsv_path = Path(args.tsv) if args.tsv else base / "datasets" / "v2_python_pass1to3of4.tsv"
    out_path = Path(args.out) if args.out else base / "datasets" / "v2_python_train_plus_sweep_middle"

    eval_ds = load_from_disk(str(eval_path))["train"]
    eval_repos = set(eval_ds["repo"])
    print(f"eval repos to exclude: {len(eval_repos)}")

    sweep_ids = set()
    with open(tsv_path) as f:
        next(f)  # header
        for line in f:
            iid, _, src = line.rstrip("\n").split("\t")
            if src.startswith(args.source_prefix):
                sweep_ids.add(iid)
    print(f"sweep middle-band ids requested: {len(sweep_ids)}")

    train_ds = load_from_disk(str(train_path))["train"]
    print(f"base train rows: {len(train_ds)}")

    sweep_parts = [
        load_from_disk(str(sweep_root / f"shard_{i:02d}"))["train"]
        for i in range(args.num_sweep_shards)
    ]
    sweep_combined = concatenate_datasets(sweep_parts)
    sweep_pick = sweep_combined.filter(lambda r: r["instance_id"] in sweep_ids, num_proc=args.num_proc)
    print(f"sweep middle-band rows pulled (pre-dedup): {len(sweep_pick)}")

    seen, keep = set(), []
    for i, iid in enumerate(sweep_pick["instance_id"]):
        if iid in seen:
            continue
        seen.add(iid)
        keep.append(i)
    sweep_pick = sweep_pick.select(keep)
    print(f"sweep middle-band rows after dedup: {len(sweep_pick)}")

    combined = concatenate_datasets([train_ds, sweep_pick])
    print(f"combined (train+sweep): {len(combined)}")
    filtered = combined.filter(lambda r: r["repo"] not in eval_repos, num_proc=args.num_proc)
    print(f"after eval-repo exclusion: {len(filtered)}  (dropped {len(combined) - len(filtered)})")

    shuffled = filtered.shuffle(seed=args.seed)
    print(f"shuffled with seed={args.seed}")

    os.makedirs(out_path, exist_ok=True)
    DatasetDict({"train": shuffled}).save_to_disk(str(out_path))
    print(f"\nsaved to: {out_path}")
    for root, _dirs, files in os.walk(out_path):
        for n in sorted(files):
            fp = os.path.join(root, n)
            print(f"  {fp}  ({os.path.getsize(fp)} bytes)")


if __name__ == "__main__":
    main()
