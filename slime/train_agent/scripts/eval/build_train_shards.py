"""Shard v2_python1k_split_train (950 rows) into 19 HF DatasetDicts of 50 rows.

Output layout:
  <out_root>/shard_00/train/...
  <out_root>/shard_01/...
  ...
  <out_root>/shard_18/...

Each shard dir is a `datasets.DatasetDict({"train": Dataset(...)})`, matching
the v2_python1k_split_eval layout that the slurm script already consumes via
`load_from_disk(EVAL_DATA)["train"]` and `dump_specs_and_score.py $EVAL_DATA/train`.
"""

import os
import sys

from datasets import DatasetDict, load_from_disk

SRC = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/datasets/v2_python1k_split_train"
OUT_ROOT = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/datasets/v2_python1k_split_train_shards"
SHARD_SIZE = 50


def main():
    ds = load_from_disk(SRC)
    if "train" in ds:
        ds = ds["train"]
    n = len(ds)
    print(f"loaded {SRC}: {n} rows")
    n_shards = (n + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"writing {n_shards} shards of up to {SHARD_SIZE} rows under {OUT_ROOT}")
    os.makedirs(OUT_ROOT, exist_ok=True)
    for i in range(n_shards):
        lo, hi = i * SHARD_SIZE, min((i + 1) * SHARD_SIZE, n)
        shard = ds.select(range(lo, hi))
        out = os.path.join(OUT_ROOT, f"shard_{i:02d}")
        DatasetDict({"train": shard}).save_to_disk(out)
        print(f"  shard_{i:02d}: rows [{lo}, {hi})  size={len(shard)}  -> {out}")
    print("done.")


if __name__ == "__main__":
    main()
