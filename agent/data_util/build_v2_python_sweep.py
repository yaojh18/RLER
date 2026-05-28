"""Build python sweep shards from nebius/SWE-rebench-v2.

Loads the upstream HF dataset, keeps `language == python`, excludes any
instance_id already covered by an existing 1k subset, drops rows with empty
`image_name` (unscorable), sorts newest-first by `created_at`, and writes
fixed-size DatasetDict shards to disk for batched pass@k evaluation.

See agent/data_util/README.md for the full pipeline.
"""
import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

from datasets import load_dataset, DatasetDict


def parse_dt(s: str) -> datetime:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime(1970, 1, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base",
        default="/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench",
        help="Lustre root for swe-rebench artifacts.",
    )
    p.add_argument(
        "--hf-dataset",
        default="nebius/SWE-rebench-v2",
        help="HuggingFace dataset id to load.",
    )
    p.add_argument("--language", default="python")
    p.add_argument(
        "--exclude-ids-file",
        default=None,
        help="Optional text file with one instance_id per line to exclude. "
             "Defaults to <base>/datasets/v2_python1k/instance_ids.txt.",
    )
    p.add_argument(
        "--out-root",
        default=None,
        help="Output dir for shard_NN subdirs. "
             "Defaults to <base>/datasets/v2_python_sweep_shards.",
    )
    p.add_argument("--shard-size", type=int, default=400)
    p.add_argument("--num-proc", type=int, default=8)
    args = p.parse_args()

    base = Path(args.base)
    out_root = Path(args.out_root) if args.out_root else base / "datasets" / "v2_python_sweep_shards"
    excl_file = Path(args.exclude_ids_file) if args.exclude_ids_file else base / "datasets" / "v2_python1k" / "instance_ids.txt"
    os.environ.setdefault("HF_HOME", str(base / "hf-cache"))

    print(f"[{datetime.now().isoformat(timespec='seconds')}] loading {args.hf_dataset} ...", flush=True)
    ds = load_dataset(args.hf_dataset, split="train")
    print(f"  total rows: {len(ds)}")

    ds = ds.filter(lambda r: r["language"] == args.language, num_proc=args.num_proc)
    print(f"  {args.language} rows: {len(ds)}")

    excluded = set()
    if excl_file.exists():
        excluded = set(excl_file.read_text().strip().split("\n"))
    print(f"  loaded {len(excluded)} excluded ids from {excl_file}")
    ds = ds.filter(lambda r: r["instance_id"] not in excluded, num_proc=args.num_proc)
    print(f"  after excluding base set: {len(ds)}")

    n_before = len(ds)
    ds = ds.filter(lambda r: bool(r.get("image_name")), num_proc=args.num_proc)
    print(f"  dropped {n_before - len(ds)} rows with empty image_name; remaining: {len(ds)}")

    order = sorted(range(len(ds)), key=lambda i: parse_dt(ds["created_at"][i]), reverse=True)
    ds = ds.select(order)
    print(f"  newest: {ds[0]['created_at']}  oldest: {ds[-1]['created_at']}")

    n = len(ds)
    n_shards = (n + args.shard_size - 1) // args.shard_size
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"  writing {n_shards} shards of up to {args.shard_size} rows under {out_root}")
    for i in range(n_shards):
        lo, hi = i * args.shard_size, min((i + 1) * args.shard_size, n)
        sub = ds.select(range(lo, hi))
        out = out_root / f"shard_{i:02d}"
        if out.exists():
            shutil.rmtree(out)
        DatasetDict({"train": sub}).save_to_disk(str(out))
        print(f"    shard_{i:02d}: rows [{lo},{hi}) size={len(sub)} -> {out}")
    print(f"[{datetime.now().isoformat(timespec='seconds')}] done.")


if __name__ == "__main__":
    main()
