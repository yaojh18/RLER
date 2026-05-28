"""Emit a TSV of instances in the GRPO-friendly middle band (1/4, 2/4, 3/4).

Scans pass@4 eval reports under <runs-root> for one or more shard families,
keeps instances with all 4 attempts reported, and writes those with
`pass_count in {1, 2, 3}` to a TSV: instance_id<TAB>pass_count<TAB>source_shard.

See agent/data_util/README.md for the full pipeline.
"""
import argparse
import glob
import json
import os
from collections import Counter


def ingest(runs_root: str, globpat: str, source_tag: str, inst: dict):
    for sd in sorted(glob.glob(os.path.join(runs_root, globpat))):
        sh = sd.split("-sh")[-1]
        for a in (1, 2, 3, 4):
            rp = f"{sd}/attempt-{a}/nebius-eval/eval_report.json"
            if not os.path.exists(rp):
                continue
            try:
                d = json.load(open(rp))
            except Exception as e:
                print(f"WARN {rp}: {e}")
                continue
            for it in d.get("items") or []:
                iid = it["instance_id"]
                if iid not in inst:
                    inst[iid] = {"atts": [None, None, None, None], "shard": f"{source_tag}-sh{sh}"}
                inst[iid]["atts"][a - 1] = bool(it.get("passed_match", False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-root",
        default="/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs",
        help="Lustre root holding the per-shard eval run dirs.",
    )
    p.add_argument(
        "--out-tsv",
        default="/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/datasets/v2_python_pass1to3of4.tsv",
    )
    p.add_argument(
        "--source",
        action="append",
        default=None,
        help="Repeatable. Format: tag=globpat. Defaults match the original 950+sweep run: "
             "'train=55*-evp4v2-9b-9b-baseline-train-sh*' and "
             "'pysweep=559*-evp4v2-9b-9b-baseline-pysweep-sh*'.",
    )
    args = p.parse_args()

    sources = args.source or [
        "train=55*-evp4v2-9b-9b-baseline-train-sh*",
        "pysweep=559*-evp4v2-9b-9b-baseline-pysweep-sh*",
    ]

    inst: dict = {}
    for src in sources:
        tag, _, glob_pat = src.partition("=")
        if not glob_pat:
            raise SystemExit(f"--source must be tag=globpat, got: {src!r}")
        ingest(args.runs_root, glob_pat, tag, inst)

    dist = Counter()
    middle = []
    for iid, rec in inst.items():
        atts = rec["atts"]
        n_known = sum(1 for x in atts if x is not None)
        if n_known < 4:
            continue
        n_pass = sum(1 for x in atts if x is True)
        dist[n_pass] += 1
        if n_pass in (1, 2, 3):
            middle.append((iid, n_pass, rec["shard"]))

    middle.sort(key=lambda r: (r[1], r[0]))

    print(f"instances with all 4 attempts known: {sum(dist.values())}")
    for k in (0, 1, 2, 3, 4):
        print(f"  {k}/4 : {dist[k]}")
    print(f"middle-band (1-3 of 4): {len(middle)}")
    print(f"writing -> {args.out_tsv}")

    os.makedirs(os.path.dirname(args.out_tsv), exist_ok=True)
    with open(args.out_tsv, "w") as f:
        f.write("instance_id\tpass_count\tsource_shard\n")
        for iid, n_pass, sh in middle:
            f.write(f"{iid}\t{n_pass}\t{sh}\n")
    print(f"done. wrote {len(middle)} lines.")


if __name__ == "__main__":
    main()
