#!/usr/bin/env python3
"""Parallel per-batch eldest-age analyzer that cross-checks json wv vs
timeline-derived wv. Designed to run on the cluster (Lustre) directly.

Usage: python3 analyze_weight_versions_v2.py <RUN_DIR> [OUT_FILE]
"""
import glob
import json
import multiprocessing as mp
import os
import re
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone

RUN = sys.argv[1].rstrip("/")
OUT = sys.argv[2] if len(sys.argv) > 2 else None
LOG = f"{RUN}/_logs/training.log"

TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
UPDATE_END_RE = re.compile(r"Timer update_weights end")
BATCH_START_RE = re.compile(r"generate_rollout START rollout_id=(\d+)")

update_ends = []
batch_starts = {}

with open(LOG, "rb") as f:
    for raw in f:
        line = raw.decode("utf-8", errors="ignore")
        m = TS_RE.search(line)
        if not m:
            continue
        dt = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
        if UPDATE_END_RE.search(line):
            update_ends.append(dt)
        bs = BATCH_START_RE.search(line)
        if bs:
            batch_starts[int(bs.group(1))] = dt


def timeline_version(ts_dt):
    return bisect_right(update_ends, ts_dt)


def parse_trace(trace_file):
    parts = trace_file.split("/")
    inst = parts[-6]
    trial = parts[-2]
    K = int(parts[-7].split("_")[1])
    try:
        with open(trace_file) as f:
            msgs = json.load(f)
    except Exception:
        return None
    if not isinstance(msgs, list):
        return None
    turns = []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        ex = m.get("extra") or {}
        wv = ex.get("weight_version")
        ts = ex.get("timestamp")
        if wv is None or ts is None:
            continue
        try:
            wv_int = int(wv)
        except (TypeError, ValueError):
            continue
        turns.append((float(ts), wv_int))
    if not turns:
        return None
    wvs = [t[1] for t in turns]
    return {
        "K": K, "inst": inst, "trial": trial,
        "n_turns": len(turns),
        "first_wv": wvs[0], "last_wv": wvs[-1], "min_wv": min(wvs),
        "distinct_wv_count": len(set(wvs)),
        "first_ts": turns[0][0], "last_ts": turns[-1][0],
    }


trace_files = sorted(glob.glob(
    f"{RUN}/naive_out/rollout_*/*/*/task-*/rollouts/rollout_*/messages_raw.json"
))
print(f"# discovered_trace_files={len(trace_files)}", flush=True)

with mp.Pool(processes=32) as pool:
    raw_results = pool.map(parse_trace, trace_files, chunksize=8)
records = [r for r in raw_results if r is not None]

for r in records:
    ts_first = datetime.fromtimestamp(r["first_ts"], tz=timezone.utc)
    ts_last = datetime.fromtimestamp(r["last_ts"], tz=timezone.utc)
    r["v_timeline_first"] = timeline_version(ts_first)
    r["v_timeline_last"] = timeline_version(ts_last)


def emit(s, fh=None):
    print(s, flush=True)
    if fh is not None:
        print(s, file=fh)


fh = open(OUT, "w") if OUT else None
emit(f"# RUN={RUN}", fh)
emit(f"# update_weights_end={len(update_ends)} v0={update_ends[0].isoformat()}", fh)
emit(f"# batches_trained={max(0, len(update_ends) - 1)}", fh)
emit(f"# trace_records_parsed={len(records)} / files={len(trace_files)}", fh)
emit("", fh)

constant = sum(1 for r in records if r["distinct_wv_count"] == 1)
emit(f"## constant-wv-within-trace: {constant}/{len(records)} "
     f"({100.0 * constant / max(len(records), 1):.1f}%)", fh)
hist = Counter(r["distinct_wv_count"] for r in records)
emit(f"## distinct-wv-per-trace histogram: {sorted(hist.items())}", fh)
emit("", fh)

emit("## (K, first_wv) cross-tab — count of traces per (K, first_wv):", fh)
ck = Counter((r["K"], r["first_wv"]) for r in records)
ks = sorted({r["K"] for r in records})
ws = sorted({r["first_wv"] for r in records})
emit("K\\wv\t" + "\t".join(str(w) for w in ws), fh)
for K in ks:
    row = [str(K)] + [str(ck.get((K, w), 0)) for w in ws]
    emit("\t".join(row), fh)
emit("", fh)

match = sum(1 for r in records if r["first_wv"] == r["v_timeline_first"])
off = Counter(r["first_wv"] - r["v_timeline_first"] for r in records)
emit(f"## json_first_wv vs timeline at first-turn ts: "
     f"match={match}/{len(records)}", fh)
emit(f"## (json_first_wv - timeline_v) histogram: {sorted(off.items())}", fh)
emit("", fh)

emit("## per-batch eldest-age trend", fh)
emit("# age_json   = K - first_wv (from messages_raw.json)", fh)
emit("# age_tlast  = K - timeline_v_at_first_ts (from training.log only)", fh)
emit("# stale_lag  = current_actor_wv - first_wv (the v2 metric this run logs)", fh)
emit("K\tn\tmax_age_json\tavg_age_json\tmax_age_tlast\tavg_age_tlast\t"
     "min_first_wv\tmax_first_wv\tn_pol_v=K\tn_first_wv=K+1", fh)
per_K = defaultdict(list)
for r in records:
    per_K[r["K"]].append(r)
for K in sorted(per_K):
    rs = per_K[K]
    age_j = [K - r["first_wv"] for r in rs]
    age_t = [K - r["v_timeline_first"] for r in rs]
    wv = [r["first_wv"] for r in rs]
    n_eq_K = sum(1 for w in wv if w == K)
    n_eq_K1 = sum(1 for w in wv if w == K + 1)
    emit(f"{K}\t{len(rs)}\t{max(age_j)}\t{sum(age_j)/len(age_j):.3f}\t"
         f"{max(age_t)}\t{sum(age_t)/len(age_t):.3f}\t"
         f"{min(wv)}\t{max(wv)}\t{n_eq_K}\t{n_eq_K1}", fh)

if fh:
    fh.close()
