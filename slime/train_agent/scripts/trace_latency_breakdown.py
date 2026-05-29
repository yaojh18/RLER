#!/usr/bin/env python3
"""Per-trace end-to-end latency breakdown for a slime naive-async RL run.

Decomposes each trace into these wall-clock phases:

  setup    docker run + initial environment setup
           = naive_done_dt (from log) - generation_dt (from messages_raw)
  gen      first-LLM-completion -> last-LLM-completion (= generation + tool exec)
           = last_turn.extra.timestamp - first_turn.extra.timestamp
  eval     last-LLM-completion -> gt score returned
           = gt_done dt= from training.log (naive_search.py:705)
  total    naive_done_dt + eval_dt
  (harvest = gt_done -> _BUFFER.append: not directly logged per trace; the
            `_harvest_ready` polling loop adds groups in batches every few
            seconds. Aggregate gap visible as buffer=N transitions in log.)

Inputs: <RUN_DIR> [OUT_FILE]
Outputs: percentile table + per-phase distribution summary.
"""
import glob
import json
import multiprocessing as mp
import os
import re
import sys
from collections import defaultdict
from statistics import mean, median

RUN = sys.argv[1].rstrip("/")
OUT = sys.argv[2] if len(sys.argv) > 2 else None
LOG = f"{RUN}/_logs/training.log"

NAIVE_DONE_RE = re.compile(
    r"INFO:swe_agent\.naive_search:\[([^\]]+)\] naive r=(\d+) done dt=([\d.]+)s"
)
GT_DONE_RE = re.compile(
    r"INFO:swe_agent\.naive_search:\[([^\]]+)\] gt_done "
    r"node=([^\s]+) dt=([\d.]+)s reward=([-\d.]+)"
)

naive_done = {}   # (inst, r) -> dt seconds
gt_done = {}      # node_id  -> dt seconds

with open(LOG, "rb") as f:
    for raw in f:
        line = raw.decode("utf-8", errors="ignore")
        m = NAIVE_DONE_RE.search(line)
        if m:
            naive_done[(m.group(1), int(m.group(2)))] = float(m.group(3))
        m = GT_DONE_RE.search(line)
        if m:
            gt_done[m.group(2)] = float(m.group(3))


def parse_trial(trial_dir: str) -> dict | None:
    parts = trial_dir.split("/")
    # naive_out / rollout_KKKK / INST / TIMESTAMP / task-N / rollouts / rollout_NN
    inst = parts[-5]
    trial_r = int(parts[-1].split("_")[1])
    K = int(parts[-6].split("_")[1])
    try:
        with open(f"{trial_dir}/summary.json") as f:
            summary = json.load(f)
        with open(f"{trial_dir}/messages_raw.json") as f:
            msgs = json.load(f)
    except Exception:
        return None
    asst_ts = []
    for m in msgs if isinstance(msgs, list) else []:
        if m.get("role") != "assistant":
            continue
        ex = m.get("extra") or {}
        ts = ex.get("timestamp")
        if ts is None:
            continue
        try:
            asst_ts.append(float(ts))
        except (TypeError, ValueError):
            continue
    if not asst_ts:
        return None
    first_ts, last_ts = asst_ts[0], asst_ts[-1]
    gen_dt = last_ts - first_ts
    n_asst = len(asst_ts)
    node_id = summary.get("node_id", "")
    started_at = summary.get("started_at")
    finished_at = summary.get("finished_at")
    perf_dt = (finished_at - started_at) if (started_at and finished_at) else None
    nd_dt = naive_done.get((inst, trial_r))
    gt_dt = gt_done.get(node_id)
    setup_dt = (nd_dt - gen_dt) if nd_dt is not None else None
    return {
        "K": K, "inst": inst, "r": trial_r, "node_id": node_id,
        "n_asst_turns": n_asst,
        "setup_dt": setup_dt,
        "gen_dt": gen_dt,
        "eval_dt": gt_dt,
        "naive_done_dt": nd_dt,
        "perf_total_dt": perf_dt,
    }


trial_dirs = sorted(glob.glob(
    f"{RUN}/naive_out/rollout_*/*/*/task-*/rollouts/rollout_*"
))


def emit(s: str, fh=None):
    print(s, flush=True)
    if fh is not None:
        print(s, file=fh)


fh = open(OUT, "w") if OUT else None
emit(f"# RUN={RUN}", fh)
emit(f"# trial_dirs_discovered={len(trial_dirs)}", fh)
emit(f"# naive_done_log_events={len(naive_done)} gt_done_log_events={len(gt_done)}", fh)

with mp.Pool(processes=32) as pool:
    raw = pool.map(parse_trial, trial_dirs, chunksize=8)
records = [r for r in raw if r is not None]
emit(f"# trials_parsed={len(records)}", fh)
emit("", fh)


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    idx = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
    return xs[idx]


def summarize(name: str, xs: list[float]):
    xs = [x for x in xs if x is not None and x >= 0]
    if not xs:
        emit(f"## {name}: no data", fh)
        return
    emit(f"## {name} (n={len(xs)})", fh)
    emit(f"#   mean={mean(xs):.1f}s  median={median(xs):.1f}s  min={min(xs):.1f}s  max={max(xs):.1f}s", fh)
    emit(f"#   p10={pct(xs, 0.10):.1f}  p25={pct(xs, 0.25):.1f}  "
         f"p50={pct(xs, 0.50):.1f}  p75={pct(xs, 0.75):.1f}  "
         f"p90={pct(xs, 0.90):.1f}  p95={pct(xs, 0.95):.1f}  p99={pct(xs, 0.99):.1f}", fh)


summarize("setup (docker run + initial env)  [s]", [r["setup_dt"] for r in records])
summarize("gen (first LLM turn -> last LLM turn)  [s]", [r["gen_dt"] for r in records])
summarize("eval (last LLM turn -> gt score)  [s]", [r["eval_dt"] for r in records])
summarize("naive_done = setup + gen (from log)  [s]",
          [r["naive_done_dt"] for r in records])
summarize("perf_total (summary.json finished - started)  [s]",
          [r["perf_total_dt"] for r in records])
emit("", fh)

# Total e2e per trial = naive_done + eval. (harvest is not measurable per-trial.)
totals = []
for r in records:
    nd = r["naive_done_dt"]
    ev = r["eval_dt"]
    if nd is None or ev is None:
        continue
    totals.append(nd + ev)
summarize("e2e_per_trial = naive_done + eval (setup+gen+eval)  [s]", totals)
emit("", fh)

# Per-batch K aggregates (so user can see whether traces in late batches are
# slower, e.g. due to noisier policies producing more steps).
emit("## per-batch (submitter K) latency summary", fh)
emit("K\tn\tmean_total\tp50_total\tp95_total\tmean_setup\tmean_gen\tmean_eval", fh)
per_K = defaultdict(list)
for r in records:
    per_K[r["K"]].append(r)
for K in sorted(per_K):
    rs = per_K[K]
    tot = [(r["naive_done_dt"] or 0) + (r["eval_dt"] or 0)
           for r in rs if r["naive_done_dt"] and r["eval_dt"]]
    setup = [r["setup_dt"] for r in rs if r["setup_dt"] is not None]
    gen = [r["gen_dt"] for r in rs if r["gen_dt"] is not None]
    ev = [r["eval_dt"] for r in rs if r["eval_dt"] is not None]
    if not tot:
        continue
    emit(f"{K}\t{len(rs)}\t{mean(tot):.1f}\t{pct(tot, 0.5):.1f}\t{pct(tot, 0.95):.1f}\t"
         f"{mean(setup) if setup else 0:.1f}\t{mean(gen) if gen else 0:.1f}\t"
         f"{mean(ev) if ev else 0:.1f}", fh)

# Coverage stats — how many trials are missing log matches.
emit("", fh)
emit(f"## coverage:", fh)
emit(f"#   trials with naive_done log match: "
     f"{sum(1 for r in records if r['naive_done_dt'] is not None)}/{len(records)}", fh)
emit(f"#   trials with gt_done log match:    "
     f"{sum(1 for r in records if r['eval_dt'] is not None)}/{len(records)}", fh)
emit(f"#   note: harvest phase (gt_done -> _BUFFER.append) is not logged per", fh)
emit(f"#         trial; collect_naive_rollout_async._harvest_ready batches", fh)
emit(f"#         additions during the WAITING loop.", fh)

if fh:
    fh.close()
