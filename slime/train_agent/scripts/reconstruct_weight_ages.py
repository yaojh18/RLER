#!/usr/bin/env python3
"""Reconstruct per-batch eldest-age trend from a 58063-style RLER training.log.

Inputs (positional): training.log path
Outputs (stdout):
  <header>
  step \t consumed_groups \t consumed_traces \t max_eldest_age \t avg_eldest_age
"""
import re
import sys
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta

LOG = sys.argv[1]

TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
UPDATE_END_RE = re.compile(r"Timer update_weights end")
BATCH_START_RE = re.compile(r"generate_rollout START rollout_id=(\d+)")
DONE_RE = re.compile(
    r"INFO:swe_agent\.naive_search:\[([^\]]+)\] naive r=(\d+) done dt=([\d.]+)s"
)

ts_linenos = []
ts_values = []
update_ends = []
batch_starts = {}
done_events = []  # (lineno, instance_id, r, dt_s)

with open(LOG, "rb") as f:
    for i, raw in enumerate(f):
        line = raw.decode("utf-8", errors="ignore")
        m = TS_RE.search(line)
        if m:
            dt = datetime.fromisoformat(m.group(1))
            ts_linenos.append(i)
            ts_values.append(dt)
            if UPDATE_END_RE.search(line):
                update_ends.append(dt)
            bs = BATCH_START_RE.search(line)
            if bs:
                batch_starts[int(bs.group(1))] = dt
        ds = DONE_RE.search(line)
        if ds:
            done_events.append((i, ds.group(1), int(ds.group(2)), float(ds.group(3))))

assert ts_linenos, "no timestamped lines found"
assert update_ends, "no weight-update events found"
assert done_events, "no trace done events found"


def time_for_line(lineno: int) -> datetime:
    """Nearest timestamp by line position (preceding preferred, else following)."""
    idx = bisect_right(ts_linenos, lineno) - 1
    if idx < 0:
        return ts_values[0]
    if idx + 1 < len(ts_linenos):
        prev_dist = lineno - ts_linenos[idx]
        next_dist = ts_linenos[idx + 1] - lineno
        if next_dist < prev_dist:
            return ts_values[idx + 1]
    return ts_values[idx]


# Per-trace start time and version-at-start.
# version is index into update_ends: 0 == initial broadcast, 1 == after step 0, ...
traces = []
for (lineno, inst, r, dt_s) in done_events:
    end_time = time_for_line(lineno)
    start_time = end_time - timedelta(seconds=dt_s)
    v = bisect_right(update_ends, start_time) - 1
    v = max(v, 0)
    traces.append({
        "instance": inst, "r": r, "lineno": lineno,
        "start": start_time, "end": end_time, "v_start": v,
    })

# Group traces by (instance_id, occurrence-of-r). Each instance can have
# multiple groups across batches; chunk its r-events into successive groups of
# 8 in line order. r-index is per-group so we use position, not r-value.
inst_traces = defaultdict(list)
for t in traces:
    inst_traces[t["instance"]].append(t)

groups = []
for inst, ts in inst_traces.items():
    ts.sort(key=lambda x: x["lineno"])
    for i in range(0, len(ts), 8):
        chunk = ts[i:i+8]
        if len(chunk) < 8:
            continue
        groups.append({
            "instance": inst,
            "traces": chunk,
            "group_end": max(t["end"] for t in chunk),
        })

groups.sort(key=lambda g: g["group_end"])

# Number of fully-trained batches = #update_weights_end events - 1
# (the first update_weights end is the pre-training initial broadcast).
n_trained_batches = max(0, len(update_ends) - 1)
BATCH_GROUPS = 16  # min_ready_groups for this run (per slurm script / config)

n_groups_consumed = min(len(groups) // BATCH_GROUPS, n_trained_batches) * BATCH_GROUPS

print(f"# log_path={LOG}")
print(f"# update_weights_end_events={len(update_ends)} "
      f"(v0=initial broadcast @ {update_ends[0].isoformat()})")
print(f"# generate_rollout_start_events={len(batch_starts)} "
      f"first={batch_starts.get(0)}")
print(f"# total_trace_done_events={len(traces)} "
      f"total_completed_groups={len(groups)} "
      f"batches_trained={n_trained_batches} "
      f"groups_consumed_for_trend={n_groups_consumed} "
      f"(BATCH_GROUPS={BATCH_GROUPS})")
print("# step = training batch index K. Eldest age per trace = K - v_start.")
print("step\tconsumed_groups\tconsumed_traces\tmax_eldest_age\tavg_eldest_age")

for K in range(n_trained_batches):
    s, e = K * BATCH_GROUPS, (K + 1) * BATCH_GROUPS
    bk_groups = groups[s:e]
    if len(bk_groups) < BATCH_GROUPS:
        break
    ages = []
    for g in bk_groups:
        for t in g["traces"]:
            ages.append(K - t["v_start"])
    print(f"{K}\t{len(bk_groups)}\t{len(ages)}\t{max(ages)}\t{sum(ages)/len(ages):.3f}")
