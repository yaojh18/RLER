#!/usr/bin/env python3
"""Log-only reconstruction of per-batch eldest-age trend that models the
`_drop_stale_buffer` policy in slime/train_agent/collect_naive_rollout_async.py:705.

At runtime each generate_rollout(K) call:
  - keeps only buffered groups whose submitter rollout_id ∈ {K-1, K}
    (line 707 — `kept = [g for g in _BUFFER if g.rollout_id >= current - 1]`),
  - shuffles + slices 16 from the survivors (line 712 — `_pop_groups`),
so the runtime cap on per-batch eldest age is 1 (in wv units). Any group whose
submitter is older — i.e., that completes after generate_rollout(K+2) starts —
is silently dropped.

The v1 of this script naively partitioned all completed groups in completion
order into chunks of 16 → step K (FIFO assumption). That over-reported
staleness because most "late completers" are dropped at runtime, not consumed
many steps later.

This rewrite:
  (a) buckets every trace into a submitter window using the
      `[naive-async] generate_rollout START rollout_id=K` markers; submitter_K
      is the most recent batch start before the trace's approx. start time
      (= done_time - dt);
  (b) caps consumer batch to {submitter_K, submitter_K + 1} (the drop-stale
      window) and emits a per-step max/avg eldest-age trend over the eligible
      kept buffer;
  (c) reports per-batch drop counts: groups whose first-turn time lands after
      generate_rollout(submitter_K + 2) starts (so the drop-stale filter
      evicts them before training ever sees them).

Caveat — without per-turn timestamps in the log, `eldest_wv` here is
approximated as wv_at(done_time - dt) (i.e., wv at the trace's worker-side
start, BEFORE docker setup). The real first-turn time is later by setup time
(usually 30-60 s, sometimes crossing a broadcast boundary). So the log-only
numbers can over-estimate age by ~1 vs the JSON ground truth (the v2 wandb
metric `current_actor_wv - first_turn_wv` computed from per-turn extras).
For runs that dump per-trial `messages_raw.json` use a JSON-based analyzer
(see slime/train_agent/scripts/analyze_weight_versions.py).

Inputs (positional): training.log path
Outputs (stdout): header + per-batch trend table + per-submitter drop table.
"""
import re
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta

LOG = sys.argv[1]

TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
UPDATE_END_RE = re.compile(r"Timer update_weights end")
BATCH_START_RE = re.compile(r"generate_rollout START rollout_id=(\d+)")
DONE_RE = re.compile(
    r"INFO:swe_agent\.naive_search:\[([^\]]+)\] naive r=(\d+) done dt=([\d.]+)s"
)

BATCH_GROUPS = 16     # min_ready_groups for slime async naive (and code default)
TRIALS_PER_GROUP = 8  # m=8 trials per instance

ts_linenos = []
ts_values = []
update_ends = []
batch_start_ks = []
batch_start_ts = []
done_events = []

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
                batch_start_ks.append(int(bs.group(1)))
                batch_start_ts.append(dt)
        ds = DONE_RE.search(line)
        if ds:
            done_events.append((i, ds.group(1), int(ds.group(2)), float(ds.group(3))))

assert ts_linenos, "no timestamped lines"
assert update_ends, "no Timer update_weights end events"
assert batch_start_ts, "no generate_rollout START events"
assert done_events, "no naive r=N done events"

# Sort batch starts by time (they should already be in order).
order = sorted(range(len(batch_start_ts)), key=lambda i: batch_start_ts[i])
batch_start_ts = [batch_start_ts[i] for i in order]
batch_start_ks = [batch_start_ks[i] for i in order]
# Lookup: K -> generate_rollout_start time.
batch_start_by_k = dict(zip(batch_start_ks, batch_start_ts))


def time_for_line(lineno: int) -> datetime:
    idx = bisect_right(ts_linenos, lineno) - 1
    if idx < 0:
        return ts_values[0]
    if idx + 1 < len(ts_linenos):
        prev_dist = lineno - ts_linenos[idx]
        next_dist = ts_linenos[idx + 1] - lineno
        if next_dist < prev_dist:
            return ts_values[idx + 1]
    return ts_values[idx]


def submitter_K_at(start_ts: datetime) -> int:
    """rollout_id active at submission time (collect_naive_rollout_async.py:566)."""
    i = bisect_right(batch_start_ts, start_ts) - 1
    if i < 0:
        return batch_start_ks[0]
    return batch_start_ks[i]


def wv_at(ts: datetime) -> int:
    """1-indexed SGLang wv at time ts: 1 = after initial broadcast, 2 = after step 0."""
    return bisect_right(update_ends, ts)


# Per-trace records. start_time = end_time - dt (worker-side rollout start).
traces = []
for (lineno, inst, r, dt_s) in done_events:
    end_time = time_for_line(lineno)
    start_time = end_time - timedelta(seconds=dt_s)
    traces.append({
        "instance": inst, "r": r, "lineno": lineno,
        "start": start_time, "end": end_time,
        "submitter_K": submitter_K_at(start_time),
        "first_wv_est": max(wv_at(start_time), 1),
    })

# Chunk per-instance trace lists into groups of 8 in line order.
inst_traces = defaultdict(list)
for t in traces:
    inst_traces[t["instance"]].append(t)

groups = []
for inst, ts in inst_traces.items():
    ts.sort(key=lambda x: x["lineno"])
    for i in range(0, len(ts), TRIALS_PER_GROUP):
        chunk = ts[i:i + TRIALS_PER_GROUP]
        if len(chunk) < TRIALS_PER_GROUP:
            continue
        sk = Counter(t["submitter_K"] for t in chunk).most_common(1)[0][0]
        groups.append({
            "instance": inst,
            "traces": chunk,
            "group_end": max(t["end"] for t in chunk),
            "submitter_K": sk,
            "eldest_wv": min(t["first_wv_est"] for t in chunk),
        })

n_trained_batches = max(0, len(update_ends) - 1)
sub_hist = Counter(g["submitter_K"] for g in groups)

print(f"# log_path={LOG}")
print(f"# update_weights_end_events={len(update_ends)} "
      f"(wv=1 broadcast end @ {update_ends[0].isoformat()})")
print(f"# generate_rollout_start_events={len(batch_start_ks)} "
      f"K range = {min(batch_start_ks)}..{max(batch_start_ks)}")
print(f"# trace_done_events={len(traces)} "
      f"completed_groups={len(groups)} "
      f"batches_trained={n_trained_batches}")
print(f"# BATCH_GROUPS={BATCH_GROUPS}  TRIALS_PER_GROUP={TRIALS_PER_GROUP}")
print()
print("## groups_by_submitter_K (count of completed groups per submitter window):")
print("submitter_K\tn_groups_completed")
for k in sorted(sub_hist):
    print(f"{k}\t{sub_hist[k]}")
print()

# Per-batch trend with drop-stale modelling.
# Batch K's eligible buffer = groups with submitter ∈ {K-1, K} that
# completed BEFORE generate_rollout(K) starts. (Groups completing during
# batch K's call are also eligible but we don't model that fine-grained
# arrival.) The runtime picks 16 uniformly at random from this set
# (random.shuffle in _pop_groups); the per-batch eldest age is the MAX age
# across the kept buffer (worst case actually consumed) and AVG is the
# expectation under uniform sampling.
print("## per-batch trend (max/avg eldest age across kept buffer for batch K)")
print("# age per group = (K + 1) - eldest_wv  (matches v2 wandb metric "
      "current_actor_wv - first_turn_wv)")
print("# n_kept = groups eligible at batch K (submitter ∈ {K-1, K},"
      " completed before generate_rollout START for K)")
print("# n_consumed = min(BATCH_GROUPS, n_kept)")
print("K\tn_kept\tn_consumed\tmax_age\tavg_age")
for K in range(n_trained_batches):
    cutoff = batch_start_by_k.get(K)
    if cutoff is None:
        continue
    eligible = [
        g for g in groups
        if g["submitter_K"] in (K - 1, K) and g["group_end"] <= cutoff
    ]
    if not eligible:
        print(f"{K}\t0\t0\t-\t-")
        continue
    ages = [(K + 1) - g["eldest_wv"] for g in eligible]
    print(f"{K}\t{len(eligible)}\t{min(len(eligible), BATCH_GROUPS)}\t"
          f"{max(ages)}\t{sum(ages)/len(ages):.3f}")

# Drop accounting: a group is "dropped" by _drop_stale_buffer if it
# completes after generate_rollout(submitter_K + 2) starts — by then the
# filter `rollout_id >= K-1` evicts it (since submitter_K < (K+2)-1).
print()
print("## stale-drop accounting (per submitter_K):")
print("# A group submitted in rollout K is consumable by batches K or K+1.")
print("# If group.completion_time > generate_rollout(K+2).start_ts, it is")
print("# dropped by _drop_stale_buffer(K+2).")
print("submitter_K\tn_completed\tn_dropped_by_filter\tdrop_pct")
total_dropped = 0
total_completed = 0
for k in sorted(sub_hist):
    drop_cutoff = batch_start_by_k.get(k + 2)
    if drop_cutoff is None:
        # Can't determine — batch K+2 hasn't started yet in the log window.
        completed = sub_hist[k]
        total_completed += completed
        print(f"{k}\t{completed}\t-\t- (K+2 not in log)")
        continue
    completed_in_k = [g for g in groups if g["submitter_K"] == k]
    dropped = [g for g in completed_in_k if g["group_end"] > drop_cutoff]
    n_done = len(completed_in_k)
    n_drop = len(dropped)
    total_dropped += n_drop
    total_completed += n_done
    pct = (100.0 * n_drop / n_done) if n_done else 0.0
    print(f"{k}\t{n_done}\t{n_drop}\t{pct:.1f}%")
print(f"# total_dropped={total_dropped}/{total_completed} groups "
      f"({100.0 * total_dropped / max(total_completed, 1):.1f}%) "
      f"≈ {total_dropped * TRIALS_PER_GROUP} traces wasted")
