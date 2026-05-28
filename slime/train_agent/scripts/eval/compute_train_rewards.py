"""Recompute soft / joint / f2p_only rewards for the 950-instance Qwen3.5-9B
TRAIN baseline (jids 55436 + 55437-55454) using the same reward formulae as
RLER/agent/swe_agent/run/run_swe_agent.py::make_evaluation_payload.

We do NOT re-run the docker harness — eval_report.json + specs.json already
contain enough state (per-instance F2P-passed list, P2P-broken list, plus the
expected F2P/P2P sets). We just route those through make_evaluation_payload.

Output:
  - per-attempt distribution per scheme (mean, std, histogram)
  - per-instance pass@4 distribution (mean of max reward over 4 attempts)
  - per-instance pass-count histogram (# of attempts with reward > 0.5)
"""
from __future__ import annotations
import json
import os
import statistics
import sys
from pathlib import Path
from collections import defaultdict, Counter

# Inlined from RLER/agent/swe_agent/run/run_swe_agent.py to avoid pulling
# the full package (which imports dotenv etc.). Keep in sync with that file.
EMPTY_REWARD = 0.0
ERROR_REWARD = 0.0
SUPPORTED_REWARD_SCHEMES = {"soft", "joint", "f2p_only"}


def make_evaluation_payload(
    status,
    passed_tests=None,
    failed_tests=None,
    pass_to_pass_expected=None,
    fail_to_pass_expected=None,
):
    if status not in {"resolved", "unresolved", "empty", "error"}:
        raise ValueError(f"Unknown evaluation status: {status}")
    passed = sorted({str(t) for t in (passed_tests or []) if str(t)}) if status in {"resolved", "unresolved"} else []
    failed = sorted({str(t) for t in (failed_tests or []) if str(t)}) if status in {"resolved", "unresolved"} else []
    scheme = os.environ.get("RLER_REWARD_SCHEME", "soft").strip().lower() or "soft"
    if scheme not in SUPPORTED_REWARD_SCHEMES:
        raise ValueError(f"Unsupported RLER_REWARD_SCHEME={scheme!r}")
    if status == "empty":
        reward = EMPTY_REWARD
    elif status == "error":
        reward = ERROR_REWARD
    elif scheme == "soft":
        total = len(set(passed) | set(failed))
        reward = len(set(passed)) / total if total else (1.0 if status == "resolved" else 0.0)
    else:
        passed_set = set(passed)
        f2p = {str(t) for t in (fail_to_pass_expected or []) if str(t)}
        p2p = {str(t) for t in (pass_to_pass_expected or []) if str(t)}
        f2p_frac = len(passed_set & f2p) / len(f2p) if f2p else 1.0
        p2p_frac = len(passed_set & p2p) / len(p2p) if p2p else 1.0
        reward = f2p_frac * p2p_frac if scheme == "joint" else f2p_frac
    return {"reward": float(reward), "status": status}

RUN_ROOT = Path("/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs")
SHARD_PREFIX = "evp4v2-9b-9b-baseline-train-sh"
ATTEMPTS = (1, 2, 3, 4)
SCHEMES = ("soft", "joint", "f2p_only")

# jid -> shard idx; we built this from the launch log earlier.
TRAIN_JIDS = {
    55436: 0, 55437: 1, 55438: 2, 55439: 3, 55440: 4, 55441: 5, 55442: 6,
    55443: 7, 55444: 8, 55445: 9, 55446: 10, 55447: 11, 55448: 12, 55449: 13,
    55450: 14, 55451: 15, 55452: 16, 55453: 17, 55454: 18,
}


def load_attempt(run_dir: Path, attempt: int):
    """Return (specs_by_id, report_by_id, preds_by_id) for one attempt, or None."""
    a = run_dir / f"attempt-{attempt}"
    ne = a / "nebius-eval"
    specs_p = ne / "specs.json"
    rep_p = ne / "eval_report.json"
    preds_p = a / "preds.json"
    if not (specs_p.exists() and rep_p.exists() and preds_p.exists()):
        return None
    specs = json.loads(specs_p.read_text())
    report = json.loads(rep_p.read_text())
    preds = json.loads(preds_p.read_text())
    spec_by_id = {s["instance_id"]: s for s in specs}
    item_by_id = {it["instance_id"]: it for it in report.get("items", [])}
    pred_by_id = {iid: r for iid, r in preds.items()}
    return spec_by_id, item_by_id, pred_by_id


def compute_one(spec: dict, item: dict | None, pred: dict | None) -> dict[str, float]:
    """Return {scheme: reward} for a single (instance, attempt)."""
    f2p = list(spec.get("FAIL_TO_PASS") or [])
    p2p = list(spec.get("PASS_TO_PASS") or [])
    patch = (pred or {}).get("model_patch") or ""
    if not patch.strip():
        status = "empty"
        passed = []
        failed = []
    elif item is None:
        status = "error"
        passed = []
        failed = []
    else:
        f2p_passed = set(item.get("from_fail_to_pass") or [])
        p2p_failed = set(item.get("failed_from_pass_to_pass") or [])
        p2p_passed = set(p2p) - p2p_failed
        passed = sorted(f2p_passed | p2p_passed)
        failed = sorted((set(f2p) - f2p_passed) | p2p_failed)
        status = "resolved" if bool(item.get("passed_match")) else "unresolved"

    out = {}
    for sch in SCHEMES:
        os.environ["RLER_REWARD_SCHEME"] = sch
        out[sch] = make_evaluation_payload(
            status=status,
            passed_tests=passed,
            failed_tests=failed,
            pass_to_pass_expected=p2p,
            fail_to_pass_expected=f2p,
        )["reward"]
    return out, status


def histogram(values, bins=(0.0, 0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999, 1.001)):
    """Inclusive-lower, exclusive-upper. Last bin includes 1.0."""
    labels = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo == 0.0 and hi == 0.001:
            labels.append("=0")
        elif lo == 0.999 and hi == 1.001:
            labels.append("=1")
        else:
            labels.append(f"[{lo:.2f},{hi:.2f})")
    counts = [0] * (len(bins) - 1)
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break
    return labels, counts


def summarize(name: str, values: list[float]):
    n = len(values)
    if n == 0:
        print(f"  {name}: n=0")
        return
    mean = sum(values) / n
    std = statistics.pstdev(values) if n > 1 else 0.0
    nz = sum(1 for v in values if v > 1e-9)
    perfect = sum(1 for v in values if v > 0.999)
    print(f"  {name:<10} n={n:5d}  mean={mean:.4f}  std={std:.4f}  >0: {nz:5d} ({100*nz/n:5.1f}%)  =1: {perfect:5d} ({100*perfect/n:5.1f}%)")


def main():
    # Collect: rewards[scheme][instance_id] = list of 4 attempt rewards
    rewards = {s: defaultdict(lambda: [None, None, None, None]) for s in SCHEMES}
    # Per-instance, per-attempt: resolved flag (passed_match=True) and patch status
    resolved_flags = defaultdict(lambda: [None, None, None, None])
    # Per-instance: baseline_pass_rate = |P2P| / (|F2P| + |P2P|) from specs.json
    baseline_pass_rate = {}
    status_counter = defaultdict(Counter)  # status_counter[attempt][status] = count
    missing = []

    for jid, shard in sorted(TRAIN_JIDS.items()):
        run_dir = RUN_ROOT / f"{jid}-{SHARD_PREFIX}{shard:02d}"
        if not run_dir.exists():
            missing.append((jid, shard, "rundir missing"))
            continue
        for attempt in ATTEMPTS:
            data = load_attempt(run_dir, attempt)
            if data is None:
                missing.append((jid, shard, f"attempt-{attempt}"))
                continue
            spec_by_id, item_by_id, pred_by_id = data
            for iid, spec in spec_by_id.items():
                if iid not in baseline_pass_rate:
                    f2p_n = len(spec.get("FAIL_TO_PASS") or [])
                    p2p_n = len(spec.get("PASS_TO_PASS") or [])
                    denom = f2p_n + p2p_n
                    baseline_pass_rate[iid] = (p2p_n / denom) if denom else 0.0
                rew, status = compute_one(spec, item_by_id.get(iid), pred_by_id.get(iid))
                status_counter[attempt][status] += 1
                for sch in SCHEMES:
                    rewards[sch][iid][attempt - 1] = rew[sch]
                item = item_by_id.get(iid)
                resolved_flags[iid][attempt - 1] = bool(item.get("passed_match")) if item is not None else False

    print(f"\n== TRAIN cohort (950 instances, 19 shards × 4 attempts) ==")
    if missing:
        print(f"  MISSING ({len(missing)}):")
        for m in missing[:20]:
            print(f"    {m}")
        if len(missing) > 20:
            print(f"    ... +{len(missing) - 20} more")

    # Per-attempt status
    print("\n== Per-attempt status counts ==")
    for a in ATTEMPTS:
        c = status_counter[a]
        total = sum(c.values())
        print(f"  attempt-{a}: total={total}  " + "  ".join(f"{k}={v}({100*v/total:.1f}%)" for k, v in sorted(c.items())))

    # Per-attempt reward distribution per scheme
    print("\n== Per-attempt reward distributions ==")
    for sch in SCHEMES:
        print(f"\nscheme={sch}")
        for a in ATTEMPTS:
            vals = [r[a - 1] for r in rewards[sch].values() if r[a - 1] is not None]
            summarize(f"att-{a}", vals)

    # Per-instance pass@4 (max reward across 4 attempts)
    print("\n== Per-instance pass@4 (MAX reward across 4 attempts) ==")
    for sch in SCHEMES:
        per_inst_max = []
        per_inst_mean = []
        for iid, lst in rewards[sch].items():
            vals = [v for v in lst if v is not None]
            if not vals:
                continue
            per_inst_max.append(max(vals))
            per_inst_mean.append(sum(vals) / len(vals))
        print(f"\nscheme={sch}")
        summarize("max@4", per_inst_max)
        summarize("mean@4", per_inst_mean)
        labels, counts = histogram(per_inst_max)
        n = sum(counts) or 1
        print("  max@4 hist:")
        for lab, c in zip(labels, counts):
            print(f"    {lab:>14}: {c:5d} ({100*c/n:5.1f}%)")

    # Middle-band intersect: count instances with at least one strict-between attempt
    print("\n== Middle-band candidates ==")
    for sch in SCHEMES:
        per_inst = rewards[sch]
        # Count instances where attempt rewards span (have both <0.5 and >0.5)
        spans = 0
        any_nonzero = 0
        for iid, lst in per_inst.items():
            vals = [v for v in lst if v is not None]
            if not vals:
                continue
            if any(v > 1e-9 for v in vals):
                any_nonzero += 1
            if any(v < 0.5 for v in vals) and any(v > 0.5 for v in vals):
                spans += 1
        n = sum(1 for lst in per_inst.values() if any(v is not None for v in lst))
        print(f"  {sch:<10} n={n}  any-nonzero={any_nonzero} ({100*any_nonzero/n:.1f}%)  span_lo<0.5_hi>0.5={spans} ({100*spans/n:.1f}%)")

    # Build per-instance hard = (# attempts resolved) / 4 (None attempts treated as 0)
    hard_by_id = {}
    for iid, flags in resolved_flags.items():
        n_resolved = sum(1 for f in flags if f is True)
        hard_by_id[iid] = n_resolved / 4.0

    # ---- Expanded histogram with hard + baseline_pass_rate ----
    print("\n== Expanded per-instance histogram (n=950) ==")
    per_inst_cols = {
        "soft": [max((v for v in rewards["soft"][i] if v is not None), default=0.0) for i in rewards["soft"]],
        "joint": [max((v for v in rewards["joint"][i] if v is not None), default=0.0) for i in rewards["joint"]],
        "f2p_only": [max((v for v in rewards["f2p_only"][i] if v is not None), default=0.0) for i in rewards["f2p_only"]],
        "hard": list(hard_by_id.values()),
        "baseline_pass_rate": list(baseline_pass_rate.values()),
    }
    bins = (0.0, 0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999, 1.001)
    labels, _ = histogram([], bins=bins)
    col_names = list(per_inst_cols.keys())
    print(f"  {'bin':>14}  " + "  ".join(f"{c:>10s}" for c in col_names))
    cols_hist = {c: histogram(per_inst_cols[c], bins=bins)[1] for c in col_names}
    for i, lab in enumerate(labels):
        row = [cols_hist[c][i] for c in col_names]
        n_per_col = [sum(cols_hist[c]) or 1 for c in col_names]
        print(f"  {lab:>14}  " + "  ".join(f"{100*v/n:>9.1f}%" for v, n in zip(row, n_per_col)))

    # Print summary per column
    print("\n  summary:")
    for c in col_names:
        vals = per_inst_cols[c]
        n = len(vals)
        mean = sum(vals) / n if n else 0.0
        print(f"    {c:<20s}  n={n:4d}  mean={mean:.4f}  min={min(vals):.4f}  max={max(vals):.4f}")

    # Dump JSON for downstream use
    out_path = Path("/tmp/train_950_rewards.json")
    dump = {
        **{sch: {iid: rewards[sch][iid] for iid in rewards[sch]} for sch in SCHEMES},
        "hard": hard_by_id,
        "baseline_pass_rate": baseline_pass_rate,
    }
    out_path.write_text(json.dumps(dump))
    print(f"\nWrote per-instance reward table → {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
