"""Per-instance full breakdown for the parallel SFT runs.

Reads instance_record.json + per-group members.json + rubric_samples.json
for each instance and reports:
  - n_mid_cps (= n fork groups)
  - n_lane_a_asst_steps (Lane A terminal step count)
  - per-branch avg steps in Lane B continuations
  - groups_accepted_old_filter (1 per group)
  - groups_accepted_new_filter (1 per group if any GT-pass + any valid rubric)
  - dropped_no_gt_pass (group had 0 branches GT-passing)
  - dropped_no_valid_rubric (group had no rubric draft passing filter)
  - rubric_drafts_generated (N per group * n_groups)
  - rubric_drafts_valid (after format/terminal/correlation filter)
  - policy_samples_old (1 per accepted group)
  - policy_samples_new (n GT-passing branches per group, summed)
"""
import json, glob, os, sys
from pathlib import Path

sys.path.insert(0, '/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/RLER-zsh-trim/agent')
from swe_agent.parallel_utils import gap_corr

ROOTS = {
    "LL_30easy":  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/sft_parallel_30easy_20260521_103031",
    "HT_30easy":  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/sft_parallel_30easy_HT_20260521_113559",
}

GT_THRESH = 1.0
RUBRIC_CORR = 0.8

def rubric_valid(sample, gts, child_rewards_by_node_id, member_nodes):
    """Same logic as ParallelSFTDataExporter._rubric_judge_corr_gt + format filter."""
    if sample.get("format_errors") or sample.get("terminal_error"):
        return False
    cr = sample.get("child_rewards") or {}
    if not cr:
        return False
    pers, gts_p = [], []
    for node_id, gt in zip(member_nodes, gts):
        if node_id in cr and gt is not None:
            try:
                pers.append(float(cr[node_id]))
                gts_p.append(float(gt))
            except (TypeError, ValueError):
                pass
    if len(pers) < 2:
        return False
    if len(set(gts_p)) <= 1:
        return len(set(pers)) > 1
    return gap_corr(pers, gts_p) > RUBRIC_CORR


def analyze_instance(inst_dir):
    rec = json.load(open(f"{inst_dir}/instance_record.json"))
    iid = rec.get("instance_id")
    n_mid_cps = rec.get("num_mid_cps", 0)
    seconds = rec.get("seconds", 0)
    n_groups = rec.get("num_groups", 0)

    lane_a_term_step = 0
    branch_step_counts = []
    # Decoupled accounting (per user 2026-05-21):
    groups_with_policy = 0  # group had >=1 GT-pass branch
    groups_with_rubric = 0  # group had >=1 valid rubric draft (allowed even if 0 GT-pass)
    groups_with_neither = 0  # zero policy AND zero rubric → fully dropped
    rubric_drafts_generated = 0
    rubric_drafts_valid = 0
    policy_samples = 0  # all GT-passing branches across the instance
    per_group: list[dict] = []

    for gdir in sorted(glob.glob(f"{inst_dir}/groups/group_*")):
        members_path = Path(gdir) / "members.json"
        if not members_path.exists():
            continue
        members = json.load(open(members_path))
        mid_cp_step = members.get("mid_cp", {}).get("asst_step", 0)
        lane_a_term_step = max(lane_a_term_step, mid_cp_step)
        branches = members.get("branches") or []
        gts = [b.get("gt_score") for b in branches]
        member_node_ids = [b.get("node_id") for b in branches]
        for b in branches:
            if b.get("n_step_cards"):
                branch_step_counts.append(b["n_step_cards"])
        n_gt_pass = sum(1 for g in gts if g is not None and float(g) >= GT_THRESH)
        rsamples_path = Path(gdir) / "rubric_samples.json"
        n_drafts = 0
        n_valid_drafts = 0
        if rsamples_path.exists():
            rs = json.load(open(rsamples_path))
            samples = rs.get("rubric_samples") or []
            n_drafts = len(samples)
            for s in samples:
                if rubric_valid(s, gts, None, member_node_ids):
                    n_valid_drafts += 1
        rubric_drafts_generated += n_drafts
        rubric_drafts_valid += n_valid_drafts
        policy_samples += n_gt_pass

        has_policy = n_gt_pass > 0
        has_rubric = n_valid_drafts > 0
        if has_policy:
            groups_with_policy += 1
        if has_rubric:
            groups_with_rubric += 1
        if not has_policy and not has_rubric:
            groups_with_neither += 1
        per_group.append({
            "gid": int(Path(gdir).name.split("_")[-1]),
            "mid_cp_step": mid_cp_step,
            "n_gt_pass": n_gt_pass,
            "n_rubric_drafts": n_drafts,
            "n_valid_drafts": n_valid_drafts,
        })

    avg_branch_steps = sum(branch_step_counts) / len(branch_step_counts) if branch_step_counts else 0.0
    return {
        "instance_id": iid,
        "n_mid_cps": n_mid_cps,
        "n_groups": n_groups,
        "lane_a_term_step": lane_a_term_step,
        "avg_branch_steps": round(avg_branch_steps, 1),
        "g_pol": groups_with_policy,
        "g_rub": groups_with_rubric,
        "g_neither": groups_with_neither,
        "rubric_drafts_gen": rubric_drafts_generated,
        "rubric_drafts_valid": rubric_drafts_valid,
        "policy_samples": policy_samples,
        "seconds": int(seconds),
    }


for label, art in ROOTS.items():
    print(f"\n{'='*120}")
    print(f"=== {label}  ({art.split('/')[-1]}) ===")
    print(f"{'='*120}")
    rows = []
    for inst_dir in sorted(glob.glob(f"{art}/search_outputs/teacher_student_parallel/*")):
        if (Path(inst_dir)/"instance_record.json").exists():
            rows.append(analyze_instance(inst_dir))
    if not rows:
        print("  (no completed instances)")
        continue

    # header
    print(f"  {'instance':45s} {'mid':>3s} {'aSt':>4s} {'bSt':>4s} {'gPol':>4s} {'gRub':>4s} {'g0':>3s} {'rGen':>4s} {'rVal':>4s} {'POL':>4s} {'sec':>5s}")
    print(f"  {'-'*45} {'---':>3s} {'----':>4s} {'----':>4s} {'----':>4s} {'----':>4s} {'---':>3s} {'----':>4s} {'----':>4s} {'----':>4s} {'-----':>5s}")
    for r in sorted(rows, key=lambda r: r["instance_id"]):
        print(f"  {r['instance_id'][:45]:45s} {r['n_mid_cps']:>3d} {r['lane_a_term_step']:>4d} {r['avg_branch_steps']:>4.0f} {r['g_pol']:>4d} {r['g_rub']:>4d} {r['g_neither']:>3d} {r['rubric_drafts_gen']:>4d} {r['rubric_drafts_valid']:>4d} {r['policy_samples']:>4d} {r['seconds']:>5d}")
    n = len(rows)
    def s(k): return sum(r[k] for r in rows)
    def a(k): return s(k) / n if n else 0
    print(f"  {'-'*45}")
    print(f"  {'TOTAL ({} insts)'.format(n)[:45]:45s} "
          f"{s('n_mid_cps'):>3d} {a('lane_a_term_step'):>4.0f} {a('avg_branch_steps'):>4.0f} "
          f"{s('g_pol'):>4d} {s('g_rub'):>4d} {s('g_neither'):>3d} "
          f"{s('rubric_drafts_gen'):>4d} {s('rubric_drafts_valid'):>4d} "
          f"{s('policy_samples'):>4d} {a('seconds'):>5.0f}")

print("\n\nLegend (DECOUPLED exporter — 2026-05-21):")
print("  mid   = n mid_cps emitted by Lane A (= n fork groups)")
print("  aSt   = Lane A terminal asst step count")
print("  bSt   = avg Lane B branch step count")
print("  gPol  = groups contributing >=1 policy sample (had >=1 GT-pass branch)")
print("  gRub  = groups contributing >=1 rubric sample (had >=1 valid rubric draft)")
print("          NOT gated on GT — even all-fail groups contribute rubric data")
print("  g0    = groups contributing NEITHER (fully dropped)")
print("  rGen  = total rubric drafts generated (= N * n_groups)")
print("  rVal  = rubric SFT samples emitted = drafts passing filter")
print("  POL   = policy SFT samples emitted = total GT-passing branches")
print("  sec   = wallclock seconds")
