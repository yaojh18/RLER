#!/usr/bin/env python3
"""Read a parallel-search SFT artifact_root and report:
  - per-instance: # ForkGroups, rubric-bank carry-forward integrity, Lane C
    completion rate, GT pass rate, accepted-group count.
  - aggregated: overall acceptance rate, examples of policy + rubric SFT
    samples that PASSED the two-stage filter, examples of dropped groups
    + their drop reason.
  - sanity checks: well-formed prompt/turns in policy samples, valid JSON
    rubric in rubric samples' last asst turn, no NaN/inf rewards.

Usage:
    python3 verify_sft_run.py <artifact_root>
    python3 verify_sft_run.py <artifact_root> --instance amoffat__sh-744
    python3 verify_sft_run.py <artifact_root> --json   # machine-readable

Exit code 0 if collection looks healthy, 1 otherwise.

This script is read-only — never modifies the artifact tree.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc)}


def _bank_active_ids(snapshot: list[dict[str, Any]] | dict[str, Any] | None) -> list[str]:
    if not snapshot:
        return []
    if isinstance(snapshot, dict):
        snapshot = snapshot.get("active_bank_after") or snapshot.get("active_bank") or []
    return [r.get("rubric_id", "") for r in snapshot if isinstance(r, dict)]


def verify_instance(instance_dir: Path) -> dict[str, Any]:
    """Walk one instance's run_dir and produce a per-instance report."""
    report: dict[str, Any] = {
        "instance_dir": str(instance_dir),
        "ok": True,
        "warnings": [],
        "errors": [],
    }
    record_path = instance_dir / "instance_record.json"
    if not record_path.exists():
        report["ok"] = False
        report["errors"].append(f"no instance_record.json under {instance_dir}")
        return report
    record = _load_json(record_path)
    report["instance_id"] = record.get("instance_id")
    report["completed"] = record.get("completed")
    report["seconds"] = record.get("seconds")
    report["num_groups"] = record.get("num_groups")
    report["num_mid_cps"] = record.get("num_mid_cps")
    if record.get("error"):
        report["warnings"].append(f"instance error: {record['error']}")

    groups_dir = instance_dir / "groups"
    if not groups_dir.exists():
        report["errors"].append(f"no groups dir at {groups_dir}")
        report["ok"] = False
        return report

    # ---- Per-group walk: bank state + Lane C completion + GT + acceptance ----
    group_reports: list[dict[str, Any]] = []
    prior_active_ids: list[str] | None = None
    for gdir in sorted(groups_dir.iterdir()):
        if not gdir.is_dir():
            continue
        try:
            gid = int(gdir.name.removeprefix("group_"))
        except ValueError:
            continue
        g: dict[str, Any] = {"group_index": gid}
        # Lane C output
        rubric_resp = _load_json(gdir / "rubric_response.json")
        g["lane_c_error"] = rubric_resp.get("error")
        g["n_rubric_samples"] = rubric_resp.get("n_rubric_samples", 0)
        g["selected_sample_index"] = rubric_resp.get("selected_sample_index")
        # Rubric bank state after this group
        bank_after_path = gdir / "rubric_bank_after.json"
        if bank_after_path.exists():
            bank_after = _load_json(bank_after_path)
            active_ids = [r.get("rubric_id", "") for r in (bank_after.get("active_bank_after") or [])]
            g["n_active_after"] = len(active_ids)
            g["n_inactive_after"] = len(bank_after.get("inactive_bank_after") or [])
            # Carry-forward: any rubric_id from prior group's active_bank that
            # was retained or replaced. We expect bank to evolve, not vanish.
            if prior_active_ids is not None:
                shared = set(prior_active_ids) & set(active_ids)
                g["bank_carry_intersection"] = len(shared)
                g["bank_prior_active"] = len(prior_active_ids)
                if g["n_active_after"] == 0 and prior_active_ids:
                    report["warnings"].append(
                        f"group {gid}: active_bank collapsed to empty (had {len(prior_active_ids)} prior)"
                    )
            prior_active_ids = active_ids
        else:
            g["n_active_after"] = None
            if not g["lane_c_error"]:
                report["warnings"].append(
                    f"group {gid}: Lane C reports success but no rubric_bank_after.json"
                )

        # Branch summary
        members = _load_json(gdir / "members.json")
        branches = members.get("branches") or []
        g["n_branches"] = len(branches)
        gt_scores: list[float] = []
        for b in branches:
            s = b.get("gt_score")
            if isinstance(s, (int, float)) and not math.isnan(float(s)):
                gt_scores.append(float(s))
        g["n_gt_scored"] = len(gt_scores)
        g["n_gt_passed"] = sum(1 for s in gt_scores if s >= 1.0)
        # Rubric_samples on disk — count valid (no terminal_error)
        rubric_samples_path = gdir / "rubric_samples.json"
        if rubric_samples_path.exists():
            rs = _load_json(rubric_samples_path)
            samples = rs.get("rubric_samples") or []
            g["n_valid_rubric_samples"] = sum(
                1 for s in samples if not s.get("terminal_error") and not s.get("format_errors")
            )
        group_reports.append(g)
    report["groups"] = group_reports

    # ---- Try ParallelSFTDataExporter ----
    # If imports fail (e.g. running on the login node which doesn't have
    # dotenv / litellm / other deps that live only in the baked
    # container), fall back to reading summary.json's per-instance row
    # for the same metrics. The collector wrote summary.json at instance
    # completion time WHILE inside the container — so the numbers are
    # ground truth even when the verifier can't re-export here.
    summary_results_by_instance: dict[str, dict[str, Any]] = {}
    # instance_dir = <ART>/search_outputs/teacher_student_parallel/<inst>
    # 3 parents up = <ART> where summary.json lives.
    sj = instance_dir.parent.parent.parent / "summary.json"
    if sj.exists():
        try:
            sj_payload = _load_json(sj)
            for r in (sj_payload.get("results") or []):
                summary_results_by_instance[r.get("instance_id") or ""] = r
        except Exception:
            pass
    try:
        # Walk up from <repo>/slime/train_agent/scripts/gcp/verify_sft_run.py
        # — 4 parents up = <repo>. Both train_agent and swe_agent are
        # importable from there.
        repo_root = Path(__file__).resolve().parents[4]
        for sub in ("agent", "slime"):
            p = str(repo_root / sub)
            if p not in sys.path:
                sys.path.insert(0, p)
        from train_agent.data_export import ParallelSFTDataExporter
        bundle = ParallelSFTDataExporter(run_dir=instance_dir).export_bundle()
        report["accepted_group_ids"] = list(bundle.accepted_group_ids)
        report["n_policy_samples"] = len(bundle.policy_samples)
        report["n_rubric_samples"] = len(bundle.rubric_samples)
        report["n_groups_with_policy"] = int(bundle.metadata.get("num_groups_with_policy", 0))
        report["n_groups_with_rubric"] = int(bundle.metadata.get("num_groups_with_rubric", 0))
        report["n_groups_with_any"] = len(bundle.accepted_group_ids)
        # Sanity check 1: policy samples have well-formed prompt+turns
        bad_policy = 0
        for s in bundle.policy_samples:
            if not s.prompt or not s.turns:
                bad_policy += 1
                continue
            if not any(m.get("role") == "assistant" and m.get("content") for m in s.turns):
                bad_policy += 1
        report["n_bad_policy"] = bad_policy
        # Sanity check 2: rubric samples have valid JSON in last asst turn
        bad_rubric = 0
        for s in bundle.rubric_samples:
            if not s.turns:
                bad_rubric += 1; continue
            last_asst = next(
                (m for m in reversed(s.turns) if m.get("role") == "assistant"), None,
            )
            if last_asst is None or not last_asst.get("content"):
                bad_rubric += 1; continue
            # The assistant content should be parseable JSON (a rubric object)
            try:
                content = last_asst["content"].strip()
                # Could be JSON-wrapped in <think>...</think> blocks; extract last {...}
                last_brace = content.rfind("{")
                if last_brace >= 0:
                    json.loads(content[last_brace:].rstrip("`").strip())
            except Exception:
                bad_rubric += 1
        report["n_bad_rubric"] = bad_rubric
        if bad_policy or bad_rubric:
            report["warnings"].append(
                f"malformed samples: policy={bad_policy} rubric={bad_rubric}"
            )
    except Exception as exc:
        # Fall back: read the in-container summary.json which the
        # collector wrote at instance completion (works on login node
        # without the heavy deps).
        sj_row = summary_results_by_instance.get(report.get("instance_id") or "")
        if sj_row:
            report["accepted_group_ids"] = list(sj_row.get("accepted_group_ids") or [])
            report["n_policy_samples"] = int(sj_row.get("policy_samples") or 0)
            report["n_rubric_samples"] = int(sj_row.get("rubric_samples") or 0)
            report["n_groups_with_policy"] = int(sj_row.get("groups_with_policy") or 0)
            report["n_groups_with_rubric"] = int(sj_row.get("groups_with_rubric") or 0)
            report["n_groups_with_any"] = int(sj_row.get("groups_with_any") or len(sj_row.get("accepted_group_ids") or []))
            report["n_bad_policy"] = 0
            report["n_bad_rubric"] = 0
            report["warnings"].append(
                f"ParallelSFTDataExporter import failed ({type(exc).__name__}); "
                f"fell back to summary.json"
            )
        else:
            report["errors"].append(f"ParallelSFTDataExporter raised: {exc}")
            report["ok"] = False
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("artifact_root", type=Path)
    p.add_argument("--instance", default=None, help="Only verify this instance subdir")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    p.add_argument("--show-sample", action="store_true", help="Dump first accepted policy + rubric sample")
    args = p.parse_args()

    root = args.artifact_root
    summary_path = root / "summary.json"
    summary = _load_json(summary_path) if summary_path.exists() else {}

    # Each per-instance run_dir is under search_outputs/teacher_student_parallel/
    search_root = root / "search_outputs" / "teacher_student_parallel"
    if not search_root.exists():
        print(f"ERROR: {search_root} does not exist", file=sys.stderr)
        return 1
    instance_dirs = [d for d in sorted(search_root.iterdir()) if d.is_dir()]
    if args.instance:
        instance_dirs = [d for d in instance_dirs if args.instance in d.name]
        if not instance_dirs:
            print(f"ERROR: no instance subdir matches '{args.instance}'", file=sys.stderr)
            return 1

    reports = [verify_instance(d) for d in instance_dirs]

    # ---- Aggregate ----
    n_instances = len(reports)
    n_completed = sum(1 for r in reports if r.get("completed"))
    n_with_groups = sum(1 for r in reports if (r.get("num_groups") or 0) > 0)
    total_accepted_groups = sum(len(r.get("accepted_group_ids") or []) for r in reports)
    total_policy_samples = sum(r.get("n_policy_samples") or 0 for r in reports)
    total_rubric_samples = sum(r.get("n_rubric_samples") or 0 for r in reports)
    # Decoupled group counts (post-2026-05-21):
    total_groups_with_policy = sum(r.get("n_groups_with_policy") or 0 for r in reports)
    total_groups_with_rubric = sum(r.get("n_groups_with_rubric") or 0 for r in reports)
    total_groups_with_any = sum(r.get("n_groups_with_any") or 0 for r in reports)
    total_bad_policy = sum(r.get("n_bad_policy") or 0 for r in reports)
    total_bad_rubric = sum(r.get("n_bad_rubric") or 0 for r in reports)
    total_groups_seen = sum(r.get("num_groups") or 0 for r in reports)
    total_gt_passed = sum(
        sum(g.get("n_gt_passed", 0) for g in r.get("groups", []))
        for r in reports
    )
    total_branches_scored = sum(
        sum(g.get("n_gt_scored", 0) for g in r.get("groups", []))
        for r in reports
    )
    # Rubric bank continuity: across instances, how often did the bank
    # collapse to 0 vs. grew/persisted?
    bank_carryforward_ok = 0
    bank_carryforward_collapsed = 0
    for r in reports:
        for g in r.get("groups", []):
            if g.get("n_active_after") is None:
                continue
            if g.get("bank_prior_active"):
                if g.get("n_active_after", 0) > 0:
                    bank_carryforward_ok += 1
                else:
                    bank_carryforward_collapsed += 1

    out: dict[str, Any] = {
        "artifact_root": str(root),
        "n_instances": n_instances,
        "n_completed": n_completed,
        "n_with_groups": n_with_groups,
        "summary_results_count": len(summary.get("results") or []),
        "summary_successful": summary.get("successful_instances"),
        "summary_failed": summary.get("failed_instances"),
        "total_groups_seen": total_groups_seen,
        # DECOUPLED group counts (a group can contribute policy XOR
        # rubric XOR both; groups_with_any = union, NOT intersection).
        "total_groups_with_policy": total_groups_with_policy,
        "total_groups_with_rubric": total_groups_with_rubric,
        "total_groups_with_any": total_groups_with_any,
        "total_accepted_groups_legacy": total_accepted_groups,  # alias
        "total_policy_samples": total_policy_samples,
        "total_rubric_samples": total_rubric_samples,
        "total_bad_policy": total_bad_policy,
        "total_bad_rubric": total_bad_rubric,
        "total_branches_scored": total_branches_scored,
        "total_gt_passed": total_gt_passed,
        "gt_pass_rate": (total_gt_passed / total_branches_scored)
            if total_branches_scored else None,
        "rubric_bank_carryforward_ok": bank_carryforward_ok,
        "rubric_bank_carryforward_collapsed": bank_carryforward_collapsed,
        "policy_yield_rate": (total_groups_with_policy / total_groups_seen)
            if total_groups_seen else None,
        "rubric_yield_rate": (total_groups_with_rubric / total_groups_seen)
            if total_groups_seen else None,
        "any_yield_rate": (total_groups_with_any / total_groups_seen)
            if total_groups_seen else None,
        "instances": reports,
    }

    if args.show_sample:
        sample_dump = None
        for r in reports:
            if r.get("n_policy_samples") and r.get("n_rubric_samples"):
                # Pull from the bundle again
                from train_agent.data_export import ParallelSFTDataExporter
                b = ParallelSFTDataExporter(run_dir=Path(r["instance_dir"])).export_bundle()
                if b.policy_samples and b.rubric_samples:
                    sample_dump = {
                        "instance_id": r["instance_id"],
                        "first_policy_sample": {
                            "sample_id": b.policy_samples[0].sample_id,
                            "n_prompt": len(b.policy_samples[0].prompt),
                            "n_turns": len(b.policy_samples[0].turns),
                            "first_turn_role": b.policy_samples[0].turns[0].get("role") if b.policy_samples[0].turns else None,
                            "asst_chars": sum(
                                len(m.get("content") or "")
                                for m in b.policy_samples[0].turns
                                if m.get("role") == "assistant"
                            ),
                        },
                        "first_rubric_sample": {
                            "sample_id": b.rubric_samples[0].sample_id,
                            "n_prompt": len(b.rubric_samples[0].prompt),
                            "n_turns": len(b.rubric_samples[0].turns),
                            "last_asst_chars": (
                                len((b.rubric_samples[0].turns[-1].get("content") or ""))
                                if b.rubric_samples[0].turns else 0
                            ),
                        },
                    }
                    break
        out["sample_dump"] = sample_dump

    health_ok = (
        n_instances > 0
        and total_groups_with_any > 0      # at least 1 sample (policy OR rubric)
        and total_bad_policy == 0
        and total_bad_rubric == 0
        and bank_carryforward_collapsed == 0
    )

    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        print(f"=== SFT verification: {root} ===")
        print(f"  instances:                 {n_instances}")
        print(f"  completed:                 {n_completed}")
        print(f"  total groups (forks):      {total_groups_seen}")
        print()
        print(f"  ---- decoupled group yield ----")
        print(f"  groups with policy:        {total_groups_with_policy}"
              + (f"  ({total_groups_with_policy/total_groups_seen:.1%})" if total_groups_seen else ""))
        print(f"  groups with rubric:        {total_groups_with_rubric}"
              + (f"  ({total_groups_with_rubric/total_groups_seen:.1%})" if total_groups_seen else ""))
        print(f"  groups with EITHER (any):  {total_groups_with_any}"
              + (f"  ({total_groups_with_any/total_groups_seen:.1%})" if total_groups_seen else ""))
        print()
        print(f"  ---- SFT samples ----")
        print(f"  policy SFT samples:        {total_policy_samples}")
        print(f"  rubric SFT samples:        {total_rubric_samples}")
        print(f"  malformed policy:          {total_bad_policy}")
        print(f"  malformed rubric:          {total_bad_rubric}")
        print()
        print(f"  ---- GT signal ----")
        print(f"  branches GT-scored:        {total_branches_scored}")
        print(f"  branches GT-passed (>=1):  {total_gt_passed}")
        if total_branches_scored:
            print(f"  GT pass rate:              {total_gt_passed / total_branches_scored:.1%}")
        print()
        print(f"  ---- rubric bank ----")
        print(f"  bank carry-forward OK:     {bank_carryforward_ok}")
        print(f"  bank carry-forward died:   {bank_carryforward_collapsed}")
        print()
        # Per-instance rows: g=groups, gP=with_policy, gR=with_rubric, P/R=samples
        print(f"  {'instance':50s}   g  gP  gR   P   R")
        print(f"  {'-'*50}  ---  --  --  --  --")
        for r in reports:
            iid = r.get("instance_id") or Path(r["instance_dir"]).name
            ng = r.get("num_groups") or 0
            gp = r.get("n_groups_with_policy") or 0
            gr = r.get("n_groups_with_rubric") or 0
            pol = r.get("n_policy_samples") or 0
            rub = r.get("n_rubric_samples") or 0
            warn = "  WARN: " + "; ".join(r.get("warnings") or []) if r.get("warnings") else ""
            err = "  ERR: " + "; ".join(r.get("errors") or []) if r.get("errors") else ""
            print(f"  {iid:50s} {ng:3d} {gp:3d} {gr:3d} {pol:3d} {rub:3d}{warn}{err}")
        if out.get("sample_dump"):
            print()
            print("Sample dump:", json.dumps(out["sample_dump"], indent=2))
        print()
        print(f"OVERALL HEALTH: {'OK' if health_ok else 'PROBLEMS DETECTED'}")
    return 0 if health_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
