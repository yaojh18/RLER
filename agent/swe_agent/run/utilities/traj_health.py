"""Scan mini-swe-agent run dirs for trajectory-health regressions.

Reads `<run>/single_run/<inst>/<inst>.traj.json` and reports per-instance and
per-run aggregates of signals that indicated the iter9 -> iter59 verbose-thrash
regression on the 55433 GRPO checkpoints.

With --grpo, instead walks the live GRPO training-rollout layout
  `<run>/{lane_out,naive_out}/rollout_NNNN/<inst>/<ts>/task-NNNNNN/rollouts/rollout_MM/`
and reads `messages.json`+`summary.json` to synthesise the same per-rollout
record so the loop/empty-bash/docker-death signals can be checked mid-training.

  * trajectory length (msg count, total assistant chars/est-tokens)
  * per-turn assistant-message length distribution (median/p90/p99/max)
  * repeated-content turn count (model loops re-emitting near-identical plans)
  * empty-bash-output turn count (heredoc-only turns yielding no feedback)
  * docker container death ("No such container" surfaced through bash)
  * litellm timeout occurrences (inside the trajectory user observations)
  * exit_status histogram

Usage:
  python traj_health.py <run_dir> [<run_dir2> ...]
      --compare         emit a side-by-side table across runs
      --json            machine-readable JSON to stdout
      --top N           print N longest trajectories per run (default 5)
      --label LABEL,..  rename runs in the comparison table

A run_dir is any directory that contains `single_run/<inst>/<inst>.traj.json`.
Glob patterns are accepted as well.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

TOKEN_PER_CHAR = 1.0 / 3.5  # rough Qwen tokenizer ratio
LOOP_PREFIX_CHARS = 400     # prefix window used for repeated-content hashing
LOOP_MIN_LEN = 1500         # ignore short assistant turns when scanning for loops
EMPTY_OUTPUT_MARKERS = ("<output>\n</output>", "<output></output>")
DOCKER_DEATH_MARKER = "No such container"
LITELLM_TIMEOUT_MARKER = "litellm.Timeout"


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((len(xs) - 1) * q))))
    return float(xs[k])


def find_trajectories(run_dir: Path) -> list[Path]:
    base = run_dir / "single_run"
    if not base.is_dir():
        return []
    return sorted(base.glob("*/*.traj.json"))


def find_grpo_rollouts(run_dir: Path) -> list[tuple[str, Path]]:
    """Locate live GRPO rollout dirs.

    Returns a list of (kind, path) where kind is:
      "naive"  -> dir holds messages.json + summary.json directly
      "lane"   -> dir is a `branch_NN` whose full traj = shared_parent.json + continuation.json
    """
    out: list[tuple[str, Path]] = []
    naive_base = run_dir / "naive_out"
    if naive_base.is_dir():
        for r in sorted(naive_base.glob("rollout_*/*/*/task-*/rollouts/rollout_*")):
            if (r / "messages.json").is_file():
                out.append(("naive", r))
    lane_base = run_dir / "lane_out"
    if lane_base.is_dir():
        for b in sorted(lane_base.glob("rollout_*/*/*/task-*/groups/group_*/branches/branch_*")):
            if (b / "summary.json").is_file():
                out.append(("lane", b))
    return out


def analyse_grpo_rollout(kind: str, path: Path) -> dict[str, Any]:
    """Analyse a single GRPO training rollout (naive rollout or lane branch)."""
    summary: dict[str, Any] = {}
    try:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8", errors="replace"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    if kind == "naive":
        try:
            msgs = json.loads((path / "messages.json").read_text(encoding="utf-8", errors="replace"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            return {"path": str(path), "error": f"messages.json: {exc}"}
        instance_short = path.parents[2].name
        rollout_step = path.parents[4].name
        rollout_idx = path.name
    else:  # lane branch
        # full conversation = shared_parent.json + continuation.json
        group_dir = path.parent.parent  # branches/.. -> group_XXX
        msgs: list[Any] = []
        try:
            parent = json.loads((group_dir / "shared_parent.json").read_text(encoding="utf-8", errors="replace"))
            if isinstance(parent, list):
                msgs.extend(parent)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        try:
            cont = json.loads((path / "continuation.json").read_text(encoding="utf-8", errors="replace"))
            if isinstance(cont, list):
                msgs.extend(cont)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            return {"path": str(path), "error": f"continuation.json: {exc}"}
        instance_short = path.parents[4].name
        rollout_step = path.parents[6].name
        rollout_idx = f"{path.parents[1].name}/{path.name}"  # group_XXX/branch_NN

    patch_chars = 0
    try:
        patch_chars = (path / "terminal_patch.txt").stat().st_size
    except OSError:
        pass

    status = summary.get("status") or "unknown"
    if summary.get("error"):
        exit_status = f"error:{str(summary['error'])[:24]}"
    elif summary.get("terminated_early"):
        exit_status = "terminated_early"
    else:
        exit_status = str(status)

    info = {"exit_status": exit_status, "submission": "x" if patch_chars > 0 else ""}
    traj = {"messages": msgs, "info": info,
            "instance_id": f"{instance_short}/{rollout_idx}"}
    rec = _analyse_messages(traj, path / "messages.json")
    rec["rollout_step"] = rollout_step
    rec["instance_short"] = instance_short
    rec["rollout_idx"] = rollout_idx
    rec["terminated_early"] = bool(summary.get("terminated_early"))
    rec["patch_chars"] = patch_chars
    rec["kind"] = kind
    return rec


def _analyse_messages(traj: dict[str, Any], path: Path) -> dict[str, Any]:
    """Shared per-trajectory analysis used by both legacy and GRPO paths."""
    msgs = traj.get("messages", []) or []
    info = traj.get("info", {}) or {}
    exit_status = str(info.get("exit_status") or "Unknown")
    submission_present = bool((info.get("submission") or "").strip())

    asst_chars: list[int] = []
    loop_sigs: list[str] = []
    empty_bash = 0
    docker_death = 0
    litellm_timeouts = 0
    first_docker_death_msg = -1

    for i, m in enumerate(msgs):
        role = m.get("role", "")
        content = m.get("content", "") or ""
        if role == "assistant":
            asst_chars.append(len(content))
            if len(content) >= LOOP_MIN_LEN:
                loop_sigs.append(loop_signature(content))
        elif role == "user":
            stripped = content.replace(" ", "").replace("\n", "")
            if "<returncode>0</returncode>" in content and any(
                m.replace(" ", "").replace("\n", "") in stripped for m in EMPTY_OUTPUT_MARKERS
            ):
                empty_bash += 1
            if DOCKER_DEATH_MARKER in content:
                docker_death += 1
                if first_docker_death_msg < 0:
                    first_docker_death_msg = i
            if LITELLM_TIMEOUT_MARKER in content:
                litellm_timeouts += 1

    sig_counts = Counter(loop_sigs)
    loop_turns = sum(c for c in sig_counts.values() if c >= 2)
    repeated_distinct_sigs = sum(1 for c in sig_counts.values() if c >= 2)
    total_asst_chars = sum(asst_chars)
    return {
        "path": str(path),
        "instance_id": traj.get("instance_id") or path.parent.name,
        "n_msgs": len(msgs),
        "n_assistant_turns": len(asst_chars),
        "asst_chars_total": total_asst_chars,
        "asst_tokens_est": int(total_asst_chars * TOKEN_PER_CHAR),
        "asst_per_turn_median": int(statistics.median(asst_chars)) if asst_chars else 0,
        "asst_per_turn_p90": int(pct(list(map(float, asst_chars)), 0.90)),
        "asst_per_turn_p99": int(pct(list(map(float, asst_chars)), 0.99)),
        "asst_per_turn_max": max(asst_chars) if asst_chars else 0,
        "loop_turns": loop_turns,
        "loop_signatures_repeated": repeated_distinct_sigs,
        "empty_bash_outputs": empty_bash,
        "docker_death_msgs": docker_death,
        "first_docker_death_msg": first_docker_death_msg,
        "litellm_timeout_msgs": litellm_timeouts,
        "exit_status": exit_status,
        "submission_present": submission_present,
    }


def loop_signature(text: str) -> str:
    head = text.strip()[:LOOP_PREFIX_CHARS]
    return hashlib.md5(head.encode("utf-8", "ignore")).hexdigest()


def analyse_trajectory(path: Path) -> dict[str, Any]:
    try:
        traj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return {"path": str(path), "error": f"JSONDecodeError: {exc}"}
    return _analyse_messages(traj, path)


def aggregate_run(run_dir: Path, grpo: bool = False) -> dict[str, Any]:
    if grpo:
        rollout_dirs = find_grpo_rollouts(run_dir)
        if not rollout_dirs:
            return {"run_dir": str(run_dir), "n_trajectories": 0}
        per = [analyse_grpo_rollout(k, p) for k, p in rollout_dirs]
    else:
        traj_paths = find_trajectories(run_dir)
        if not traj_paths:
            return {"run_dir": str(run_dir), "n_trajectories": 0}
        per = [analyse_trajectory(p) for p in traj_paths]
    per = [r for r in per if "error" not in r]
    n = len(per)
    turns = [r["n_msgs"] for r in per]
    tokens = [r["asst_tokens_est"] for r in per]
    max_turn_chars = [r["asst_per_turn_max"] for r in per]
    exit_hist = Counter(r["exit_status"] for r in per)
    docker_death_rate = sum(1 for r in per if r["docker_death_msgs"] > 0) / n
    looped_rate = sum(1 for r in per if r["loop_turns"] >= 4) / n
    empty_heavy_rate = sum(1 for r in per if r["empty_bash_outputs"] >= 3) / n
    timeout_rate = sum(1 for r in per if r["litellm_timeout_msgs"] > 0) / n
    submitted_rate = sum(1 for r in per if r["submission_present"]) / n

    return {
        "run_dir": str(run_dir),
        "n_trajectories": n,
        "turns_p50": int(pct(list(map(float, turns)), 0.5)),
        "turns_p90": int(pct(list(map(float, turns)), 0.9)),
        "turns_p99": int(pct(list(map(float, turns)), 0.99)),
        "turns_max": max(turns) if turns else 0,
        "asst_tokens_p50": int(pct(list(map(float, tokens)), 0.5)),
        "asst_tokens_p90": int(pct(list(map(float, tokens)), 0.9)),
        "asst_tokens_p99": int(pct(list(map(float, tokens)), 0.99)),
        "asst_per_turn_p99_p99": int(pct(list(map(float, max_turn_chars)), 0.99)),
        "docker_death_rate": round(docker_death_rate, 3),
        "looped_traj_rate": round(looped_rate, 3),
        "empty_heavy_rate": round(empty_heavy_rate, 3),
        "litellm_timeout_rate": round(timeout_rate, 3),
        "submitted_rate": round(submitted_rate, 3),
        "exit_status_hist": dict(exit_hist.most_common()),
        "_per_trajectory": per,
    }


def print_run(agg: dict[str, Any], top: int) -> None:
    n = agg["n_trajectories"]
    if n == 0:
        print(f"\n== {agg['run_dir']} == (no trajectories)")
        return
    print(f"\n== {agg['run_dir']}  ({n} trajectories) ==")
    print(f"  turns                p50={agg['turns_p50']:<4} p90={agg['turns_p90']:<4} p99={agg['turns_p99']:<4} max={agg['turns_max']}")
    print(f"  asst tokens (est)    p50={agg['asst_tokens_p50']:<6} p90={agg['asst_tokens_p90']:<6} p99={agg['asst_tokens_p99']}")
    print(f"  per-turn chars p99-of-max: {agg['asst_per_turn_p99_p99']}  (single longest turn anywhere)")
    print(f"  docker_death_rate    {agg['docker_death_rate']:.1%}")
    print(f"  looped_traj_rate     {agg['looped_traj_rate']:.1%}   (>=4 repeated-prefix asst turns)")
    print(f"  empty_heavy_rate     {agg['empty_heavy_rate']:.1%}   (>=3 empty bash outputs)")
    print(f"  litellm_timeout_rate {agg['litellm_timeout_rate']:.1%}")
    print(f"  submitted_rate       {agg['submitted_rate']:.1%}")
    print(f"  exit_status:         {dict(list(agg['exit_status_hist'].items())[:6])}")
    if top > 0:
        print(f"  top-{top} longest trajectories:")
        ranked = sorted(agg["_per_trajectory"], key=lambda r: -r["asst_tokens_est"])[:top]
        for r in ranked:
            print(
                f"    {r['asst_tokens_est']:>6}t  {r['n_msgs']:>3}msg  "
                f"loop={r['loop_turns']:>2} empty={r['empty_bash_outputs']:>2} "
                f"docker_death@{r['first_docker_death_msg']:>3}  "
                f"{r['exit_status']:<18} {r['instance_id']}"
            )


COMPARE_COLS = [
    ("n", "n_trajectories", "{:>5}"),
    ("turns_p50", "turns_p50", "{:>4}"),
    ("turns_p99", "turns_p99", "{:>5}"),
    ("tokens_p50", "asst_tokens_p50", "{:>7}"),
    ("tokens_p99", "asst_tokens_p99", "{:>7}"),
    ("docker_die%", "docker_death_rate", "{:>7.1%}"),
    ("looped%", "looped_traj_rate", "{:>7.1%}"),
    ("empty%", "empty_heavy_rate", "{:>6.1%}"),
    ("timeout%", "litellm_timeout_rate", "{:>7.1%}"),
    ("submit%", "submitted_rate", "{:>7.1%}"),
]


def print_compare(aggs: list[dict[str, Any]], labels: list[str]) -> None:
    header = "label".ljust(30) + " ".join(f"{c[0]:>{len(c[2].format(0))}}" for c in COMPARE_COLS)
    print(header)
    print("-" * len(header))
    for label, agg in zip(labels, aggs):
        if agg["n_trajectories"] == 0:
            print(f"{label[:30]:<30}  (no trajectories)")
            continue
        row = label[:30].ljust(30) + " ".join(c[2].format(agg[c[1]]) for c in COMPARE_COLS)
        print(row)


def expand_run_dirs(patterns: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in patterns:
        for match in sorted(glob.glob(p)):
            key = os.path.realpath(match)
            if key in seen:
                continue
            seen.add(key)
            out.append(Path(match))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("run_dirs", nargs="+", help="run dir(s) or globs containing single_run/")
    ap.add_argument("--compare", action="store_true", help="emit side-by-side table across runs")
    ap.add_argument("--json", action="store_true", help="machine-readable JSON to stdout")
    ap.add_argument("--top", type=int, default=5, help="N longest trajectories per run (0=skip)")
    ap.add_argument("--label", default="", help="comma-separated labels for compare mode")
    ap.add_argument("--grpo", action="store_true",
                    help="walk live GRPO rollout layout instead of single_run/*.traj.json")
    args = ap.parse_args()

    dirs = expand_run_dirs(args.run_dirs)
    if not dirs:
        print("no matching run dirs", file=sys.stderr)
        return 2
    aggs = [aggregate_run(d, grpo=args.grpo) for d in dirs]

    if args.json:
        out = [{k: v for k, v in a.items() if k != "_per_trajectory"} for a in aggs]
        print(json.dumps(out, indent=2))
        return 0

    if args.compare:
        labels = [s.strip() for s in args.label.split(",")] if args.label else [d.name for d in dirs]
        if len(labels) < len(aggs):
            labels += [d.name for d in dirs[len(labels):]]
        print_compare(aggs, labels)
        return 0

    for agg in aggs:
        print_run(agg, top=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
