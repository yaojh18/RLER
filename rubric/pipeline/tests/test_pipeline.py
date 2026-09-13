from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import numpy as np
import pytest

import build_weight_problems
import generate_refine
import optimize_weights
from common import rubric_key


def _rubric(title: str = "Visible implementation") -> dict:
    return {
        "title": title,
        "polarity": "positive",
        "weight": 1.0,
        "applies_when": ["Task-relevant source evidence is visible."],
        "description": "Score only the visible task-causal implementation.",
        "metadata": {
            "stage": "editing",
            "judge_focus": "Applied source changes.",
            "evidence_authority": "Current source supersedes prose.",
            "hard_gate": "A concrete task-relevant change is visible.",
            "contradiction_rule": "A visible revert removes credit.",
            "oracle_test": "The requested behavior is implemented.",
            "failure_mode": "Plans are mistaken for applied work.",
        },
        "scale": {str(value): f"anchor {value}" for value in range(1, 6)},
    }


def _candidate(stage: str, *, action: str = "add", parent: str | None = None) -> dict:
    rubric = _rubric()
    return {
        "candidate_id": f"{stage}-candidate",
        "candidate_rubric_key": rubric_key(rubric),
        "stage": stage,
        "action": action,
        "parent_rubric_key": parent,
        "rubric": rubric,
    }


def _ledger(candidate: dict, raw_scores: tuple[int, int], *, parent_correct: int = 0) -> dict:
    contract = candidate["candidate_rubric_key"]
    return {
        "instance_id": "fixture__one",
        "current_groups": [
            {
                "group_key": "g0",
                "node_ids": ["left", "right"],
                "node_aliases": {"left": "T1", "right": "T2"},
                "gt_joint_rewards": {"left": 1.0, "right": 0.0},
                "strict_pairs": [{"left": "left", "right": "right", "gt_sign": 1}],
                "rubric_scores": {
                    contract: {
                        "raw_scores": {
                            "left": {"raw_score": raw_scores[0], "evidence": "left"},
                            "right": {"raw_score": raw_scores[1], "evidence": "right"},
                        }
                    }
                },
            }
        ],
        "current_reference_statistics": [
            {"rubric_key": candidate.get("parent_rubric_key"), "strict_correct": parent_correct}
        ],
        "counts": {"strict_gt_difference_pairs": 1},
    }


def test_initial_generation_uses_zero_baseline_not_majority_gate():
    candidate = _candidate("initial_gen")
    result = asyncio.run(
        generate_refine.evaluate_candidate(
            candidate=candidate,
            ledger=_ledger(candidate, (5, 1)),
            semaphore=asyncio.Semaphore(1),
            max_tokens=128,
        )
    )
    assert result["baseline_kind"] == "generation_zero"
    assert result["candidate_correct"] == 1
    assert result["strictly_improves_baseline"] is True


def test_refinement_must_strictly_improve_its_parent():
    parent = "a" * 64
    candidate = _candidate("sol_refine", action="modify", parent=parent)
    result = asyncio.run(
        generate_refine.evaluate_candidate(
            candidate=candidate,
            ledger=_ledger(candidate, (1, 5), parent_correct=0),
            semaphore=asyncio.Semaphore(1),
            max_tokens=128,
        )
    )
    assert result["candidate_correct"] == result["baseline_correct"] == 0
    assert result["strictly_improves_baseline"] is False


def test_later_generation_uses_zero_baseline():
    rejected = _candidate("sol_gen", action="add")
    rejected_result = asyncio.run(
        generate_refine.evaluate_candidate(
            candidate=rejected,
            ledger=_ledger(rejected, (1, 5)),
            semaphore=asyncio.Semaphore(1),
            max_tokens=128,
        )
    )
    accepted = _candidate("sol_gen", action="add")
    accepted_result = asyncio.run(
        generate_refine.evaluate_candidate(
            candidate=accepted,
            ledger=_ledger(accepted, (5, 1)),
            semaphore=asyncio.Semaphore(1),
            max_tokens=128,
        )
    )
    assert rejected_result["baseline_kind"] == "generation_zero"
    assert rejected_result["strictly_improves_baseline"] is False
    assert accepted_result["strictly_improves_baseline"] is True


def test_refinement_cannot_skip_a_failed_candidate():
    original = _candidate("sol_gen", action="add")
    failed = [{"candidate_id": original["candidate_id"]}]
    candidates = {"candidates": [original]}
    with pytest.raises(ValueError, match="cover failed candidates"):
        generate_refine.validate_refinement(
            {"decisions": []}, failed, candidates
        )
    with pytest.raises(ValueError, match="action must be refine"):
        generate_refine.validate_refinement(
            {
                "decisions": [
                    {
                        "candidate_id": original["candidate_id"],
                        "action": "no-fix",
                    }
                ]
            },
            failed,
            candidates,
        )


def test_candidate_filter_keeps_any_record_with_strict_gain(tmp_path: Path):
    candidate = _candidate("sol_refine", action="modify", parent="b" * 64)
    candidate_dir = tmp_path / "candidate"
    evaluation_dir = tmp_path / "evaluation"
    candidate_dir.mkdir()
    evaluation_dir.mkdir()
    (candidate_dir / "fixture__one.json").write_text(
        json.dumps({"status": "success", "candidates": [candidate]}), encoding="utf-8"
    )
    (evaluation_dir / "fixture__one.json").write_text(
        json.dumps(
            {
                "status": "success",
                "records": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "strictly_improves_baseline": True,
                        "candidate_accuracy": 0.1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    accepted = build_weight_problems.load_candidates(
        "fixture__one", [(candidate_dir, evaluation_dir)]
    )
    assert len(accepted) == 1


def test_weight_canonicalization_enforces_final_limits():
    weights = optimize_weights.canonicalize_weights(
        [50, 45, 40, 35, 30, 25, 20, 15]
    )
    assert np.count_nonzero(weights) == 6
    assert set(weights) <= {0, *range(5, 51)}


def test_weight_optimizer_accepts_a_given_mapping_without_search_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optimize_weights.py",
            "--problem-manifest",
            str(tmp_path / "problems.json"),
            "--base-handbook-dir",
            str(tmp_path / "handbooks"),
            "--output-root",
            str(tmp_path / "output"),
        ],
    )
    args = optimize_weights.parse_args()
    assert args.fixed_score_mapping == "1,2,4,6,8"
    assert not hasattr(args, "screen_keep")
    assert not hasattr(args, "refine_keep")
    assert not hasattr(args, "scale_limit")
