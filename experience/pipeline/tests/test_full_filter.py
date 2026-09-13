from __future__ import annotations

import json
from pathlib import Path

from experience_gen.pipeline import case_slug, full_replay_decisions


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_landed_v107_format_filters_by_retrieval_mean(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    for scope in ("siblings", "pc"):
        _write(
            checkpoint / f"bank/{scope}/experience_bank.json",
            {
                "experiences": [
                    {"experience_id": f"{scope}-good", "title": "good"},
                    {"experience_id": f"{scope}-bad", "title": "bad"},
                    {"experience_id": f"{scope}-unused", "title": "unused"},
                ]
            },
        )
    _write(
        checkpoint / "retrieval/keyword_bank.json",
        {
            "bank": {
                scope: {
                    f"{scope}-{name}": {
                        "generated_keywords": [name],
                        "selected_keywords": [name],
                    }
                    for name in ("good", "bad", "unused")
                }
                for scope in ("siblings", "pc")
            }
        },
    )
    keyword_bank = json.loads(
        (checkpoint / "retrieval/keyword_bank.json").read_text()
    )
    keyword_bank["bank"]["pc"]["pc-good"]["selected_keywords"] = [
        "good",
        "missing phrase",
    ]
    _write(checkpoint / "retrieval/keyword_bank.json", keyword_bank)
    replay = tmp_path / "replay"
    contexts = [
        {
            "context_id": "repo__repo-1:R1",
            "instance_id": "repo__repo-1",
            "round_index": 1,
            "view": {"question": "good evidence is visible"},
            "scope_metrics": {
                "siblings": {"pairwise_accuracy": 0.4},
                "pc": {"pairwise_accuracy": 0.4},
            },
        }
    ]
    events = [
        {
            "case_id": f"{scope}::repo__repo-1:R1",
            "context_id": "repo__repo-1:R1",
            "instance_id": "repo__repo-1",
            "round_index": 1,
            "scope": scope,
        }
        for scope in ("siblings", "pc")
    ]
    _write(
        replay / "final_scope.json",
        {
            "population_definition": "test",
            "contexts": contexts,
            "events": events,
            "counts": {"contexts": 1, "instances": 1, "events": 2},
        },
    )
    _write(replay / "run_summary.json", {"targets": 2})
    for scope in ("siblings", "pc"):
        _write(
            replay
            / "runs"
            / case_slug("repo__repo-1:R1", scope)
            / "result.json",
            {
                "status": "success",
                "context_id": "repo__repo-1:R1",
                "scope": scope,
                "metrics": {"pairwise_accuracy": 0.75},
                "retrieved_experience_ids": [
                    f"{scope}-good",
                    f"{scope}-bad",
                ],
            },
        )
    sibling_bad = (
        replay
        / "runs"
        / case_slug("repo__repo-1:R1", "siblings")
        / "result.json"
    )
    value = json.loads(sibling_bad.read_text())
    value["metrics"]["pairwise_accuracy"] = 0.25
    value["retrieved_experience_ids"] = ["siblings-bad"]
    _write(sibling_bad, value)

    result = full_replay_decisions(
        checkpoint=checkpoint,
        replay_root=replay,
        threshold=0.5,
    )
    decisions = {
        (row["scope"], row["experience_id"]): row
        for row in result["decisions"]
    }
    assert decisions[("siblings", "siblings-bad")]["keep"] is False
    assert decisions[("siblings", "siblings-good")]["reason"] == "never_retrieved"
    assert decisions[("pc", "pc-good")]["keep"] is True
    assert decisions[("pc", "pc-unused")]["reason"] == "never_retrieved"
    assert decisions[("pc", "pc-good")]["keyword_selection"] == {
        "selected_keywords": ["good"],
        "policy": "positive_joint_replay_evidence",
        "candidates": [
            {
                "keyword": "good",
                "joint_replay_hits": 1,
                "mean_judge_pairwise_accuracy": 0.75,
                "selected": True,
            },
            {
                "keyword": "missing phrase",
                "joint_replay_hits": 0,
                "mean_judge_pairwise_accuracy": None,
                "selected": False,
            },
        ],
    }
