from __future__ import annotations

import asyncio
import json

from experience_gen.generation import ExperienceGenerator
from experience_gen.cli import build_parser
from experience_gen.replay import TrajectoryReplayEvaluator
from experience_gen.pipeline import generate_overfits


def _context():
    return {
        "context_id": "repo__repo-1:R1",
        "instance_id": "repo__repo-1",
        "round_index": 1,
        "node_ids": ["a", "b"],
        "gt_rewards": {"a": 2.0, "b": 0.0},
        "scope_metrics": {
            "siblings": {
                "tie_aware_success": 0.0,
                "pairwise_accuracy": 0.5,
            }
        },
        "view": {
            "question": {"system_prompt": "", "user_prompt": "task"},
            "previous_state": {},
            "parent_trajectory": None,
            "continuations": [
                {"node_id": "a", "summary": {}},
                {"node_id": "b", "summary": {}},
            ],
        },
    }


def test_glm_then_gpt_keeps_only_locally_improving_attempt(
    monkeypatch, tmp_path
) -> None:
    generator = ExperienceGenerator()
    calls = []

    async def fake_call(**kwargs):
        calls.append(kwargs["model"])
        index = len(calls)
        return (
            {
                "content": {
                    "title": f"Visible contract evidence {index}",
                    "description": "Use when branches expose different contract evidence.",
                    "context": "Apply only to evidence visible in the paused state.",
                    "experience": "Prefer the branch with direct falsifiable evidence.",
                    "metadata": {"reference_golden_rubrics": []},
                }
            },
            [{"role": "assistant", "content": "{}"}],
        )

    evaluations = iter(
        [
            {
                "baseline_metrics": {
                    "tie_aware_success": 0.0,
                    "pairwise_accuracy": 0.5,
                },
                "candidate_metrics": {
                    "tie_aware_success": 0.0,
                    "pairwise_accuracy": 0.5,
                },
                "judge_errors": [],
            },
            {
                "baseline_metrics": {
                    "tie_aware_success": 0.0,
                    "pairwise_accuracy": 0.5,
                },
                "candidate_metrics": {
                    "tie_aware_success": 1.0,
                    "pairwise_accuracy": 0.75,
                },
                "judge_errors": [],
            },
        ]
    )

    async def fake_evaluate(self, **kwargs):
        return next(evaluations)

    monkeypatch.setattr(generator.client, "call", fake_call)
    monkeypatch.setattr(TrajectoryReplayEvaluator, "evaluate", fake_evaluate)
    result = asyncio.run(
        generator.generate_and_evaluate(
            context=_context(),
            scope="siblings",
            track="rubric",
            output_dir=tmp_path,
        )
    )
    assert calls == [
        "nvidia/zai-org/glm-5.2",
        "openai/openai/gpt-5.5",
    ]
    assert [row["accepted"] for row in result["attempts"]] == [False, True]
    assert result["accepted_attempt"]["attempt"] == 2
    assert (
        result["attempts"][0]["card"]["experience_id"]
        == result["attempts"][1]["card"]["experience_id"]
    )


def test_cli_exposes_only_complete_automatic_stages() -> None:
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    assert set(subparsers.choices) == {
        "prepare-contexts",
        "generate-overfits",
        "build-provisional",
        "full-replay",
        "filter-full-replay",
    }
    for command, canonical_input in (
        ("generate-overfits", "--optimization-contexts"),
        ("build-provisional", "--optimization-contexts"),
        ("full-replay", "--validation-contexts"),
    ):
        option_strings = {
            option
            for action in subparsers.choices[command]._actions
            for option in action.option_strings
        }
        assert canonical_input in option_strings
        assert "--contexts" not in option_strings


def test_generation_targets_only_historical_selection_errors(
    monkeypatch, tmp_path
) -> None:
    failed = {**_context(), "selected_node_id": "b", "selection_success": False}
    passed = {
        **_context(),
        "context_id": "repo__repo-1:R2",
        "round_index": 2,
        "selected_node_id": "a",
        "selection_success": True,
    }
    checkpoint = tmp_path / "checkpoint"
    bank = {"experiences": []}
    for scope in ("siblings", "pc"):
        path = checkpoint / f"bank/{scope}/experience_bank.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bank), encoding="utf-8")

    seen = []

    class FakeGenerator:
        def __init__(self, models=None):
            pass

        async def generate_and_evaluate(self, **kwargs):
            seen.append(kwargs["context"]["context_id"])
            return {"status": "filtered", "accepted_attempt": None}

    monkeypatch.setattr("experience_gen.generation.ExperienceGenerator", FakeGenerator)
    result = asyncio.run(
        generate_overfits(
            contexts=[failed, passed],
            base_checkpoint=checkpoint,
            output=tmp_path / "output",
            scopes=("siblings",),
        )
    )
    assert seen == ["repo__repo-1:R1"]
    assert result["targets"] == 1
