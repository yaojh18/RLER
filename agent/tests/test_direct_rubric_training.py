from __future__ import annotations

import math

from swe_agent.collapse_detector import trajectory_collapse_reason
from swe_agent.direct_rubric_judge import (
    DIRECT_ABSTAIN_SYSTEM_PROMPT,
    DIRECT_RUBRIC_JUDGE_PROMPT,
    DirectRubricBank,
    _numeric_prediction,
    _weighted_scores,
)
from swe_agent.variance_detector import load_variance_detector


def _assistant(content: str, command: str | None = None) -> dict:
    message = {"role": "assistant", "content": content}
    if command is not None:
        message["tool_calls"] = [{"command": command}]
    return message


def test_frozen_direct_assets_supply_training_configuration():
    bank = DirectRubricBank()
    detector = load_variance_detector()

    assert len(bank.payload["instances"]) == 452
    assert len(bank.payload["eligible_instance_ids"]) == 452
    assert bank.score_mapping == (1.0, 2.0, 4.0, 6.0, 8.0)
    assert detector["threshold"] == 0.18
    assert detector["variance_direction"] == "ge"


def test_direct_prompts_preserve_frozen_reference_only_contract():
    assert "all eight prefixes" in DIRECT_ABSTAIN_SYSTEM_PROMPT
    assert "reference rubrics do not cover a visible difference" in (
        DIRECT_ABSTAIN_SYSTEM_PROMPT
    )
    assert "latest workspace summary and optional cutoff git diff" in (
        DIRECT_RUBRIC_JUDGE_PROMPT
    )
    assert "Match the most specific applicable anchor" in (
        DIRECT_RUBRIC_JUDGE_PROMPT
    )


def test_direct_weighted_score_uses_frozen_scale_and_weights():
    bank = DirectRubricBank()
    instance_id = bank.payload["eligible_instance_ids"][0]
    rubrics = bank.rubrics(instance_id, "fix the bug")
    node_ids = ["a", "b"]
    score_records = [
        [
            {"rubric_id": rubric.rubric_id, "score_raw": raw}
            for rubric in rubrics
        ]
        for raw in (1, 5)
    ]

    rewards, signed, range_mean = _weighted_scores(
        node_ids=node_ids,
        score_records=score_records,
        rubrics=rubrics,
        mapping=bank.score_mapping,
    )

    assert set(rewards) == set(node_ids)
    assert set(signed) == set(node_ids)
    assert all(math.isfinite(value) for value in rewards.values())
    assert math.isclose(range_mean, 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_serial_numeric_detector_respects_frozen_boundary():
    detector = load_variance_detector()

    assert _numeric_prediction(0.17, detector)["predicts_zero_variance"]
    assert _numeric_prediction(0.18, detector)["predicts_variance"]


def test_collapse_detector_is_prefix_bounded_and_recovery_aware():
    messages: list[dict] = []
    for index in range(10):
        messages.extend(
            [
                _assistant(f"attempt {index}"),
                {"role": "user", "content": f"Format error: attempt {index}"},
            ]
        )
    assert trajectory_collapse_reason(
        messages, assistant_step_limit=40
    ) == "format_error_parser_loop:total=10:streak=10"

    for index in range(4):
        messages.extend(
            [
                _assistant(
                    f"productive source check {index}",
                    f"sed -n '{index + 1}p' source.py",
                ),
                {"role": "user", "content": f"source line {index}"},
            ]
        )
    assert trajectory_collapse_reason(messages) is None
