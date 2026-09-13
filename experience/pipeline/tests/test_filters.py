from experience_gen.filters import evaluate_candidate, normalize_card


def card():
    return {
        "experience_id": "EXP_A",
        "title": "Preserve inherited implementation evidence",
        "description": "Use when sibling branches share a parent implementation.",
        "context": "Apply only when the parent state visibly contains the change.",
        "experience": "Attribute the parent fact equally and rank only child deltas.",
        "metadata": {"reference_golden_rubrics": []},
    }


def test_normalize_card():
    value = normalize_card({"content": card()}, experience_id="EXP_B")
    assert value["experience_id"] == "EXP_B"
    assert value["title"] == card()["title"]


def test_candidate_gate_rejects_judge_error():
    decision = evaluate_candidate(
        card=card(),
        baseline_metrics={"tie_aware_success": 0.0, "pairwise_accuracy": 0.5},
        candidate_metrics={"tie_aware_success": 1.0, "pairwise_accuracy": 0.8},
        errors=[{"error": "InvalidJudgeResponse"}],
    )
    assert not decision.accepted
    assert "rubric/judge evaluation returned errors" in decision.reasons

