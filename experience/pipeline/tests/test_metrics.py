from experience_gen.metrics import improved, ranking_metrics


def test_tie_aware_and_pairwise_metrics():
    metrics = ranking_metrics(
        {"a": 1.0, "b": 1.0, "c": 0.0},
        {"a": 2.0, "b": 0.0, "c": -1.0},
        ["a", "b", "c"],
    )
    assert metrics["tie_aware_success"] == 0.5
    assert metrics["pairwise_accuracy"] == 5 / 6


def test_positive_or_gate():
    baseline = {"tie_aware_success": 0.0, "pairwise_accuracy": 0.7}
    candidate = {"tie_aware_success": 0.5, "pairwise_accuracy": 0.6}
    assert improved(candidate, baseline)
