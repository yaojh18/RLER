from experience_gen.keywords import (
    query_overlap_candidates,
    sanitize_keyword,
    sanitized_keywords,
)


def test_keyword_static_filter():
    assert sanitize_keyword("QuerySet.query setter") == "QuerySet.query setter"
    assert sanitize_keyword("hidden test reward") is None
    assert sanitize_keyword("bug") is None


def test_keyword_extraction_only_applies_the_static_safety_filter():
    assert sanitized_keywords(
        ["QuerySet.query setter", "generic broad phrase", "hidden test reward"]
    ) == ["QuerySet.query setter", "generic broad phrase"]


def test_query_overlap_is_a_distinct_candidate_source():
    card = {
        "title": "Django grouped queryset ordering",
        "description": "Inspect QuerySet.ordered and compiler GROUP BY behavior.",
        "context": "Use for aggregate annotations with Meta.ordering.",
        "experience": "Preserve explicit ordering while suppressing defaults.",
        "metadata": {},
    }
    candidates = query_overlap_candidates(
        card,
        [{"retrieval_query": "inspect compiler.py annotations and orderings"}],
    )
    assert "annotations" in candidates
    assert "orderings" in candidates
    assert "and" not in candidates
