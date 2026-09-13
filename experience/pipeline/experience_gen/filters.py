from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable

from .metrics import improved, metric_delta


REQUIRED_CARD_FIELDS = ("experience_id", "title", "description", "context", "experience")
FORBIDDEN_PUBLIC_PATTERNS = {
    "node_id": re.compile(r"\bnode[-_ ]?id\b", re.I),
    "terminal reward": re.compile(r"\b(?:terminal|ground[- ]truth|gt)[-_ ]?rewards?\b", re.I),
    "hidden test": re.compile(r"\bhidden tests?\b", re.I),
    "golden patch": re.compile(r"\bgolden (?:patch|solution)\b", re.I),
    "experiment identity": re.compile(r"\b(?:this|the) experiment\b|\bv\d{2,3}\b", re.I),
    "instance identity": re.compile(r"\binstance[-_ ]?id\b", re.I),
}


@dataclass(frozen=True)
class CandidateDecision:
    accepted: bool
    reasons: tuple[str, ...]
    delta: dict[str, float]


def normalize_card(
    parsed: dict[str, Any] | None,
    *,
    experience_id: str,
) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    content = parsed.get("content") if isinstance(parsed.get("content"), dict) else parsed
    required = ("title", "description", "context", "experience")
    if any(not isinstance(content.get(key), str) or not content[key].strip() for key in required):
        return None
    metadata = content.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    references = metadata.get("reference_golden_rubrics")
    if not isinstance(references, list):
        references = []
    return {
        "experience_id": experience_id,
        "title": content["title"].strip(),
        "description": content["description"].strip(),
        "context": content["context"].strip(),
        "experience": content["experience"].strip(),
        "metadata": {
            "reference_golden_rubrics": references,
        },
    }


def public_card_errors(card: dict[str, Any]) -> list[str]:
    errors = []
    for field in REQUIRED_CARD_FIELDS:
        if not str(card.get(field) or "").strip():
            errors.append(f"missing {field}")
    if not isinstance(card.get("metadata"), dict):
        errors.append("metadata is not an object")
    public_text = "\n".join(
        str(card.get(field) or "")
        for field in ("title", "description", "context", "experience")
    )
    for name, pattern in FORBIDDEN_PUBLIC_PATTERNS.items():
        if pattern.search(public_text):
            errors.append(f"public leakage: {name}")
    return errors


def duplicate_errors(
    card: dict[str, Any],
    existing: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.94,
) -> list[str]:
    title = str(card.get("title") or "").strip().casefold()
    body = "\n".join(str(card.get(key) or "") for key in ("context", "experience")).casefold()
    for candidate in existing:
        if title == str(candidate.get("title") or "").strip().casefold():
            return [f"duplicate title: {candidate.get('experience_id')}"]
        other = "\n".join(
            str(candidate.get(key) or "") for key in ("context", "experience")
        ).casefold()
        if body and other and SequenceMatcher(None, body, other).ratio() >= threshold:
            return [f"near-duplicate content: {candidate.get('experience_id')}"]
    return []


def evaluate_candidate(
    *,
    card: dict[str, Any],
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    errors: list[dict[str, Any]] | None = None,
    existing_cards: Iterable[dict[str, Any]] = (),
) -> CandidateDecision:
    reasons = [
        *public_card_errors(card),
        *duplicate_errors(card, existing_cards),
    ]
    if errors:
        reasons.append("rubric/judge evaluation returned errors")
    if not improved(candidate_metrics, baseline_metrics):
        reasons.append("neither tie-aware nor pairwise metric improved")
    return CandidateDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        delta=metric_delta(candidate_metrics, baseline_metrics),
    )
