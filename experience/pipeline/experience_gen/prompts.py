from __future__ import annotations

import json
from typing import Any


CARD_SCHEMA = """{
  "title": "short reusable title",
  "description": "when this experience should be considered",
  "context": "visible activation, exclusion, and abstention boundaries",
  "experience": "specific evidence-reading or rubric-generation policy",
  "metadata": {"reference_golden_rubrics": []}
}"""

GENERATION_SYSTEM_PROMPT = """You diagnose one concrete SWE trajectory-selection
failure and write one task-specific experience for a second-pass evaluator.
Reason freely, then append exactly one JSON object matching the requested schema."""

REFINE_SYSTEM_PROMPT = """You are the refinement stage for SWE trajectory-judge
experiences. Correct the measured failure of the previous card without leaking
privileged outcomes. Reason freely, then append exactly one valid JSON object."""

KEYWORD_SYSTEM_PROMPT = """You extract compact visible retrieval phrases for one
SWE trajectory-judge experience. Prefer task contracts, implementation boundaries,
symbols, stage signals, and failure modes. Do not use hidden outcomes or IDs."""


def compact_markdown(value: Any) -> str:
    if isinstance(value, dict):
        rows = []
        for key, child in value.items():
            rendered = compact_markdown(child)
            if "\n" in rendered:
                indented = "\n".join(
                    f"  {line}" for line in rendered.splitlines()
                )
                rows.append(f"- **{key}**:\n{indented}")
            else:
                rows.append(f"- **{key}**: {rendered}")
        return "\n".join(rows) or "None"
    if isinstance(value, list):
        return "\n".join(f"- {compact_markdown(child)}" for child in value) or "None"
    if value is None:
        return "None"
    return str(value)


def generation_prompt(
    *,
    context: dict[str, Any],
    scope: str,
    track: str,
    diagnostic: dict[str, Any],
    attempt: int,
    previous: dict[str, Any] | None = None,
) -> str:
    if track == "rubric":
        role = (
            "a second-pass rubric portfolio editor after initial rubrics are generated "
            "and before any replacement scores are produced"
        )
        defect = (
            "coverage, semantic alignment, redundancy, weight allocation, stage "
            "alignment, tie discipline, or displacement of the primary contract"
        )
    elif track == "judge":
        role = "a second-pass evidence-first rubric score judge"
        defect = (
            "evidence attribution, anchor selection, polarity, parent/child provenance, "
            "validation provenance, or unsupported score changes"
        )
    else:
        raise ValueError("track must be rubric or judge")

    privileged = {
        "error_analysis": diagnostic.get("analysis"),
        "error_pattern": diagnostic.get("primary_cluster"),
        "baseline_metrics": diagnostic.get("baseline_metrics"),
        "baseline_rubrics": diagnostic.get("rubrics"),
        "baseline_score_by_rubric": diagnostic.get("score_by_rubric"),
        "key_visible_evidence": diagnostic.get("key_visible_evidence"),
        "terminal_rewards_for_diagnosis_only": context["gt_rewards"],
    }
    retry = ""
    if previous:
        retry += (
            "\n\n## Previous attempted card and measured result\n"
            + compact_markdown(previous)
            + "\nIt failed to improve both target metrics. Diagnose why it did not alter "
            "the rubric/judge behavior and make a substantive correction, not a paraphrase."
        )
    return f"""## Objective
Create one deliberately narrow experience for {role}. It must correct this round's
observed {defect} failure and is evaluated alone in the `{scope}` bank.

The privileged diagnostic packet may be used only to discover the bias. The public
card must state what visible evidence to inspect and how to act. It must not mention
rewards, hidden tests, node IDs, instance IDs, golden patches, this experiment, or
future outcomes. Do not prescribe an exact patch when equivalent implementations are
valid. Include:

- a narrow activation condition;
- decisive positive and contradictory evidence;
- close-looking cases that must be excluded;
- an explicit abstention rule when the prefix cannot distinguish candidates.

Do not merely restate the issue. End with one JSON object, either directly or under
`content`, matching:

{CARD_SCHEMA}

## Attempt
{attempt} of 2

## Privileged diagnostic packet
{compact_markdown(privileged)}

## Visible paused judge view
{compact_markdown(context["view"])}{retry}"""


def keyword_prompt(card: dict[str, Any], *, max_keywords: int = 24) -> str:
    public = {
        key: card.get(key)
        for key in ("title", "description", "context", "experience", "metadata")
    }
    return f"""Generate at most {max_keywords} retrieval phrases for this public card.
Each phrase must be observable before terminal evaluation. Cover, where present:

- literal task contract and behavioral transition;
- implementation boundary, call site, path, API, exception, or symbol;
- trajectory stage and visible branch discriminator;
- activation failure mode and abstention/exclusion clue.

Reject generic SWE words, card IDs, repository/instance labels, outcome labels,
rewards, hidden tests, golden patches, and phrases true only after completion.
Prefer 2-8 token phrases and preserve exact code symbols where useful.

Return {{"keywords":["..."],"rejected":[{{"phrase":"...","reason":"..."}}]}}.

## Card
{json.dumps(public, indent=2, ensure_ascii=False)}"""
