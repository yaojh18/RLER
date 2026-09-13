#!/usr/bin/env python3
"""Sol synthesis/refinement and current-batch Luna rubric evaluation."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
import traceback
from pathlib import Path
from typing import Any

from common import (
    atomic_json,
    load_json,
    rubric_contract,
    rubric_key,
)
from judge_replay import (
    HANDBOOK_RUBRIC_JUDGE_PROMPT,
    JUDGE_TEMPERATURE,
    JUDGE_TOP_P,
    judge_call,
    judge_request_content,
    parse_generation_view,
)


SOL_MODEL = "openai/azure/openai/gpt-5.6-sol"
LUNA_MODEL = "openai/azure/openai/gpt-5.6-luna"
MAX_PROPOSALS = 3
CONTEXT_TOKEN_TARGET = 250_000
APPLIES_WHEN_FORBIDDEN = re.compile(
    r"\b(?:judge|score|scoring|weight|anchor|credit|penali[sz]e|grade)\b|"
    r"\b(?:inspect|look for|compare|verify)\b",
    re.IGNORECASE,
)
LEAKAGE = re.compile(
    r"\b(?:group[_ -]?id|branch[_ -]?id|trajectory[_ -]?alias|gt[_ -]?reward|"
    r"ground[_ -]?truth|private[_ -]?score)\b|\bG\d+\b|\bT\d+\b",
    re.IGNORECASE,
)


SOL_GENERATION_SYSTEM = """
You synthesize a small set of candidate improvements to one instance-specific
golden rubric portfolio for SWE early-trajectory judging. You receive the
current handbook, all current-batch rollout-40 groups for this instance, Luna's
individual rubric scores and evidence, and private final joint values used only
as supervision.

Propose one to three high-impact changes. A proposal is either:
- `modify`: a replacement for one exact current golden rubric while preserving
  its useful semantic dimension; or
- `add`: one genuinely new, non-redundant observable dimension that the current
  golden portfolio does not cover.

The downstream acceptance test is the candidate rubric's own strict
GT-difference pairwise accuracy on all supplied groups. Do not optimize aggregate
portfolio weights here. Diagnose why the current rubric ties or reverses visible
behavior: missing evidence authority, ambiguous scale anchors, wrong polarity,
overlapping scope, failure to distinguish applied work from proposed work, or a
missing causal dimension. A candidate must be usable on future rollouts of this
same task, not memorize these branches.

Preserve cutoff-visible, task-causal evidence only. Never encode aliases, group
IDs, private values, reward orderings, future actions, final outcomes, model
names, or stage progress as a proxy. Legitimate ties remain ties. Do not invent
distinctions when future outcomes cannot be inferred from the visible prefix.

The individual judge sees title, polarity, description, metadata, and scale. Put
precise evidence sources, precedence, prerequisites, non-evidence, and
contradiction caps in metadata and mutually exclusive scale anchors.
`applies_when` is visible only to the rubric generator and must contain selection
conditions, never instructions for how to judge or score evidence.

Output exactly one JSON object:
{
  "proposals": [
    {
      "action": "modify or add",
      "parent_rubric_key": "exact supplied parent id, or null for add",
      "semantic_dimension": "short task-causal dimension",
      "reason": "why this can improve current single-rubric ordering",
      "rubric": {
        "title": "...",
        "polarity": "positive or negative",
        "weight": 1.0,
        "applies_when": ["selection condition only"],
        "description": "observable criterion",
        "metadata": {
          "stage": "...",
          "judge_focus": "specific visible evidence to inspect",
          "evidence_authority": "which visible evidence wins conflicts",
          "hard_gate": "prerequisite for higher anchors",
          "contradiction_rule": "visible evidence that caps credit",
          "oracle_test": "task-specific semantic reference",
          "failure_mode": "common misjudgment this rubric prevents"
        },
        "scale": {"1":"...","2":"...","3":"...","4":"...","5":"..."}
      }
    }
  ]
}
Return no prose outside the JSON object.
""".strip()


INITIAL_GENERATION_SYSTEM = """
You create the initial instance-specific golden rubric portfolio for SWE
early-trajectory judging. You receive historical rollout groups, visible
trajectory prefixes, and private final joint values used only as supervision.

Generate one to six non-redundant, task-causal rubrics. Each rubric must judge
only evidence visible at the cutoff and must distinguish behavior that can
explain strict final-value differences. Do not generate task summaries, good or
bad behavior prose, aliases, group IDs, private values, reward orderings, future
actions, or final outcomes. Legitimate ties remain ties.

Output the same JSON proposal schema as the supplied generation contract. Every
proposal must use action `add` and a null parent_rubric_key. Return no
prose outside the JSON object.
""".strip()


SOL_REFINE_SYSTEM = """
You repair failed candidate golden rubrics for one fixed SWE instance. Each
candidate was already judged by Luna on every current-batch rollout-40 group.
You receive its full wrong-pair evidence, the private final values, the complete
current portfolio, and the visible trajectory views.

For every supplied failed candidate, return one corrected rubric. You may not
delete, skip, or return no-fix: an incorrect repair is rejected downstream, but
an unattempted repair can never improve the bank. Preserve the candidate's
intended semantic dimension; do not turn it into a broad outcome predictor.
Use Luna's actual evidence to repair the precise judge interface:
evidence authority, hard gates, contradictory-evidence caps, mutually exclusive
scale anchors, polarity, or scope. See the other golden rubrics and successful
candidates so the replacement is not redundant.

Never encode aliases, group IDs, scores, private values, reward orderings,
future actions, final outcomes, model names, or progress proxies. `applies_when`
contains only rubric-generator selection conditions. Judge-facing evidence
instructions belong in description, metadata, and scale.

Output exactly one JSON object:
{
  "decisions": [
    {
      "candidate_id": "exact supplied id",
      "action": "refine",
      "reason": "specific causal diagnosis",
      "rubric": {"complete": "rubric schema"}
    }
  ]
}
For refine decisions, `rubric` has the complete schema used by the supplied
candidate, including all five scale anchors and all seven metadata fields.
Return every candidate_id exactly once and no prose outside JSON.
""".strip()


def criterion(rubric: dict[str, Any]) -> str:
    contract = rubric_contract(rubric)
    lines = [
        f"Title: {str(contract['title']).strip()}",
        f"Type: {str(contract['direction']).strip().lower()}",
        f"Description: {str(contract['description']).strip()}",
        "Scale:",
        *[
            f"{score}: {str(contract['scale'][str(score)])}"
            for score in range(1, 6)
        ],
        "Metadata:",
    ]
    for key, value in contract["metadata"].items():
        text = str(value)
        if "\n" not in text:
            lines.append(f"- **{key}**: {text}")
        else:
            lines.append(f"- **{key}**:")
            lines.extend(f"  {line}" for line in text.splitlines())
    return "\n".join(lines)


def validate_rubric(rubric: Any) -> dict[str, Any]:
    if not isinstance(rubric, dict):
        raise ValueError("rubric must be an object")
    if rubric.get("polarity") not in {"positive", "negative"}:
        raise ValueError("rubric polarity must be positive or negative")
    if not isinstance(rubric.get("weight"), (int, float)) or rubric["weight"] <= 0:
        raise ValueError("rubric weight must be positive")
    applies = rubric.get("applies_when")
    if not (
        isinstance(applies, list)
        and applies
        and all(isinstance(value, str) and value.strip() for value in applies)
    ):
        raise ValueError("rubric applies_when must be a nonempty string list")
    if any(APPLIES_WHEN_FORBIDDEN.search(value) for value in applies):
        raise ValueError("applies_when contains judge-facing instructions")
    scale = rubric.get("scale")
    if not isinstance(scale, dict) or set(map(str, scale)) != {"1", "2", "3", "4", "5"}:
        raise ValueError("rubric scale must have exactly anchors 1 through 5")
    for key in ("title", "description"):
        if not isinstance(rubric.get(key), str) or not rubric[key].strip():
            raise ValueError(f"rubric lacks {key}")
    metadata = rubric.get("metadata")
    required = (
        "stage",
        "judge_focus",
        "evidence_authority",
        "hard_gate",
        "contradiction_rule",
        "oracle_test",
        "failure_mode",
    )
    if not isinstance(metadata, dict) or any(
        not isinstance(metadata.get(key), str) or not metadata[key].strip()
        for key in required
    ):
        raise ValueError("rubric metadata lacks required judge-facing fields")
    public_text = json.dumps(rubric, ensure_ascii=False)
    if LEAKAGE.search(public_text):
        raise ValueError("rubric contains supervision or trajectory alias leakage")
    return rubric


def selected_ids(args: argparse.Namespace) -> list[str]:
    ids = list(
        map(
            str,
            load_json(args.collection_root / "instance_manifest.json")[
                "instance_ids"
            ],
        )
    )
    return ids[args.rank :: args.world_size]


def truncate_middle(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    half = (limit - 80) // 2
    return value[:half] + "\n[... middle truncated ...]\n" + value[-half:]


def prompt_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def node_aliases(group: dict[str, Any]) -> dict[str, str]:
    return {
        node_id: f"T{index}"
        for index, node_id in enumerate(group["node_ids"], start=1)
    }


def compact_group(
    group: dict[str, Any], trajectory_limit: int, evidence_limit: int
) -> dict[str, Any]:
    source_path = Path(group["source"])
    source = load_json(source_path)
    view = parse_generation_view(source)
    aliases = node_aliases(group)
    if len(view["continuations"]) != len(group["node_ids"]):
        raise ValueError("continuation/node count drift")
    rubric_scores = []
    for contract_sha, record in group["rubric_scores"].items():
        rubric_scores.append(
            {
                "rubric_key": contract_sha,
                "title": record["rubric"]["title"],
                "strict_correct": record["strict_correct"],
                "strict_pairs": record["strict_pairs"],
                "scores": {
                    aliases[node_id]: {
                        "raw": value["raw_score"],
                        "evidence": truncate_middle(
                            str(value.get("evidence") or ""), evidence_limit
                        ),
                    }
                    for node_id, value in record["raw_scores"].items()
                },
            }
        )
    return {
        "update_step": group["update_step"],
        "trajectories": {
            aliases[node_id]: truncate_middle(trajectory, trajectory_limit)
            for node_id, trajectory in zip(
                group["node_ids"], view["continuations"]
            )
        },
        "current_golden_rubric_outcomes": rubric_scores,
        "private_final_joint_values": {
            aliases[node_id]: value
            for node_id, value in group["gt_joint_rewards"].items()
        },
    }


def build_generation_prompt(ledger: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    handbook_wrapper = load_json(Path(ledger["current_handbook"]))
    handbook = handbook_wrapper.get("handbook", handbook_wrapper)
    first_source = load_json(Path(ledger["current_groups"][0]["source"]))
    problem = parse_generation_view(first_source)["question_text"]
    profiles = ((20_000, 600), (16_000, 500), (12_000, 400), (8_000, 320), (5_000, 240))
    for trajectory_limit, evidence_limit in profiles:
        groups = [
            compact_group(group, trajectory_limit, evidence_limit)
            for group in ledger["current_groups"]
        ]
        payload = {
            "task": problem,
            "current_handbook": handbook,
            "current_reference_single_rubric_statistics": ledger[
                "current_reference_statistics"
            ],
            "current_rollout_groups": groups,
        }
        prompt = SOL_GENERATION_SYSTEM + "\n\n## Instance packet\n" + json.dumps(
            payload, ensure_ascii=False, indent=2
        )
        if prompt_tokens(prompt) <= CONTEXT_TOKEN_TARGET:
            return prompt, {
                "trajectory_char_limit": trajectory_limit,
                "evidence_char_limit": evidence_limit,
                "approximate_prompt_tokens": prompt_tokens(prompt),
            }
    raise ValueError(f"Sol generation prompt exceeds context: {ledger['instance_id']}")


def validate_generation(
    parsed: dict[str, Any],
    ledger: dict[str, Any],
    *,
    initial: bool = False,
) -> list[dict[str, Any]]:
    values = parsed.get("proposals")
    maximum = 6 if initial else MAX_PROPOSALS
    if not isinstance(values, list) or not 1 <= len(values) <= maximum:
        raise ValueError(f"proposals must contain one through {maximum} items")
    references = {
        item["rubric_key"]: item["rubric"]
        for item in ledger["current_reference_statistics"]
    }
    reference_contracts = set(references)
    reference_titles = {item["title"] for item in references.values()}
    candidate_contracts = []
    candidate_titles = []
    normalized = []
    for index, value in enumerate(values):
        if not isinstance(value, dict) or value.get("action") not in {"modify", "add"}:
            raise ValueError("proposal action must be modify or add")
        action = value["action"]
        if initial and action != "add":
            raise ValueError("initial generation may only add rubrics")
        parent = value.get("parent_rubric_key")
        if action == "modify" and parent not in reference_contracts:
            raise ValueError("modify proposal has unknown parent contract")
        if action == "add" and parent is not None:
            raise ValueError("add proposal must have null parent contract")
        if not isinstance(value.get("semantic_dimension"), str) or not value[
            "semantic_dimension"
        ].strip():
            raise ValueError("proposal lacks semantic_dimension")
        if not isinstance(value.get("reason"), str) or not value["reason"].strip():
            raise ValueError("proposal lacks reason")
        rubric = validate_rubric(value.get("rubric"))
        contract_sha = rubric_key(rubric)
        if contract_sha in reference_contracts:
            raise ValueError("proposal duplicates a current reference contract")
        if action == "add" and rubric["title"] in reference_titles:
            raise ValueError("add proposal duplicates a current reference title")
        candidate_contracts.append(contract_sha)
        candidate_titles.append(rubric["title"])
        normalized.append(
            {
                "candidate_id": f"sol-gen-{index}-{contract_sha[:12]}",
                "stage": "initial_gen" if initial else "sol_gen",
                "action": action,
                "parent_rubric_key": parent,
                "semantic_dimension": value["semantic_dimension"],
                "reason": value["reason"],
                "rubric": rubric,
                "candidate_rubric_key": contract_sha,
            }
        )
    if len(candidate_contracts) != len(set(candidate_contracts)):
        raise ValueError("generation contains duplicate candidate contracts")
    if len(candidate_titles) != len(set(candidate_titles)):
        raise ValueError("generation contains duplicate candidate titles")
    return normalized


def build_refine_prompt(
    ledger: dict[str, Any], candidates: dict[str, Any], evaluation: dict[str, Any]
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    by_candidate = {item["candidate_id"]: item for item in candidates["candidates"]}
    failed = [
        item
        for item in evaluation["records"]
        if item.get("strictly_improves_baseline") is not True
    ]
    if not failed:
        return "", [], {"approximate_prompt_tokens": 0}
    feedback = []
    for record in failed:
        candidate = by_candidate[record["candidate_id"]]
        wrong_pairs = record["wrong_pairs"]
        compact_pairs = [
            {
                key: value
                for key, value in pair.items()
                if key != "judge_evidence"
            }
            for pair in wrong_pairs
        ]
        by_group: dict[str, tuple[float, dict[str, Any]]] = {}
        for pair in wrong_pairs:
            group_key = str(pair["group_key"])
            gap = abs(float(pair["gt_values"][0]) - float(pair["gt_values"][1]))
            if group_key not in by_group or gap > by_group[group_key][0]:
                by_group[group_key] = (gap, pair)
        ordered = [value[1] for _, value in sorted(by_group.items())]
        seen = {id(value) for value in ordered}
        ordered.extend(
            pair
            for pair in sorted(
                wrong_pairs,
                key=lambda value: abs(
                    float(value["gt_values"][0]) - float(value["gt_values"][1])
                ),
                reverse=True,
            )
            if id(pair) not in seen
        )
        feedback.append(
            {
                "candidate_id": candidate["candidate_id"],
                "action": candidate["action"],
                "parent_rubric_key": candidate["parent_rubric_key"],
                "semantic_dimension": candidate["semantic_dimension"],
                "candidate_rubric": candidate["rubric"],
                "baseline_correct": record["baseline_correct"],
                "candidate_correct": record["candidate_correct"],
                "strict_pairs": record["strict_pairs"],
                "wrong_pairs": compact_pairs,
                "representative_wrong_pair_evidence": [
                    {
                        **{
                            key: value
                            for key, value in pair.items()
                            if key != "judge_evidence"
                        },
                        "judge_evidence": [
                            truncate_middle(str(value), 1_200)
                            for value in pair["judge_evidence"]
                        ],
                    }
                    for pair in ordered[:16]
                ],
            }
        )
    generation_prompt, profile = build_generation_prompt(ledger)
    packet_marker = generation_prompt.index("## Instance packet")
    prompt = "\n\n".join(
        [
            SOL_REFINE_SYSTEM,
            generation_prompt[packet_marker:],
            "## Failed candidate Luna outcomes",
            json.dumps(feedback, ensure_ascii=False, indent=2),
        ]
    )
    if prompt_tokens(prompt) > CONTEXT_TOKEN_TARGET:
        raise ValueError(f"Sol refinement prompt exceeds context: {ledger['instance_id']}")
    profile["approximate_prompt_tokens"] = prompt_tokens(prompt)
    return prompt, failed, profile


def validate_refinement(
    parsed: dict[str, Any], failed: list[dict[str, Any]], candidates: dict[str, Any]
) -> list[dict[str, Any]]:
    decisions = parsed.get("decisions")
    expected = [item["candidate_id"] for item in failed]
    if not isinstance(decisions, list) or [item.get("candidate_id") for item in decisions] != expected:
        raise ValueError("refinement decisions must cover failed candidates in order")
    source = {item["candidate_id"]: item for item in candidates["candidates"]}
    output = []
    contracts = []
    titles = []
    for value in decisions:
        candidate_id = value["candidate_id"]
        if value.get("action") != "refine":
            raise ValueError("refinement action must be refine")
        rubric = validate_rubric(value.get("rubric"))
        contract_sha = rubric_key(rubric)
        if contract_sha == source[candidate_id]["candidate_rubric_key"]:
            raise ValueError("refinement did not change the candidate contract")
        contracts.append(contract_sha)
        titles.append(rubric["title"])
        output.append(
            {
                "candidate_id": f"sol-refine-{candidate_id}-{contract_sha[:12]}",
                "stage": "sol_refine",
                "action": source[candidate_id]["action"],
                "parent_rubric_key": source[candidate_id][
                    "parent_rubric_key"
                ],
                "semantic_dimension": source[candidate_id]["semantic_dimension"],
                "reason": str(value.get("reason") or ""),
                "source_candidate_id": candidate_id,
                "rubric": rubric,
                "candidate_rubric_key": contract_sha,
            }
        )
    if len(contracts) != len(set(contracts)) or len(titles) != len(set(titles)):
        raise ValueError("refinement contains duplicate rubrics")
    return output


async def call_sol(
    *, prompt: str, validator: Any, semaphore: asyncio.Semaphore
) -> tuple[dict[str, Any], Any, list[str]]:
    from model_runtime import call_json

    conversation: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    errors = []
    async with semaphore:
        for attempt in range(3):
            result = await call_json(
                route_name="final_golden_rubric_refinement",
                model=SOL_MODEL,
                messages=conversation,
                temperature=0.02,
                max_tokens=20_480,
                correction_rounds=8,
            )
            try:
                validated = validator(result["parsed"])
            except ValueError as exc:
                errors.append(str(exc))
                if attempt == 2:
                    raise
                conversation = result["messages"] + [
                    {
                        "role": "user",
                        "content": (
                            f"Semantic validation error: {exc}\n"
                            "Return one complete corrected JSON object."
                        ),
                    }
                ]
                continue
            return result, validated, errors
    raise AssertionError("unreachable Sol validation loop")


async def run_sol(args: argparse.Namespace, *, refine: bool) -> None:
    ids = selected_ids(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(instance_id: str) -> None:
        output = args.output_dir / f"{instance_id}.json"
        ledger_path = args.collection_root / "current_ledgers" / f"{instance_id}.json"
        ledger = load_json(ledger_path)
        if refine:
            candidate_path = args.candidate_dir / f"{instance_id}.json"
            evaluation_path = args.evaluation_dir / f"{instance_id}.json"
        if output.is_file() and not args.force:
            cached = load_json(output)
            if cached.get("status") == "success":
                return
            raise ValueError(f"incomplete cached Sol artifact: {output}")
        try:
            if not refine:
                prompt, profile = build_generation_prompt(ledger)
                initial = args.command == "initial-gen"
                if initial:
                    prompt = INITIAL_GENERATION_SYSTEM + prompt[prompt.index("\n\n## Instance packet") :]
                result, candidates, semantic_errors = await call_sol(
                    prompt=prompt,
                    validator=lambda parsed: validate_generation(
                        parsed, ledger, initial=initial
                    ),
                    semaphore=semaphore,
                )
            else:
                candidates_payload = load_json(candidate_path)
                evaluation_payload = load_json(evaluation_path)
                prompt, failed, profile = build_refine_prompt(
                    ledger, candidates_payload, evaluation_payload
                )
                if not failed:
                    atomic_json(
                        output,
                        {
                            "schema_version": "final_golden_sol_refinement.v1",
                            "status": "success",
                            "instance_id": instance_id,
                            "model": SOL_MODEL,
                            "temperature": 0.02,
                            "profile": profile,
                            "candidates": [],
                            "messages": [],
                            "semantic_validation_errors": [],
                        },
                        sort_keys=True,
                    )
                    return
                result, candidates, semantic_errors = await call_sol(
                    prompt=prompt,
                    validator=lambda parsed: validate_refinement(
                        parsed, failed, candidates_payload
                    ),
                    semaphore=semaphore,
                )
            atomic_json(
                output,
                {
                    "schema_version": (
                        "final_golden_sol_refinement.v1"
                        if refine
                        else (
                            "initial_golden_sol_generation.v1"
                            if args.command == "initial-gen"
                            else "final_golden_sol_generation.v1"
                        )
                    ),
                    "status": "success",
                    "instance_id": instance_id,
                    "model": SOL_MODEL,
                    "temperature": 0.02,
                    "profile": profile,
                    "candidates": candidates,
                    "messages": result["messages"],
                    "semantic_validation_errors": semantic_errors,
                },
                sort_keys=True,
            )
        except Exception as exc:
            atomic_json(
                output.with_suffix(".error.json"),
                {
                    "instance_id": instance_id,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                sort_keys=True,
            )
            raise

    await asyncio.gather(*(one(instance_id) for instance_id in ids))


def candidate_values(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("status") != "success":
        raise ValueError("candidate source is not successful")
    return [item for item in payload.get("candidates") or [] if item.get("rubric")]


def exact_source_records(
    group: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    contract_sha = candidate["candidate_rubric_key"]
    current = group["rubric_scores"].get(contract_sha)
    if current is not None:
        return copy.deepcopy(current["raw_scores"])
    source = load_json(Path(group["source"]))
    candidate_criterion = criterion(candidate["rubric"])
    matches = [
        index
        for index, rubric in enumerate(source["rubrics"])
        if criterion(rubric) == candidate_criterion
    ]
    if len(matches) > 1:
        raise ValueError("candidate matches duplicate adaptive rubrics")
    if not matches:
        return {}
    index = matches[0]
    output = {}
    for node_id, records in zip(group["node_ids"], source["score_records"]):
        record = records[index]
        output[node_id] = {
            "raw_score": int(record["score_raw"]),
            "evidence": str(record.get("evidence") or ""),
            "origin": "current_adaptive_exact_reuse",
        }
    return output


async def evaluate_candidate(
    *,
    candidate: dict[str, Any],
    ledger: dict[str, Any],
    semaphore: asyncio.Semaphore,
    max_tokens: int,
) -> dict[str, Any]:
    judge_prompt = HANDBOOK_RUBRIC_JUDGE_PROMPT
    group_outcomes = []
    total_correct = total_pairs = fresh = reused = 0
    all_wrong = []
    for group in ledger["current_groups"]:
        aliases = node_aliases(group)
        records = exact_source_records(group, candidate)
        missing_nodes = [node for node in group["node_ids"] if node not in records]
        if missing_nodes:
            source = load_json(Path(group["source"]))
            view = parse_generation_view(source)
            node_index = {node: index for index, node in enumerate(group["node_ids"])}
            calls = []
            for node_id in missing_nodes:
                prompt = judge_request_content(
                    judge_prompt,
                    view,
                    node_index[node_id],
                    criterion(candidate["rubric"]),
                )
                calls.append(
                    judge_call(
                        request=[{"role": "user", "content": prompt}],
                        semaphore=semaphore,
                        model_name=LUNA_MODEL,
                        max_tokens=max_tokens,
                    )
                )
            results = await asyncio.gather(*calls)
            for node_id, result in zip(missing_nodes, results):
                records[node_id] = {
                    "raw_score": result["score_raw"],
                    "evidence": result["evidence"],
                    "origin": "current_candidate_luna",
                    "messages": result["messages"],
                    "format_errors": result["format_errors"],
                }
                fresh += 1
        reused += len(group["node_ids"]) - len(missing_nodes)
        sign = -1 if candidate["rubric"]["polarity"] == "negative" else 1
        correct = 0
        wrong = []
        for pair in group["strict_pairs"]:
            left = pair["left"]
            right = pair["right"]
            predicted = (sign * records[left]["raw_score"] > sign * records[right]["raw_score"]) - (
                sign * records[left]["raw_score"] < sign * records[right]["raw_score"]
            )
            if predicted == pair["gt_sign"]:
                correct += 1
            else:
                item = {
                    "group_key": group["group_key"],
                    "pair": [aliases[left], aliases[right]],
                    "node_ids": [left, right],
                    "gt_values": [
                        group["gt_joint_rewards"][left],
                        group["gt_joint_rewards"][right],
                    ],
                    "raw_scores": [records[left]["raw_score"], records[right]["raw_score"]],
                    "judge_evidence": [records[left]["evidence"], records[right]["evidence"]],
                }
                wrong.append(item)
                all_wrong.append(item)
        total_correct += correct
        total_pairs += len(group["strict_pairs"])
        group_outcomes.append(
            {
                "group_key": group["group_key"],
                "strict_correct": correct,
                "strict_pairs": len(group["strict_pairs"]),
                "raw_scores": records,
                "wrong_pairs": wrong,
            }
        )
    if total_pairs != ledger["counts"]["strict_gt_difference_pairs"]:
        raise ValueError("candidate evaluation denominator drift")
    parent = candidate.get("parent_rubric_key")
    if candidate["action"] == "add":
        baseline_correct = 0
        improves = total_correct > 0
        baseline_kind = "generation_zero"
    else:
        baseline = next(
            item
            for item in ledger["current_reference_statistics"]
            if item["rubric_key"] == parent
        )
        baseline_correct = int(baseline["strict_correct"])
        improves = total_correct > baseline_correct
        baseline_kind = "parent_single_rubric"
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_rubric_key": candidate["candidate_rubric_key"],
        "action": candidate["action"],
        "parent_rubric_key": parent,
        "rubric": candidate["rubric"],
        "baseline_kind": baseline_kind,
        "baseline_correct": baseline_correct,
        "candidate_correct": total_correct,
        "strict_pairs": total_pairs,
        "candidate_accuracy": total_correct / total_pairs,
        "strictly_improves_baseline": improves,
        "fresh_luna_node_scores": fresh,
        "reused_exact_node_scores": reused,
        "group_outcomes": group_outcomes,
        "wrong_pairs": all_wrong,
    }


async def run_luna_eval(args: argparse.Namespace) -> None:
    ids = selected_ids(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    group_limit = asyncio.Semaphore(args.instance_concurrency)

    async def one(instance_id: str) -> None:
        async with group_limit:
            output = args.output_dir / f"{instance_id}.json"
            candidate_path = args.candidate_dir / f"{instance_id}.json"
            ledger_path = args.collection_root / "current_ledgers" / f"{instance_id}.json"
            if output.is_file() and not args.force:
                cached = load_json(output)
                if cached.get("status") == "success":
                    return
                raise ValueError(f"incomplete cached Luna artifact: {output}")
            payload = load_json(candidate_path)
            ledger = load_json(ledger_path)
            records = []
            for candidate in candidate_values(payload):
                records.append(
                    await evaluate_candidate(
                        candidate=candidate,
                        ledger=ledger,
                        semaphore=semaphore,
                        max_tokens=args.max_tokens,
                    )
                )
            atomic_json(
                output,
                {
                    "schema_version": "final_golden_current_candidate_evaluation.v1",
                    "status": "success",
                    "stage": args.stage,
                    "instance_id": instance_id,
                    "model": LUNA_MODEL,
                    "temperature": JUDGE_TEMPERATURE,
                    "top_p": JUDGE_TOP_P,
                    "max_tokens": args.max_tokens,
                    "records": records,
                },
                sort_keys=True,
            )

    await asyncio.gather(*(one(instance_id) for instance_id in ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("initial-gen", "sol-gen", "sol-refine"):
        child = sub.add_parser(name)
        child.add_argument("--collection-root", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--rank", type=int, default=0)
        child.add_argument("--world-size", type=int, default=1)
        child.add_argument("--concurrency", type=int, default=4)
        child.add_argument("--force", action="store_true")
        if name == "sol-refine":
            child.add_argument("--candidate-dir", type=Path, required=True)
            child.add_argument("--evaluation-dir", type=Path, required=True)
    child = sub.add_parser("luna-eval")
    child.add_argument("--collection-root", type=Path, required=True)
    child.add_argument("--candidate-dir", type=Path, required=True)
    child.add_argument("--output-dir", type=Path, required=True)
    child.add_argument("--stage", required=True)
    child.add_argument("--rank", type=int, default=0)
    child.add_argument("--world-size", type=int, default=1)
    child.add_argument("--concurrency", type=int, default=32)
    child.add_argument("--instance-concurrency", type=int, default=8)
    child.add_argument("--max-tokens", type=int, default=20_480)
    child.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError("invalid rank/world-size")
    if args.command in {"initial-gen", "sol-gen"}:
        asyncio.run(run_sol(args, refine=False))
    elif args.command == "sol-refine":
        asyncio.run(run_sol(args, refine=True))
    elif args.command == "luna-eval":
        asyncio.run(run_luna_eval(args))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
