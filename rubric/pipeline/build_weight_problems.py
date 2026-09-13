#!/usr/bin/env python3
"""Filter positive generated/refined rubrics and build robust-weight inputs.

The input is the frozen collection produced for ``generate_refine.py``.  Every
candidate must already have a Luna evaluation on exactly those groups.  Initial
generation is retained when it beats its zero baseline. A modification must
strictly improve its recorded parent rubric.
No candidate is deleted here for portfolio size: zero weights and the six-rubric
limit are applied only by ``optimize_weights.py``.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from common import (
    atomic_json,
    load_json,
    normalized_pair,
    rubric_contract,
    rubric_key,
)


def direction(rubric: dict[str, Any]) -> int:
    value = str(rubric_contract(rubric)["direction"]).lower().strip()
    if value == "positive":
        return 1
    if value == "negative":
        return -1
    raise ValueError(f"invalid rubric direction: {value!r}")


def references(wrapper: dict[str, Any]) -> list[dict[str, Any]]:
    handbook = wrapper.get("handbook", wrapper)
    return handbook["rubric_and_weight_guidance"]["reference_golden_rubrics"]


def candidate_sources(specifications: list[str]) -> list[tuple[Path, Path]]:
    output = []
    for value in specifications:
        candidate, separator, evaluation = value.partition(":")
        if not separator:
            raise ValueError("--candidate-evaluation must be CANDIDATE_DIR:EVALUATION_DIR")
        output.append((Path(candidate), Path(evaluation)))
    return output


def load_candidates(
    instance_id: str,
    sources: list[tuple[Path, Path]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    accepted = []
    for candidate_dir, evaluation_dir in sources:
        candidate_path = candidate_dir / f"{instance_id}.json"
        evaluation_path = evaluation_dir / f"{instance_id}.json"
        if not candidate_path.is_file() and not evaluation_path.is_file():
            continue
        candidates = load_json(candidate_path)
        evaluation = load_json(evaluation_path)
        if candidates.get("status") != "success" or evaluation.get("status") != "success":
            raise ValueError(f"incomplete candidate stage: {instance_id}/{candidate_dir}")
        by_id = {item["candidate_id"]: item for item in candidates.get("candidates") or []}
        records = evaluation.get("records") or []
        if set(by_id) != {item["candidate_id"] for item in records}:
            raise ValueError(f"candidate/evaluation ID mismatch: {instance_id}/{candidate_dir}")
        for record in records:
            if record.get("strictly_improves_baseline") is True:
                accepted.append((by_id[record["candidate_id"]], record))
    return accepted


def score_cells(record: dict[str, Any]) -> dict[str, dict[str, int]]:
    output = {}
    for group in record["group_outcomes"]:
        group_key = str(group["group_key"])
        output[group_key] = {
            str(node_id): int(value["raw_score"])
            for node_id, value in group["raw_scores"].items()
        }
    return output


def fraction_payload(value: Any) -> dict[str, int]:
    tenths = Fraction(Decimal(str(value))) * 10
    return {"numerator": tenths.numerator, "denominator": tenths.denominator}


def build_instance(
    ledger: dict[str, Any],
    accepted: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    output_root: Path,
) -> dict[str, Any]:
    instance_id = str(ledger["instance_id"])
    groups = ledger["current_groups"]
    group_by_key = {str(group["group_key"]): group for group in groups}
    if len(group_by_key) != len(groups):
        raise ValueError(f"duplicate group key: {instance_id}")

    handbook_path = Path(ledger["current_handbook"]["path"])
    wrapper = load_json(handbook_path)
    rubric_by_contract: dict[str, dict[str, Any]] = {}
    scores_by_contract: dict[str, dict[str, dict[str, int]]] = {}
    for rubric in references(wrapper):
        contract = rubric_key(rubric)
        rubric_by_contract[contract] = copy.deepcopy(rubric)
        scores_by_contract[contract] = {
            key: {
                str(node_id): int(value["raw_score"])
                for node_id, value in group["rubric_scores"][contract]["raw_scores"].items()
            }
            for key, group in group_by_key.items()
        }

    for candidate, evaluation in accepted:
        rubric = candidate["rubric"]
        contract = rubric_key(rubric)
        cells = score_cells(evaluation)
        if set(cells) != set(group_by_key):
            raise ValueError(f"candidate group coverage mismatch: {instance_id}/{candidate['candidate_id']}")
        if contract in scores_by_contract and scores_by_contract[contract] != cells:
            raise ValueError(f"conflicting duplicate candidate scores: {instance_id}/{contract}")
        rubric_by_contract.setdefault(contract, copy.deepcopy(rubric))
        scores_by_contract.setdefault(contract, cells)

    candidate_ids = list(rubric_by_contract)
    pairs = []
    for group_key, group in group_by_key.items():
        node_ids = list(map(str, group["node_ids"]))
        rewards = {str(key): float(value) for key, value in group["gt_joint_rewards"].items()}
        for contract in candidate_ids:
            if set(scores_by_contract[contract][group_key]) != set(node_ids):
                raise ValueError(f"node coverage mismatch: {instance_id}/{group_key}/{contract}")
        for left, right in itertools.combinations(node_ids, 2):
            gt_sign = normalized_pair(rewards[left], rewards[right])
            if not gt_sign:
                continue
            pairs.append(
                {
                    "pair_id": f"{group_key}|{left}|{right}",
                    "gt_sign": gt_sign,
                    "left_raw_samples": [
                        [scores_by_contract[contract][group_key][left]]
                        for contract in candidate_ids
                    ],
                    "right_raw_samples": [
                        [scores_by_contract[contract][group_key][right]]
                        for contract in candidate_ids
                    ],
                    "group_id": group_key,
                    "left_node_id": left,
                    "right_node_id": right,
                }
            )

    rubrics = list(rubric_by_contract.values())
    problem = {
        "schema_version": "robust_scale_problem.v1",
        "status": "OPTIMIZE" if pairs else "NO_TRAINING_SIGNAL",
        "instance_id": instance_id,
        "candidate_ids": candidate_ids,
        "directions": [direction(rubric) for rubric in rubrics],
        "initial_weight_tenths_exact": [
            fraction_payload(rubric.get("weight", 1.0)) for rubric in rubrics
        ],
        "candidate_rubrics": rubrics,
        "semantic_clusters": [
            {"keep_contract": contract, "members": [contract]}
            for contract in candidate_ids
        ],
        "pairs": pairs,
        "counts": {
            "candidate_rubrics": len(rubrics),
            "groups": len(groups),
            "strict_gt_difference_pairs": len(pairs),
        },
        "accepted_generated_or_refined_rubrics": len(accepted),
    }
    base_wrapper = copy.deepcopy(wrapper)
    base_handbook = base_wrapper.get("handbook", base_wrapper)
    base_handbook["rubric_and_weight_guidance"][
        "reference_golden_rubrics"
    ] = rubrics
    atomic_json(output_root / "instances" / f"{instance_id}.json", problem)
    atomic_json(output_root / "base_handbooks" / f"{instance_id}.json", base_wrapper)
    return {
        "instance_id": instance_id,
        "candidate_rubrics": len(rubrics),
        "accepted_generated_or_refined_rubrics": len(accepted),
        "strict_gt_difference_pairs": len(pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--candidate-evaluation", action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise ValueError(f"output root already exists: {args.output_root}")
    sources = candidate_sources(args.candidate_evaluation)
    instance_ids = list(map(str, load_json(args.collection_root / "instance_manifest.json")["instance_ids"]))
    records = []
    for instance_id in instance_ids:
        ledger_path = args.collection_root / "current_ledgers" / f"{instance_id}.json"
        ledger = load_json(ledger_path)
        records.append(
            build_instance(
                ledger,
                load_candidates(instance_id, sources),
                output_root=args.output_root,
            )
        )
    manifest = {
        "schema_version": "robust_scale_problem_collection.v1",
        "status": "complete",
        "instance_ids": instance_ids,
        "records": records,
        "inputs": {
            "collection_root": str(args.collection_root),
            "candidate_evaluation_sources": args.candidate_evaluation,
        },
    }
    atomic_json(args.output_root / "problem_manifest.json", manifest)
    print(json.dumps({"status": "complete", "instances": len(instance_ids)}, sort_keys=True))


if __name__ == "__main__":
    main()
