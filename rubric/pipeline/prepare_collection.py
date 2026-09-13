#!/usr/bin/env python3
"""Build frozen rubric ledgers directly from historical rollout-group artifacts.

The JSONL manifest contains ``instance_id``, ``group_key``, ``update_step``, and
``source``. Source artifacts are the normal calculate-GT rubric artifacts with
``rubrics``, ``score_records``, ``judge_scores``, and ``terminal_gt_scores``.
Only groups with a strict terminal-GT difference enter rubric optimization.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (
    atomic_json,
    load_json,
    normalized_pair,
    rubric_contract,
    rubric_key,
)


def criterion(rubric: dict[str, Any]) -> str:
    contract = rubric_contract(rubric)
    return json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def references(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    handbook = payload.get("handbook", payload)
    values = handbook["rubric_and_weight_guidance"].get("reference_golden_rubrics") or []
    if len(values) > 6 or len({rubric_key(value) for value in values}) != len(values):
        raise ValueError(f"invalid or duplicate base rubric set: {path}")
    return values


def source_scores(
    source: dict[str, Any],
    node_ids: list[str],
    refs: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    generated = source["rubrics"]
    generated_criteria = [criterion(rubric) for rubric in generated]
    score_rows = source["score_records"]
    if list(map(str, source["judge_scores"])) != node_ids or len(score_rows) != len(node_ids):
        raise ValueError("source node/score row order drift")
    output = {}
    for rubric in refs:
        matches = [
            index
            for index, value in enumerate(generated_criteria)
            if value == criterion(rubric)
        ]
        if len(matches) != 1:
            raise ValueError(
                "base rubric must have exactly one exact source score: "
                f"{rubric['title']}"
            )
        index = matches[0]
        by_node = {}
        for node_id, row in zip(node_ids, score_rows):
            record = row[index]
            raw = record.get("score_raw")
            if not isinstance(raw, int) or not 1 <= raw <= 5:
                raise ValueError("invalid source raw rubric score")
            by_node[node_id] = {
                "raw_score": raw,
                "evidence": str(record.get("evidence") or ""),
                "origin": "historical_exact_reuse",
            }
        output[rubric_key(rubric)] = by_node
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--base-handbook-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise ValueError(f"output root already exists: {args.output_root}")
    entries = [
        json.loads(line)
        for line in args.group_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        source = Path(entry["source"])
        if not source.is_absolute():
            source = args.group_manifest.parent / source
        payload = load_json(source)
        node_ids = list(map(str, payload["judge_scores"]))
        rewards = {str(key): float(value) for key, value in payload["terminal_gt_scores"].items()}
        if set(node_ids) != set(rewards):
            raise ValueError(f"GT node coverage drift: {source}")
        pairs = [
            {"left": left, "right": right, "gt_sign": sign}
            for left, right in itertools.combinations(node_ids, 2)
            if (sign := normalized_pair(rewards[left], rewards[right]))
        ]
        if not pairs:
            continue
        instance_id = str(entry["instance_id"])
        handbook_path = args.base_handbook_dir / f"{instance_id}.json"
        refs = references(handbook_path)
        cells = source_scores(payload, node_ids, refs)
        rubric_scores = {}
        for rubric in refs:
            contract = rubric_key(rubric)
            sign = -1 if rubric_contract(rubric)["direction"] == "negative" else 1
            correct = sum(
                normalized_pair(
                    sign * cells[contract][pair["left"]]["raw_score"],
                    sign * cells[contract][pair["right"]]["raw_score"],
                )
                == pair["gt_sign"]
                for pair in pairs
            )
            rubric_scores[contract] = {
                "rubric": rubric,
                "raw_scores": cells[contract],
                "strict_correct": correct,
                "strict_pairs": len(pairs),
            }
        by_instance[instance_id].append(
            {
                "group_key": str(entry["group_key"]),
                "update_step": int(entry["update_step"]),
                "source": str(source),
                "node_ids": node_ids,
                "gt_joint_rewards": rewards,
                "strict_pairs": pairs,
                "rubric_scores": rubric_scores,
            }
        )

    instance_ids = sorted(by_instance)
    for instance_id in instance_ids:
        groups = by_instance[instance_id]
        handbook_path = args.base_handbook_dir / f"{instance_id}.json"
        refs = references(handbook_path)
        statistics = []
        for rubric in refs:
            contract = rubric_key(rubric)
            correct = sum(group["rubric_scores"][contract]["strict_correct"] for group in groups)
            pairs = sum(group["rubric_scores"][contract]["strict_pairs"] for group in groups)
            statistics.append(
                {
                    "rubric_key": contract,
                    "rubric": rubric,
                    "strict_correct": correct,
                    "strict_pairs": pairs,
                    "pairwise_accuracy": correct / pairs,
                }
            )
        ledger = {
            "schema_version": "automatic_golden_rubric_ledger.v1",
            "instance_id": instance_id,
            "current_handbook": str(handbook_path),
            "current_groups": groups,
            "current_reference_statistics": statistics,
            "counts": {
                "groups": len(groups),
                "strict_gt_difference_pairs": sum(len(group["strict_pairs"]) for group in groups),
                "reference_rubrics": len(refs),
            },
        }
        output = args.output_root / "current_ledgers" / f"{instance_id}.json"
        atomic_json(output, ledger)
    atomic_json(
        args.output_root / "instance_manifest.json",
        {
            "schema_version": "automatic_golden_rubric_instances.v1",
            "instance_ids": instance_ids,
        },
    )
    print(
        json.dumps(
            {
                "instances": len(instance_ids),
                "groups": sum(len(value) for value in by_instance.values()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
