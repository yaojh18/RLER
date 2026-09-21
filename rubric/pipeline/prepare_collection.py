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

from RLER.rubric.pipeline.common import atomic_json, load_json, normalized_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-manifest", type=Path, required=True)
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
        by_instance[instance_id].append(
            {
                "group_key": str(entry["group_key"]),
                "update_step": int(entry["update_step"]),
                "source": str(source),
                "node_ids": node_ids,
                "gt_joint_rewards": rewards,
                "strict_pairs": pairs,
            }
        )

    instance_ids = sorted(by_instance)
    for instance_id in instance_ids:
        groups = by_instance[instance_id]
        ledger = {
            "schema_version": "teacher_rubric_ledger.v1",
            "instance_id": instance_id,
            "current_groups": groups,
            "current_reference_statistics": [],
            "counts": {
                "groups": len(groups),
                "strict_gt_difference_pairs": sum(len(group["strict_pairs"]) for group in groups),
                "reference_rubrics": 0,
            },
        }
        output = args.output_root / "current_ledgers" / f"{instance_id}.json"
        atomic_json(output, ledger)
    atomic_json(
        args.output_root / "instance_manifest.json",
        {
            "schema_version": "teacher_rubric_instances.v1",
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
