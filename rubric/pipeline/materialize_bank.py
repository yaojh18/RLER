#!/usr/bin/env python3
"""Apply optimized rubrics to the given golden-reference bank."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from RLER.rubric.pipeline.common import atomic_json, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-bank", type=Path, required=True)
    parser.add_argument("--optimization-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank = copy.deepcopy(load_json(args.base_bank))
    updated = 0
    for instance_id in bank["eligible_instance_ids"]:
        handbook_path = args.optimization_root / "handbooks" / f"{instance_id}.json"
        wrapper = load_json(handbook_path)
        handbook = wrapper.get("handbook", wrapper)
        bank["instances"][instance_id]["reference_golden_rubrics"] = handbook[
            "rubric_and_weight_guidance"
        ]["reference_golden_rubrics"]
        updated += 1
    atomic_json(args.output, bank)
    print(
        json.dumps(
            {"status": "complete", "eligible_instances_updated": updated},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
