#!/usr/bin/env python3
"""Materialize the frozen final folds and Direct-eligible train cohorts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "final_fold_config.json"


def _read_jsonl(path: Path) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append((line, payload))
    return rows


def _atomic_jsonl(path: Path, rows: list[tuple[str, dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for line, _ in rows:
                stream.write(line)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _source_rows(
    workspace: Path, config: dict[str, Any], fold: int
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    root = workspace / config["source_dataset"]
    fold_config = config["folds"][str(fold)]
    rows = {
        name: _read_jsonl(root / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    pool = rows["train"] + rows["validation"] + rows["test"]
    by_id = {
        str(row[1]["metadata"]["instance_id"]): row for row in pool
    }
    materialized = {}
    for name in ("train", "validation", "test"):
        instance_ids = list(map(str, fold_config[name]["instance_ids"]))
        materialized[name] = [by_id[instance_id] for instance_id in instance_ids]
    return materialized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = _source_rows(args.workspace_root, config, args.fold)

    bank_path = args.workspace_root / config["direct"]["bank"]
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    eligible = set(map(str, bank["eligible_instance_ids"]))
    direct_train = [
        row
        for row in rows["train"]
        if str(row[1]["metadata"]["instance_id"]) in eligible
    ]
    output = args.output_root / f"fold_{args.fold}"
    for variant, train in (("baseline", rows["train"]), ("direct", direct_train)):
        _atomic_jsonl(output / variant / "train.jsonl", train)
        _atomic_jsonl(output / variant / "validation.jsonl", rows["validation"])
        _atomic_jsonl(output / variant / "test.jsonl", rows["test"])
    print(
        json.dumps(
            {
                "status": "complete",
                "fold": args.fold,
                "baseline_train": len(rows["train"]),
                "direct_train": len(direct_train),
                "validation": len(rows["validation"]),
                "test": len(rows["test"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
