"""Small, schema-neutral helpers shared by the automatic rubric pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(
    path: Path,
    payload: Any,
    *,
    sort_keys: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=sort_keys,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def rubric_contract(rubric: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": rubric["title"],
        "direction": rubric.get("direction") or rubric.get("polarity"),
        "description": rubric["description"],
        "scale": rubric["scale"],
        "metadata": rubric.get("metadata") or {},
    }


def rubric_key(rubric: dict[str, Any]) -> str:
    """Canonical exact-rubric key used to join judge scores across stages."""

    encoded = json.dumps(
        rubric_contract(rubric), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def normalized_pair(a: float, b: float, *, epsilon: float = 1e-12) -> int:
    if a > b + epsilon:
        return 1
    if b > a + epsilon:
        return -1
    return 0
