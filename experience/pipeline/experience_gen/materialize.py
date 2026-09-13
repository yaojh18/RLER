from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any, Iterable

from .filters import public_card_errors
from .io import atomic_json, load_json


SCOPES = ("siblings", "pc")


def update_counts(root: str | Path) -> dict[str, int]:
    checkpoint = Path(root)
    counts = {}
    for scope in SCOPES:
        path = checkpoint / "bank" / scope / "experience_bank.json"
        payload = load_json(path)
        counts[scope] = len(payload["experiences"])
        payload["experience_count"] = counts[scope]
        atomic_json(path, payload)
    path = checkpoint / "retrieval/keyword_bank.json"
    payload = load_json(path)
    payload["scope_counts"] = {
        scope: len(payload["bank"][scope]) for scope in SCOPES
    }
    payload["experience_count"] = sum(payload["scope_counts"].values())
    atomic_json(path, payload)
    return counts


def _accepted_card(result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    accepted = result.get("accepted_attempt")
    if not isinstance(accepted, dict) or not isinstance(accepted.get("card"), dict):
        raise ValueError("Result has no accepted experience card")
    scope = str(result.get("scope") or accepted.get("scope") or "")
    if scope not in SCOPES:
        raise ValueError("Accepted result has no valid scope")
    return scope, copy.deepcopy(accepted["card"])


def materialize_checkpoint(
    *,
    base: str | Path,
    output: str | Path,
    accepted_results: Iterable[str | Path] = (),
    keyword_records: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Apply accepted cards and extracted keyword candidates atomically."""

    source = Path(base)
    destination = Path(output)
    temporary = destination.with_name(destination.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    changes = []
    for result_path in accepted_results:
        scope, card = _accepted_card(load_json(result_path))
        errors = public_card_errors(card)
        if errors:
            raise ValueError(
                f"Rejected public card {card.get('experience_id')}: "
                + ", ".join(errors)
            )
        bank_path = temporary / "bank" / scope / "experience_bank.json"
        bank = load_json(bank_path)
        rows = {row["experience_id"]: row for row in bank["experiences"]}
        experience_id = card["experience_id"]
        action = "update" if experience_id in rows else "add"
        rows[experience_id] = card
        bank["experiences"] = sorted(
            rows.values(), key=lambda row: row["experience_id"]
        )
        atomic_json(bank_path, bank)
        changes.append(
            {"scope": scope, "experience_id": experience_id, "action": action}
        )
    if keyword_records:
        from .keywords import materialize_keyword_artifacts

        for scope, records in keyword_records.items():
            materialize_keyword_artifacts(
                checkpoint=temporary,
                scope=scope,
                records=records,
            )
    counts = update_counts(temporary)
    if destination.exists():
        backup = destination.with_name(destination.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        destination.replace(backup)
    temporary.replace(destination)
    return {
        "scope_counts": counts,
        "experience_count": sum(counts.values()),
        "changes": changes,
    }
