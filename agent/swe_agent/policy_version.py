"""Shared policy-weight epoch coordination for asynchronous SWE rollouts.

The training driver is the only writer.  Rollout subprocesses on every Ray
node read the same small JSON file before and after each SGLang request.  A
trajectory is therefore accepted only when every policy request ran entirely
under the policy version pinned when the trajectory was dispatched.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from swe_agent.exceptions import PolicyVersionMismatch


POLICY_VERSION_STATE_ENV = "RLER_POLICY_VERSION_STATE_PATH"


def policy_version_for_checkpoint(last_rollout_id: int) -> str:
    """Return the stable label for weights after ``last_rollout_id``."""
    if int(last_rollout_id) < 0:
        return "checkpoint-base"
    return f"checkpoint-{int(last_rollout_id):07d}"


def _checkpoint_id_for_policy_version(policy_version: str) -> int | None:
    version = str(policy_version or "").strip()
    if version == "checkpoint-base":
        return -1
    match = re.fullmatch(r"checkpoint-(\d{7})", version)
    return None if match is None else int(match.group(1))


def checkpoint_policy_stale_lag(
    policy_version: str,
    *,
    consumer_rollout_id: int,
) -> int:
    """Return true checkpoint lag for coordinated policy labels.

    Before optimizer update ``r``, checkpoint ``r - 1`` is current; therefore
    lag 0 and 1 are the accepted stale=1 window. Invalid labels fail closed so
    a resumed collector cannot silently bypass the coordinated version check.
    """

    return policy_version_stale_lag(
        policy_version,
        policy_version_for_checkpoint(int(consumer_rollout_id) - 1),
    )


def policy_version_stale_lag(
    policy_version: str,
    observed_policy_version: str,
) -> int:
    """Return checkpoint distance between a dispatched and observed policy."""

    checkpoint_id = _checkpoint_id_for_policy_version(policy_version)
    observed_id = _checkpoint_id_for_policy_version(observed_policy_version)
    if checkpoint_id is None or observed_id is None:
        raise PolicyVersionMismatch(
            "invalid coordinated policy versions: "
            f"dispatched={policy_version!r} observed={observed_policy_version!r}"
        )
    return observed_id - checkpoint_id


def policy_version_state_path() -> Path | None:
    raw = str(os.environ.get(POLICY_VERSION_STATE_ENV) or "").strip()
    return Path(raw) if raw else None


def _read_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PolicyVersionMismatch(
            f"policy version state is not initialized: {path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyVersionMismatch(
            f"policy version state is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PolicyVersionMismatch(
            f"policy version state must be a JSON object: {path}"
        )
    required = {
        "committed_version",
        "target_version",
        "updating",
        "transition_id",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise PolicyVersionMismatch(
            f"policy version state is missing {missing}: {path}"
        )
    return payload


def _atomic_write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def begin_policy_weight_update(target_version: str) -> str | None:
    """Mark the SGLang policy as transitioning before broadcasting weights."""
    path = policy_version_state_path()
    if path is None:
        return None
    transition_id = uuid.uuid4().hex
    committed_version: str | None = None
    sequence = 0
    try:
        previous = _read_state(path)
    except PolicyVersionMismatch:
        previous = None
    if previous is not None:
        committed_version = str(previous.get("committed_version") or "") or None
        sequence = int(previous.get("sequence", 0) or 0) + 1
    _atomic_write_state(
        path,
        {
            "committed_version": committed_version,
            "target_version": str(target_version),
            "updating": True,
            "transition_id": transition_id,
            "sequence": sequence,
            "updated_at": time.time(),
        },
    )
    return transition_id


def commit_policy_weight_update(
    target_version: str,
    transition_id: str | None,
) -> None:
    """Publish a completed weight broadcast as the new committed policy."""
    path = policy_version_state_path()
    if path is None:
        return
    state = _read_state(path)
    if (
        not bool(state.get("updating"))
        or str(state.get("target_version")) != str(target_version)
        or str(state.get("transition_id")) != str(transition_id)
    ):
        raise RuntimeError(
            "policy weight transition changed before commit: "
            f"expected target={target_version!r} transition={transition_id!r}, "
            f"observed={state}"
        )
    _atomic_write_state(
        path,
        {
            "committed_version": str(target_version),
            "target_version": str(target_version),
            "updating": False,
            "transition_id": str(transition_id),
            "sequence": int(state.get("sequence", 0) or 0),
            "updated_at": time.time(),
        },
    )


def fail_policy_weight_update(
    target_version: str,
    transition_id: str | None,
    error: BaseException,
) -> None:
    """Leave the state fail-closed when a weight broadcast fails."""
    path = policy_version_state_path()
    if path is None:
        return
    try:
        state = _read_state(path)
    except PolicyVersionMismatch:
        return
    if str(state.get("transition_id")) != str(transition_id):
        return
    state.update(
        {
            "updating": True,
            "target_version": str(target_version),
            "error": f"{type(error).__name__}: {error}",
            "updated_at": time.time(),
        }
    )
    _atomic_write_state(path, state)


def committed_policy_version() -> str | None:
    """Return a dispatchable policy version.

    ``None`` means coordination is disabled. During an update, dispatch fails
    closed; diagnostic validation records the last committed version instead.
    """
    path = policy_version_state_path()
    if path is None:
        return None
    state = _read_state(path)
    if bool(state.get("updating")):
        raise PolicyVersionMismatch(
            "policy weights are transitioning: "
            f"committed={state.get('committed_version')!r} "
            f"target={state.get('target_version')!r}"
        )
    version = str(state.get("committed_version") or "").strip()
    if not version:
        raise PolicyVersionMismatch(
            f"policy version state has no committed version: {state}"
        )
    return version


def observed_policy_version() -> str | None:
    """Return the currently committed policy, including during an update."""

    path = policy_version_state_path()
    if path is None:
        return None
    version = str(_read_state(path).get("committed_version") or "").strip()
    if not version:
        raise PolicyVersionMismatch(
            f"policy version state has no committed version: {path}"
        )
    return version


def assert_policy_version(expected_version: str | None, *, stage: str) -> None:
    """Fail when a policy request starts or ends under another weight epoch."""
    if expected_version is None or policy_version_state_path() is None:
        return
    state = _read_state(policy_version_state_path())
    observed = str(state.get("committed_version") or "")
    if bool(state.get("updating")) or observed != str(expected_version):
        raise PolicyVersionMismatch(
            "policy version changed during trajectory generation: "
            f"stage={stage} expected={expected_version!r} "
            f"observed={observed!r} updating={bool(state.get('updating'))} "
            f"target={state.get('target_version')!r}"
        )
