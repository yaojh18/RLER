"""Request-level model-token accounting for RLER experiments.

The ledger deliberately records metadata and token counts only. Prompts,
responses, API bases, credentials, and provider-specific request payloads must
never be written here.

Callers should configure one append-only ledger per experiment and attach
logical rollout context around model work::

    configure_usage_ledger("/path/to/usage.jsonl")
    with usage_context(phase="train", group_id=group_id):
        ...

Every physical model request is recorded by the request wrappers in
``agent_rl.run_utils``. After the group outcome is known, collectors can append
one disposition event so filtered/invalid token cost remains visible::

    record_group_disposition(group_id, disposition="filtered",
                             reason="zero_variance")

``UsageMetricsTracker`` consumes the ledger with event-id deduplication and
returns W&B-ready cumulative and delta metrics without exposing group ids.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import json
import math
import os
import re
import socket
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - RLER production hosts are Linux.
    fcntl = None


_SCHEMA_VERSION = 1
_LEDGER_ENV = "RLER_USAGE_LEDGER_PATH"
_RESUME_ENV = "RLER_USAGE_RESUME"
_ATTEMPT_NONCE_ENV = "RLER_USAGE_ATTEMPT_NONCE"
_DEFAULT_LEDGER_PATH: str | None = None
_WRITE_LOCK = threading.Lock()
_SAFE_LABEL = re.compile(r"[^A-Za-z0-9_.:/-]+")
_SAFE_NONCE = re.compile(r"[^A-Za-z0-9_.-]+")
_PROCESS_ATTEMPT_NONCE = uuid.uuid4().hex[:16]
_RESUME_HISTORY_END_OFFSET: int | None = None


@dataclasses.dataclass(frozen=True)
class UsageContext:
    ledger_path: str | None = None
    phase: str = "unspecified"
    group_id: str = ""
    model_family: str = ""
    model_role: str = ""
    logical_call_id: str = ""
    attempt_index: int = 0
    correction_index: int = 0
    suppress_accounting: bool = False


_USAGE_CONTEXT: contextvars.ContextVar[UsageContext] = contextvars.ContextVar(
    "rler_usage_context",
    default=UsageContext(),
)


def _safe_label(value: Any, *, default: str = "", max_length: int = 160) -> str:
    if value is None:
        return default
    normalized = _SAFE_LABEL.sub("_", str(value).strip())
    return (normalized or default)[:max_length]


_PROCESS_HOSTNAME = _safe_label(socket.gethostname(), max_length=255)
if not _PROCESS_HOSTNAME:  # pragma: no cover - production hosts have names.
    raise RuntimeError("socket.gethostname() returned no safe host identity")


def new_logical_call_id() -> str:
    return uuid.uuid4().hex


def usage_attempt_nonce() -> str:
    """Return a process-attempt namespace for collector group ids.

    A restarted rollout driver can replay the same rollout id and resets its
    task counter. Without a separate attempt namespace, a new disposition
    event could therefore reclassify the token cost of the previous process.
    Launchers may provide a human-readable nonce; otherwise Slurm restart
    identity is used when available and a process-start UUID is the safe
    fallback.
    """

    configured = str(os.environ.get(_ATTEMPT_NONCE_ENV) or "").strip()
    if configured:
        raw_nonce = configured
    else:
        slurm_job_id = str(os.environ.get("SLURM_JOB_ID") or "").strip()
        if slurm_job_id:
            restart_count = str(
                os.environ.get("SLURM_RESTART_COUNT")
                or os.environ.get("SLURM_RESTART_CNT")
                or "0"
            ).strip()
            raw_nonce = f"slurm-{slurm_job_id}-restart-{restart_count}"
        else:
            raw_nonce = f"process-{_PROCESS_ATTEMPT_NONCE}"
    return _SAFE_NONCE.sub("_", raw_nonce)[:96]


def build_usage_group_prefix(
    *,
    phase: str,
    rollout_id: int,
    instance_id: str,
    task_index: int,
    dataset_name: str = "",
) -> str:
    """Build a restart-safe logical group prefix for request accounting."""

    dataset = f"/{dataset_name}" if dataset_name else ""
    return (
        f"{phase}/attempt-{usage_attempt_nonce()}/r{rollout_id:04d}"
        f"{dataset}/{instance_id}/t{task_index:06d}"
    )


def current_usage_context() -> UsageContext:
    return _USAGE_CONTEXT.get()


@contextlib.contextmanager
def usage_context(**updates: Any) -> Iterator[UsageContext]:
    """Temporarily add request-accounting context.

    Unspecified fields inherit from the current context, so nested helpers can
    set a model role without discarding a collector's phase/group metadata.
    """

    current = current_usage_context()
    values = dataclasses.asdict(current)
    for key, value in updates.items():
        if key not in values:
            raise TypeError(f"unknown usage context field: {key}")
        if value is not None:
            values[key] = value
    updated = UsageContext(**values)
    token = _USAGE_CONTEXT.set(updated)
    try:
        yield updated
    finally:
        _USAGE_CONTEXT.reset(token)


def configure_usage_ledger(path: str | os.PathLike[str] | None) -> str | None:
    """Configure the process-local default ledger.

    Set ``RLER_USAGE_LEDGER_PATH`` before worker launch to give all workers the
    same default. Calling this function is still recommended in each Ray worker
    so the path is explicit and testable.
    """

    global _DEFAULT_LEDGER_PATH
    _DEFAULT_LEDGER_PATH = None if path is None else str(Path(path))
    if _DEFAULT_LEDGER_PATH is not None:
        ledger_path = Path(_DEFAULT_LEDGER_PATH)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.touch(exist_ok=True)
    return _DEFAULT_LEDGER_PATH


def get_usage_ledger_path() -> str | None:
    context = current_usage_context()
    if context.suppress_accounting:
        return None
    context_path = context.ledger_path
    if context_path:
        return context_path
    if _DEFAULT_LEDGER_PATH:
        return _DEFAULT_LEDGER_PATH
    return os.environ.get(_LEDGER_ENV) or None


def configure_usage_resume_window(
    historical_end_offset: int | None,
) -> None:
    """Select the durable ledger prefix for a checkpoint resume.

    Events appended after the model/data checkpoint are orphaned when that
    checkpoint is restored. A new tracker reads only the durable prefix and
    then advances to the ledger's current end so those orphaned events never
    enter cumulative or delta metrics.
    """

    global _RESUME_HISTORY_END_OFFSET
    if historical_end_offset is None:
        _RESUME_HISTORY_END_OFFSET = None
        return
    offset = int(historical_end_offset)
    if offset < 0:
        raise ValueError(
            "historical usage-ledger offset must be non-negative"
        )
    _RESUME_HISTORY_END_OFFSET = offset


def usage_ledger_offset(
    path: str | os.PathLike[str] | None = None,
) -> int | None:
    """Return a complete-line byte offset while excluding concurrent writes."""

    resolved = str(path or get_usage_ledger_path() or "").strip()
    if not resolved:
        return None
    ledger_path = Path(resolved)
    if not ledger_path.exists():
        return 0
    with ledger_path.open("rb") as stream:
        if fcntl is not None:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
        try:
            stream.seek(0, os.SEEK_END)
            return int(stream.tell())
        finally:
            if fcntl is not None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def infer_model_family(model_name: str | None) -> str:
    normalized = (model_name or "").lower()
    if "qwen" in normalized:
        return "qwen"
    if "glm" in normalized:
        return "glm"
    return "other"


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return None


def normalize_usage(
    usage: Any,
    *,
    known_input_tokens: int | None = None,
    known_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Normalize OpenAI/LiteLLM/SGLang usage without double-counting cache.

    Cached input is reported as a subset of input tokens and is therefore not
    added again to ``total_tokens``.
    """

    payload = _as_mapping(usage)
    input_tokens = _nonnegative_int(payload.get("prompt_tokens"))
    if input_tokens is None:
        input_tokens = _nonnegative_int(payload.get("input_tokens"))
    output_tokens = _nonnegative_int(payload.get("completion_tokens"))
    if output_tokens is None:
        output_tokens = _nonnegative_int(payload.get("output_tokens"))

    input_from_request = input_tokens is None and known_input_tokens is not None
    output_from_request = output_tokens is None and known_output_tokens is not None
    if input_from_request:
        input_tokens = _nonnegative_int(known_input_tokens)
    if output_from_request:
        output_tokens = _nonnegative_int(known_output_tokens)

    cached_tokens = _nonnegative_int(payload.get("cached_input_tokens"))
    if cached_tokens is None:
        cached_tokens = _nonnegative_int(payload.get("cached_tokens"))
    details = _as_mapping(payload.get("prompt_tokens_details"))
    if not details:
        details = _as_mapping(payload.get("input_tokens_details"))
    if cached_tokens is None:
        cached_tokens = _nonnegative_int(details.get("cached_tokens"))
    cached_reported = cached_tokens is not None

    input_known = input_tokens is not None
    output_known = output_tokens is not None
    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    cached_tokens = min(input_tokens, cached_tokens or 0)

    if input_from_request or output_from_request:
        usage_source = "token_ids"
        if payload:
            usage_source = "provider_and_token_ids"
    elif payload and input_known and output_known:
        usage_source = "provider"
    elif payload:
        usage_source = "provider_partial"
    else:
        usage_source = "missing"

    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_usage_known": input_known,
        "output_usage_known": output_known,
        "cached_usage_reported": cached_reported,
        "exact_usage": bool(input_known and output_known),
        "usage_source": usage_source,
    }


class UsageLedger:
    """Small append-only JSONL writer with process/thread serialization."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, event: Mapping[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("schema_version", _SCHEMA_VERSION)
        payload.setdefault("event_id", uuid.uuid4().hex)
        payload.setdefault("timestamp", time.time())
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        with _WRITE_LOCK:
            with self.path.open("a", encoding="utf-8") as stream:
                if fcntl is not None:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    stream.write(line)
                    stream.flush()
                finally:
                    if fcntl is not None:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _resolve_event_context(
    *,
    model_family: str | None = None,
    model_role: str | None = None,
    logical_call_id: str | None = None,
    attempt_index: int | None = None,
    correction_index: int | None = None,
) -> dict[str, Any]:
    context = current_usage_context()
    return {
        "phase": _safe_label(context.phase, default="unspecified"),
        "group_id": str(context.group_id or ""),
        # A broad rollout context supplies defaults for policy calls, while a
        # nested hosted rubric call supplies its actual provider family and
        # role explicitly. The physical call label must win in that case;
        # otherwise Luna traffic is silently accounted as Qwen policy usage.
        "model_family": _safe_label(model_family or context.model_family, default="other"),
        "model_role": _safe_label(model_role or context.model_role, default="unspecified"),
        "logical_call_id": str(context.logical_call_id or logical_call_id or new_logical_call_id()),
        "attempt_index": max(
            0,
            int(context.attempt_index if attempt_index is None else attempt_index),
        ),
        "correction_index": max(
            0,
            int(context.correction_index if correction_index is None else correction_index),
        ),
    }


def record_model_usage(
    *,
    usage: Any = None,
    known_input_tokens: int | None = None,
    known_output_tokens: int | None = None,
    status: str,
    model_family: str | None = None,
    model_role: str | None = None,
    logical_call_id: str | None = None,
    attempt_index: int | None = None,
    correction_index: int | None = None,
    latency_ms: float | None = None,
    request_started_at: float | None = None,
    error: BaseException | None = None,
    event_id: str | None = None,
) -> dict[str, Any] | None:
    """Append one physical request event, returning it for tests/debugging."""

    ledger_path = get_usage_ledger_path()
    if ledger_path is None:
        return None
    normalized = normalize_usage(
        usage,
        known_input_tokens=known_input_tokens,
        known_output_tokens=known_output_tokens,
    )
    context = _resolve_event_context(
        model_family=model_family,
        model_role=model_role,
        logical_call_id=logical_call_id,
        attempt_index=attempt_index,
        correction_index=correction_index,
    )
    normalized_request_started_at: float | None = None
    if request_started_at is not None:
        if isinstance(request_started_at, bool):
            raise ValueError(
                "request_started_at must be a finite non-negative number"
            )
        try:
            normalized_request_started_at = float(request_started_at)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "request_started_at must be a finite non-negative number"
            ) from exc
        if (
            not math.isfinite(normalized_request_started_at)
            or normalized_request_started_at < 0.0
        ):
            raise ValueError(
                "request_started_at must be a finite non-negative number"
            )
    event = {
        "schema_version": _SCHEMA_VERSION,
        "event_type": "request",
        "event_id": event_id or uuid.uuid4().hex,
        "timestamp": time.time(),
        "hostname": _PROCESS_HOSTNAME,
        "pid": os.getpid(),
        **context,
        **normalized,
        "status": _safe_label(status, default="unknown"),
        "latency_ms": max(0.0, float(latency_ms or 0.0)),
        # Never persist exception messages: provider errors can echo request
        # bodies, endpoints, or credentials.
        "error_type": type(error).__name__ if error is not None else "",
    }
    if normalized_request_started_at is not None:
        event["request_started_at"] = normalized_request_started_at
    UsageLedger(ledger_path).append(event)
    return event


def record_group_disposition(
    group_id: str,
    *,
    disposition: str,
    reason: str = "",
    phase: str | None = None,
    event_id: str | None = None,
) -> dict[str, Any] | None:
    """Append the final outcome for a group without rewriting request events."""

    ledger_path = get_usage_ledger_path()
    if ledger_path is None:
        return None
    context = current_usage_context()
    event = {
        "schema_version": _SCHEMA_VERSION,
        "event_type": "group_disposition",
        "event_id": event_id or uuid.uuid4().hex,
        "timestamp": time.time(),
        "hostname": _PROCESS_HOSTNAME,
        "pid": os.getpid(),
        "phase": _safe_label(phase or context.phase, default="unspecified"),
        "group_id": str(group_id),
        "disposition": _safe_label(disposition, default="unknown"),
        "reason": _safe_label(reason),
    }
    UsageLedger(ledger_path).append(event)
    return event


_CUMULATIVE_METRIC_KEYS = (
    "total_tokens_cumulative",
    "train_tokens_cumulative",
    "validation_tokens_cumulative",
    "qwen_tokens_cumulative",
    "qwen_input_tokens_cumulative",
    "qwen_cached_input_tokens_cumulative",
    "qwen_output_tokens_cumulative",
    "glm_tokens_cumulative",
    "glm_input_tokens_cumulative",
    "glm_cached_input_tokens_cumulative",
    "glm_output_tokens_cumulative",
    "dropped_tokens_cumulative",
    "zero_variance_dropped_tokens_cumulative",
    "requests_cumulative",
    "failed_requests_cumulative",
    "missing_usage_attempts_cumulative",
    "qwen_requests_cumulative",
    "qwen_failed_requests_cumulative",
    "qwen_missing_usage_attempts_cumulative",
    "glm_requests_cumulative",
    "glm_failed_requests_cumulative",
    "glm_missing_usage_attempts_cumulative",
)


class UsageMetricsTracker:
    """Incremental, event-id-deduplicated ledger consumer for W&B logging.

    ``peek``/``snapshot`` may be called by asynchronous telemetry (for
    example, heartbeats and validation) without consuming the token deltas
    that belong to the next optimizer update.  ``commit_update`` is the sole
    operation that advances that delta baseline.
    """

    _DROPPED_DISPOSITIONS = {
        "dropped",
        "filtered",
        "invalid",
        "rejected",
        # Collector code records these outcomes as ``dropped`` today. Keep
        # explicit aliases so historical or external writers cannot silently
        # omit unused rollout cost from the dropped-token metric.
        "excess",
        "partial_dropped",
    }

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        resume: bool | None = None,
    ):
        self.path = Path(path)
        self._offset = 0
        self._seen_event_ids: set[str] = set()
        self._totals: dict[str, float] = defaultdict(float)
        self._group_token_totals: dict[str, float] = defaultdict(float)
        self._group_dispositions: dict[str, tuple[str, str]] = {}
        self._previous_cumulative: dict[str, float] = defaultdict(float)
        if resume is None:
            resume = str(os.environ.get(_RESUME_ENV) or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if _RESUME_HISTORY_END_OFFSET is not None:
            current_end = (
                int(self.path.stat().st_size)
                if self.path.exists()
                else 0
            )
            self._read_new_events(
                end_offset=min(
                    _RESUME_HISTORY_END_OFFSET,
                    current_end,
                )
            )
            # Skip work performed after the durable checkpoint. New events
            # append after current_end and are consumed normally.
            self._offset = current_end
            self._set_delta_baseline(self._cumulative())
        elif resume:
            # Cumulative metrics must still contain all historical cost, but a
            # freshly created tracker in a requeued process must not publish
            # the complete ledger history as the first per-refresh delta.
            self._read_new_events()
            self._set_delta_baseline(self._cumulative())

    @classmethod
    def _dropped_flags(cls, disposition: str, reason: str) -> tuple[bool, bool]:
        normalized_disposition = disposition.strip().lower()
        normalized_reason = reason.strip().lower()
        dropped = normalized_disposition in cls._DROPPED_DISPOSITIONS
        return dropped, dropped and normalized_reason == "zero_variance"

    def _apply_request_event(self, event: Mapping[str, Any]) -> None:
        total = float(event.get("total_tokens") or 0)
        input_tokens = float(event.get("input_tokens") or 0)
        cached_tokens = float(event.get("cached_input_tokens") or 0)
        output_tokens = float(event.get("output_tokens") or 0)
        family = str(event.get("model_family") or "other")
        phase = str(event.get("phase") or "unspecified")
        status = str(event.get("status") or "")
        exact_usage = bool(event.get("exact_usage"))
        failed_request = status not in {"success", "validation_error"}

        self._totals["total_tokens_cumulative"] += total
        self._totals["requests_cumulative"] += 1
        self._totals[f"{family}_input_tokens_cumulative"] += input_tokens
        self._totals[f"{family}_cached_input_tokens_cumulative"] += cached_tokens
        self._totals[f"{family}_output_tokens_cumulative"] += output_tokens
        if phase == "train":
            self._totals["train_tokens_cumulative"] += total
        if phase in {"validation", "val", "eval"}:
            self._totals["validation_tokens_cumulative"] += total
        if family == "qwen":
            self._totals["qwen_tokens_cumulative"] += total
        if family == "glm":
            self._totals["glm_tokens_cumulative"] += total
        if family in {"qwen", "glm"}:
            self._totals[f"{family}_requests_cumulative"] += 1
            if not exact_usage:
                self._totals[
                    f"{family}_missing_usage_attempts_cumulative"
                ] += 1
            if failed_request:
                self._totals[f"{family}_failed_requests_cumulative"] += 1
        if not exact_usage:
            self._totals["missing_usage_attempts_cumulative"] += 1
        if failed_request:
            self._totals["failed_requests_cumulative"] += 1

        group_id = str(event.get("group_id") or "")
        if not group_id:
            return
        self._group_token_totals[group_id] += total
        disposition, reason = self._group_dispositions.get(group_id, ("", ""))
        dropped, zero_variance = self._dropped_flags(disposition, reason)
        if dropped:
            self._totals["dropped_tokens_cumulative"] += total
        if zero_variance:
            self._totals["zero_variance_dropped_tokens_cumulative"] += total

    def _apply_group_disposition(self, event: Mapping[str, Any]) -> None:
        group_id = str(event.get("group_id") or "")
        if not group_id:
            return
        new_state = (
            str(event.get("disposition") or ""),
            str(event.get("reason") or ""),
        )
        old_state = self._group_dispositions.get(group_id, ("", ""))
        if new_state == old_state:
            return

        group_tokens = self._group_token_totals.get(group_id, 0.0)
        old_dropped, old_zero_variance = self._dropped_flags(*old_state)
        new_dropped, new_zero_variance = self._dropped_flags(*new_state)
        self._totals["dropped_tokens_cumulative"] += group_tokens * (
            int(new_dropped) - int(old_dropped)
        )
        self._totals["zero_variance_dropped_tokens_cumulative"] += (
            group_tokens
            * (int(new_zero_variance) - int(old_zero_variance))
        )
        self._group_dispositions[group_id] = new_state

    def _read_new_events(
        self,
        *,
        end_offset: int | None = None,
    ) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as stream:
            stream.seek(self._offset)
            if end_offset is None:
                chunk = stream.read()
            else:
                chunk = stream.read(
                    max(0, int(end_offset) - self._offset)
                )
        if not chunk:
            return
        complete_length = chunk.rfind(b"\n") + 1
        if complete_length <= 0:
            return
        self._offset += complete_length
        for raw_line in chunk[:complete_length].splitlines():
            try:
                event = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            event_id = str(event.get("event_id") or "")
            if not event_id or event_id in self._seen_event_ids:
                continue
            self._seen_event_ids.add(event_id)
            event_type = event.get("event_type")
            if event_type == "request":
                self._apply_request_event(event)
            elif event_type == "group_disposition":
                self._apply_group_disposition(event)

    def _cumulative(self) -> dict[str, float]:
        cumulative = dict(self._totals)
        requests = cumulative.get("requests_cumulative", 0.0)
        missing = cumulative.get("missing_usage_attempts_cumulative", 0.0)
        cumulative["exact_usage_fraction"] = (requests - missing) / requests if requests else 1.0
        for family in ("qwen", "glm"):
            family_requests = cumulative.get(
                f"{family}_requests_cumulative",
                0.0,
            )
            family_missing = cumulative.get(
                f"{family}_missing_usage_attempts_cumulative",
                0.0,
            )
            family_failed = cumulative.get(
                f"{family}_failed_requests_cumulative",
                0.0,
            )
            cumulative[f"{family}_exact_usage_fraction"] = (
                (family_requests - family_missing) / family_requests
                if family_requests
                else 1.0
            )
            cumulative[f"{family}_failed_request_fraction"] = (
                family_failed / family_requests if family_requests else 0.0
            )
        cumulative["event_step"] = float(len(self._seen_event_ids))
        return cumulative

    def _set_delta_baseline(self, cumulative: Mapping[str, Any]) -> None:
        self._previous_cumulative = defaultdict(
            float,
            {
                key: float(value)
                for key, value in cumulative.items()
                if key.endswith("_cumulative")
            },
        )

    @staticmethod
    def _snapshot_metrics(
        cumulative: Mapping[str, Any],
    ) -> dict[str, float | int]:
        """Format cumulative and coverage metrics without update deltas."""

        output: dict[str, float | int] = {
            "usage/event_step": int(cumulative["event_step"]),
            "usage/exact_usage_fraction": float(cumulative["exact_usage_fraction"]),
        }
        for family in ("qwen", "glm"):
            output[f"usage/{family}_exact_usage_fraction"] = float(
                cumulative[f"{family}_exact_usage_fraction"]
            )
            output[f"usage/{family}_failed_request_fraction"] = float(
                cumulative[f"{family}_failed_request_fraction"]
            )
        for key in _CUMULATIVE_METRIC_KEYS:
            value = float(cumulative.get(key, 0.0))
            output[f"usage/{key}"] = int(value)
        return output

    def snapshot(self) -> dict[str, float | int]:
        """Return current cumulative/coverage metrics without consuming delta."""

        self._read_new_events()
        return self._snapshot_metrics(self._cumulative())

    def peek(self) -> dict[str, float | int]:
        """Alias for :meth:`snapshot` used by non-update telemetry."""

        return self.snapshot()

    def commit_update(self) -> dict[str, float | int]:
        """Return one update's deltas and advance the update delta baseline."""

        self._read_new_events()
        cumulative = self._cumulative()
        output = self._snapshot_metrics(cumulative)

        delta_sources = {
            "qwen_input_tokens_delta": "qwen_input_tokens_cumulative",
            "qwen_cached_input_tokens_delta": "qwen_cached_input_tokens_cumulative",
            "qwen_output_tokens_delta": "qwen_output_tokens_cumulative",
            "glm_input_tokens_delta": "glm_input_tokens_cumulative",
            "glm_cached_input_tokens_delta": "glm_cached_input_tokens_cumulative",
            "glm_output_tokens_delta": "glm_output_tokens_cumulative",
            "missing_usage_attempts_delta": "missing_usage_attempts_cumulative",
        }
        for output_key, source_key in delta_sources.items():
            current = float(cumulative.get(source_key, 0.0))
            previous = float(self._previous_cumulative.get(source_key, 0.0))
            output[f"usage/{output_key}"] = int(current - previous)

        self._set_delta_baseline(cumulative)
        return output

__all__ = [
    "UsageContext",
    "UsageLedger",
    "UsageMetricsTracker",
    "build_usage_group_prefix",
    "configure_usage_ledger",
    "current_usage_context",
    "get_usage_ledger_path",
    "infer_model_family",
    "new_logical_call_id",
    "normalize_usage",
    "record_group_disposition",
    "record_model_usage",
    "usage_attempt_nonce",
    "usage_context",
]
