"""Naive GRPO baseline runner — M independent linear rollouts per instance.

No search. No Lane A / B / C. No docker commit. No rubric/judge. Each of the
M rollouts is a fresh agent session that starts from the base image, runs
linearly to termination or step_limit, gets GT-scored by the swebench
harness, and contributes one ExportSample to a single ExportGroup.

This module is intentionally parallel to (not coupled with)
``trajectory_search_parallel``: it imports a few stateless helpers from
``trajectory_search`` / ``parallel_utils`` (for terminal patch extraction,
GT eval, workspace meta, step-card serialization, token-info extraction)
but does NOT import the lanes runner. Editing the lanes runner does not
affect this file and vice-versa.

API mirrors TrajectorySearchParallelRunner.run():

    runner = NaiveSearchRunner(...)
    record = await runner.run()  # NaiveRecord

NaiveRecord has a single ``group`` field (one ExportGroup per instance,
M=8 samples) plus per-rollout bookkeeping.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import json
import logging
import subprocess
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_rl import RolloutSessionSpec
from swe_agent.backend import SWEAgentRolloutBackend

from swe_agent.parallel_utils import (
    TurnTokenInfo,
    _ensure_litellm_prefix,
    _stamp_steps,
    _strip_token_fields,
)
from swe_agent.prompt import EMPTY_WORKSPACE_META
from swe_agent.run.run_swe_agent import evaluate_swebench_instance_patches
from swe_agent.trajectory_search import _build_step_cards, _collect_workspace_meta


logger = logging.getLogger("swe_agent.naive_search")


# ---------------------------------------------------------------------------
# Config & data model
# ---------------------------------------------------------------------------


@dataclass
class NaiveSearchConfig:
    """Knobs for naive M-independent-rollout baseline."""

    m: int = 8  # number of independent rollouts per instance
    step_limit: int = 120  # hard cap per rollout
    seed: int | None = None

    policy_temperature: float = 1.0
    policy_top_p: float = 0.95

    gt_eval_workers: int = 8
    rollout_pool_size: int | None = None  # default = m

    # Same fallback-patch penalty recipe as v0/lanes.
    fallback_patch_penalty: float = 0.5
    # Stricter penalty applied when the rollout emitted ZERO environment
    # actions (degenerate <think></think><|im_end|> collapse seen in 56724).
    # 0.0 = hard zero reward, 1.0 = no penalty. Multiplied AFTER the regular
    # fallback penalty above.
    no_action_patch_penalty: float = 0.0

    # Reward formula. 'soft' = raw passed_set / (passed ∪ failed) on the
    # patched repo (always >= 0). 'delta' = max(0, raw - baseline), where
    # baseline is the soft score of an empty patch (only p2p_total contributes
    # to passed). 'delta' floors at 0 so a rollout that merely preserves
    # baseline pass-rate earns no credit; only NEW pass-rate gets signal.
    reward_kind: str = "delta"

    # ---- Format-error penalties (job 58062 mode-collapse fix) -------------
    # 58062 (v6b soft) collapsed: most rollouts emitted ```bash fences
    # instead of ```mswea_bash_command, so every assistant turn raised
    # FormatError ("Expected exactly 1 action, found 0") and the model
    # reinforced the broken pattern. Two compounding fixes below — both
    # default to OFF so existing runs are unaffected.
    #
    # Fix 1: linear per-turn decay. Reward *= max(0, 1 - n_fe * k). At
    # k=0.02 a rollout with 50 format-error turns has reward zeroed; with
    # 10 format-error turns it loses 20%. Mild push away from spamming
    # malformed turns without nuking single-mistake rollouts.
    format_error_per_step_penalty: float = 0.0
    # Fix 2: hard gate. If format-error / total-asst-turn rate exceeds
    # this threshold, reward is forced to 0 regardless of patch outcome.
    # Catches the pathological case (e.g. 109/120 = 91% in 58062) where
    # even an accidentally-correct fallback patch would otherwise leak
    # positive reward through to the policy. 0.0 = disabled.
    format_ok_gate_threshold: float = 0.0

    # ---- Stale-docker kill switch ----------------------------------------
    # When > 0, the runner steps the agent one turn at a time and, between
    # turns, compares the OLDEST observed SGLang weight_version in the
    # trajectory against the MOST RECENT observed one. The most-recent turn
    # was just served by SGLang's CURRENT weights, so
    # `latest - eldest` is a tight intra-trajectory bound on how stale the
    # earliest turn is relative to the live actor. If that spread exceeds
    # this threshold we tear down the docker env (env.cleanup) to abort
    # the rollout — the next env.execute() raises and the agent loop
    # exits with status="killed_stale_docker". 0 (default) disables the
    # check entirely and preserves the legacy single-call agent loop.
    kill_stale_docker_threshold: int = 0


@dataclass
class NaiveRollout:
    """One independent linear rollout of an instance.

    No fork point — each rollout starts from the base docker image with a
    fresh session and runs to termination or step_limit.
    """

    rollout_index: int  # 0..m-1
    node_id: str
    snapshot_after: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    step_cards: list[dict[str, Any]] = field(default_factory=list)
    workspace_meta: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(EMPTY_WORKSPACE_META)
    )
    turns: list[TurnTokenInfo] = field(default_factory=list)
    total_tokens: dict[str, int] = field(default_factory=dict)
    status: str = ""
    terminated_early: bool = False  # True iff agent invoked formal submit
    terminal_patch: str = ""
    terminal_patch_from_fallback: bool = False
    n_action_steps: int = 0  # count of step_cards with at least one command
    terminal_no_action_emitted: bool = False  # True iff zero env actions across rollout
    n_assistant_turns: int = 0  # total assistant turns in this rollout (denominator for FE rate)
    n_format_errors: int = 0  # assistant turns flagged with extra.format_error = True
    # NaiveSearchConfig.kill_stale_docker_threshold mid-rollout abort:
    # set True iff the runner tore down env.cleanup() because intra-trajectory
    # weight_version spread exceeded the threshold. killed_stale_lag records
    # (max_observed_wv - min_observed_wv) at the moment of the kill.
    killed_stale_docker: bool = False
    killed_stale_lag: int | None = None
    error: str | None = None
    gt_score: float | None = None
    gt_payload: dict[str, Any] | None = None
    started_at: float = 0.0
    finished_at: float = 0.0


@dataclass
class NaiveRecord:
    """Top-level record per instance — M independent rollouts grouped together."""

    instance_id: str
    run_dir: str
    task_id: str
    config: dict[str, Any]
    rollouts: list[NaiveRollout] = field(default_factory=list)
    completed: bool = False
    error: str | None = None
    seconds: float = 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class NaiveSearchRunner:
    """Run M independent fresh-session rollouts per instance and GT-score them.

    Sibling to TrajectorySearchParallelRunner: same constructor shape so the
    async collector can swap them. ``rubric_model_name`` / ``judge_model_name``
    / ``rubric_base_url`` are accepted for signature parity but ignored — naive
    baseline never calls rubric/judge.
    """

    def __init__(
        self,
        *,
        instance: dict[str, Any],
        backend: SWEAgentRolloutBackend,
        run_dir: Path,
        policy_model_name: str,
        config: NaiveSearchConfig,
        harness_namespace: str | None,
        policy_base_url: str,
        api_key: str = "EMPTY",
        # Per-trial URL list; trial ri uses policy_base_urls[ri % len(...)].
        # Falls back to policy_base_url for all trials if not provided.
        policy_base_urls: list[str] | None = None,
        # Accepted for call-site parity with the lanes runner; unused here.
        rubric_model_name: str | None = None,
        judge_model_name: str | None = None,
        rubric_base_url: str | None = None,
    ) -> None:
        self.instance = copy.deepcopy(instance)
        self.backend = backend
        self.run_dir = Path(run_dir)
        self.policy_model_name = policy_model_name
        self.config = config
        self.harness_namespace = harness_namespace
        self.policy_base_url = policy_base_url.rstrip("/")
        if policy_base_urls:
            self.policy_base_urls = [u.rstrip("/") for u in policy_base_urls]
        else:
            self.policy_base_urls = [self.policy_base_url]
        self.api_key = api_key

        self.task = instance["problem_statement"]
        self.task_id = instance["instance_id"]
        self.run_id = f"{self.task_id}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.base_image = str(self.backend.environment_config.get("image", ""))
        self.docker_executable = str(
            self.backend.environment_config.get("executable", "docker")
        )

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "rollouts").mkdir(parents=True, exist_ok=True)
        self._live_sessions: list[Any] = []

    # -- session plumbing ----------------------------------------------------

    def _make_session(self, rollout_index: int) -> Any:
        """Create a fresh agent session pinned to the policy sglang endpoint.

        Mirrors TrajectorySearchParallelRunner._make_initial_session but with
        a per-rollout session_id so M concurrent sessions don't collide.
        """
        spec = RolloutSessionSpec(
            session_id=f"naive-{rollout_index:02d}-{uuid.uuid4().hex}",
            task=self.task,
            task_id=self.task_id,
            sample_index=rollout_index,
            policy_ref=self.policy_model_name,
            policy_version=self.policy_model_name,
            dataset_name="swebench",
            ground_truth=self.instance.get("patch"),
            raw_user_query=self.task,
            limits={
                "step_limit": (
                    self.backend.agent_config.get("step_limit", 0)
                    if hasattr(self.backend, "agent_config")
                    else 0
                )
            },
            metadata={"template_vars": copy.deepcopy(self.instance)},
        )
        session = self.backend.create_session(spec)
        try:
            mk = session.agent.model.config.model_kwargs
            trial_url = self.policy_base_urls[
                rollout_index % len(self.policy_base_urls)
            ]
            mk["api_base"] = (
                trial_url + "/v1" if not trial_url.endswith("/v1") else trial_url
            )
            mk.setdefault("api_key", self.api_key)
            mk["temperature"] = float(self.config.policy_temperature)
            mk["top_p"] = float(self.config.policy_top_p)
            session.agent.model.config.model_name = _ensure_litellm_prefix(
                session.agent.model.config.model_name
            )
        except Exception:
            logger.warning(
                "[%s] could not pin api_base / model_name on naive session %d",
                self.task_id, rollout_index,
            )
        self._live_sessions.append(session)
        return session

    def _step_session(self, session: Any, max_steps: int) -> dict[str, Any]:
        return session.run_until_pause(max_steps=max_steps).model_dump(mode="json")

    # -- stale-docker watchdog ----------------------------------------------

    def _peek_latest_assistant_weight_version(self, session: Any) -> int | None:
        """Walk this session's live messages in reverse and return the
        most recent assistant turn's ``extra.weight_version`` (the SGLang
        weight version that served that turn). Returns None if no asst
        turn has been observed yet or the field is missing. Tolerates
        both Pydantic-model and dict message shapes so we don't depend on
        the agent_rl message class staying static."""
        try:
            state = getattr(getattr(session, "agent", None), "state", None)
            msgs = getattr(state, "messages", None) if state is not None else None
            if not msgs:
                return None
            for m in reversed(msgs):
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                if role != "assistant":
                    continue
                if isinstance(m, dict):
                    extra = m.get("extra") or {}
                    wv = extra.get("weight_version") if isinstance(extra, dict) else None
                else:
                    extra = getattr(m, "extra", None) or {}
                    if isinstance(extra, dict):
                        wv = extra.get("weight_version")
                    else:
                        wv = getattr(extra, "weight_version", None)
                if wv is None:
                    return None
                try:
                    return int(wv)
                except (TypeError, ValueError):
                    return None
            return None
        except Exception:
            return None

    def _abort_docker_env(self, session: Any) -> None:
        """Tear down the agent's docker env so the next env.execute() call
        raises and the agent loop exits cleanly. Used by the stale-docker
        watchdog. Tolerant of partial / already-cleaned state because it
        runs as a kill signal."""
        try:
            env = getattr(getattr(session, "agent", None), "env", None)
            if env is None:
                return
            if hasattr(env, "cleanup"):
                try:
                    env.cleanup()
                except Exception:
                    pass
        except Exception:
            pass

    def _step_session_with_stale_check(
        self, session: Any, rollout: "NaiveRollout"
    ) -> dict[str, Any]:
        """Step the agent one turn at a time; abort the rollout (via
        env.cleanup) if the intra-trajectory weight_version spread exceeds
        ``config.kill_stale_docker_threshold``. The most recent observed
        turn was just served by SGLang's current weights, so
        ``latest - eldest`` is a tight bound on (current_actor_wv -
        eldest_turn_wv) without needing a cross-process call into the
        actor.

        Returns the same dict shape as ``_step_session`` — the most recent
        ``run_until_pause`` result, or a copy with ``exit_status`` stamped
        to ``"killed_stale_docker"`` when the kill fires."""
        threshold = int(self.config.kill_stale_docker_threshold)
        eldest_wv: int | None = None
        result: dict[str, Any] = {}
        for step_idx in range(int(self.config.step_limit)):
            result = self._step_session(session, max_steps=1)
            # Agent reached a terminal state (Submitted / errored / etc.)
            # — exit_status is set. Hand the result back unmodified.
            if result.get("exit_status"):
                return result
            latest_wv = self._peek_latest_assistant_weight_version(session)
            if latest_wv is None:
                continue
            if eldest_wv is None or latest_wv < eldest_wv:
                eldest_wv = latest_wv
            lag = latest_wv - eldest_wv
            if lag > threshold:
                self._abort_docker_env(session)
                rollout.killed_stale_docker = True
                rollout.killed_stale_lag = lag
                result = dict(result)
                result["exit_status"] = "killed_stale_docker"
                result["killed_stale_lag"] = lag
                logger.info(
                    "[%s] naive r=%d killed_stale_docker after step %d "
                    "(latest_wv=%d eldest_wv=%d lag=%d > threshold=%d)",
                    self.task_id, rollout.rollout_index, step_idx + 1,
                    latest_wv, eldest_wv, lag, threshold,
                )
                return result
        return result

    def _extract_turn_token_info(
        self, snapshot_dict: dict[str, Any], starting_turn_index: int
    ) -> list[TurnTokenInfo]:
        # Copied verbatim from TrajectorySearchParallelRunner — same shape.
        turns_meta = snapshot_dict.get("metadata", {}).get("model_turns", [])
        result: list[TurnTokenInfo] = []
        for offset, turn_meta in enumerate(turns_meta[starting_turn_index:], start=0):
            response_msg = turn_meta.get("response_message") or {}
            response_meta = (
                response_msg.get("metadata") if isinstance(response_msg, dict) else None
            )
            response_meta = response_meta or {}
            usage = response_meta.get("usage") or turn_meta.get("usage") or {}
            output_token_ids = (
                response_meta.get("output_token_ids")
                or turn_meta.get("output_token_ids")
                or []
            )
            output_logprobs = (
                response_meta.get("output_logprobs")
                or turn_meta.get("output_logprobs")
                or []
            )
            role = (
                response_msg.get("role") if isinstance(response_msg, dict) else None
            ) or turn_meta.get("role", "assistant")
            result.append(
                TurnTokenInfo(
                    turn_index=starting_turn_index + offset,
                    role=role,
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    output_token_ids=list(output_token_ids),
                    output_logprobs=list(output_logprobs),
                )
            )
        return result

    def _extract_terminal_patch(
        self, result: dict[str, Any], session: Any
    ) -> tuple[str, bool]:
        def _normalize(p: str) -> str:
            p = p.rstrip()
            return p + "\n" if p else ""

        patch = _normalize(result.get("submission") or "")
        if patch:
            return patch, False
        try:
            diff = session.agent.env.execute(
                {"command": "git add -N . >/dev/null 2>&1; git diff"},
                timeout=30,
            )
            return _normalize(diff.get("output") or ""), True
        except Exception:
            return "", True

    def _cleanup_session(self, session: Any) -> None:
        if session is None:
            return
        try:
            session.close()
        except Exception:
            pass
        try:
            env = getattr(getattr(session, "agent", None), "env", None)
            if env is not None and hasattr(env, "cleanup"):
                env.cleanup()
        except Exception:
            pass
        try:
            if session in self._live_sessions:
                self._live_sessions.remove(session)
        except Exception:
            pass

    def _cleanup_all(self) -> None:
        for session in list(self._live_sessions):
            self._cleanup_session(session)
        self._live_sessions = []

    # -- one rollout end-to-end ---------------------------------------------

    def _run_one_rollout(self, rollout_index: int) -> NaiveRollout:
        """Run a single fresh-session rollout end-to-end (blocking, called
        from the thread-pool executor)."""
        node_id = f"naive-{self.task_id[:20]}-r{rollout_index:02d}-{uuid.uuid4().hex[:6]}"
        rollout = NaiveRollout(
            rollout_index=rollout_index,
            node_id=node_id,
            started_at=time.perf_counter(),
        )
        session = None
        try:
            session = self._make_session(rollout_index)
            # Snapshot the empty session so we know how many sys+user
            # messages exist before any assistant turn runs. Used as the
            # parent_message_count when slicing assistant turns.
            try:
                snap0 = session.snapshot().model_dump(mode="json")
                base_msg_count = len(
                    snap0.get("agent", {}).get("state", {}).get("messages", [])
                )
                base_turn_count = len(snap0.get("metadata", {}).get("model_turns", []))
                base_event_count = len(snap0.get("metadata", {}).get("events", []))
            except Exception:
                base_msg_count = 0
                base_turn_count = 0
                base_event_count = 0

            if int(self.config.kill_stale_docker_threshold) > 0:
                result = self._step_session_with_stale_check(session, rollout)
            else:
                result = self._step_session(session, max_steps=self.config.step_limit)
            snapshot_after = session.snapshot().model_dump(mode="json")
            if rollout.killed_stale_docker:
                # Docker env is gone — _collect_workspace_meta would try to
                # execute commands against it and raise. Use the empty
                # workspace meta sentinel; the agent's in-memory snapshot
                # (messages, step cards, turn token info) is still valid.
                workspace_meta = copy.deepcopy(EMPTY_WORKSPACE_META)
            else:
                workspace_meta = _collect_workspace_meta(session.agent.env)
            # The training sample wants the FULL chat (sys+user+all asst+tool)
            # so we keep messages from index 0 — the lane-to-grpo path will
            # treat the sys+user prefix as the masked-out parent.
            all_messages = copy.deepcopy(
                snapshot_after.get("agent", {}).get("state", {}).get("messages", [])
            )
            segment_events = copy.deepcopy(
                snapshot_after.get("metadata", {}).get("events", [])[base_event_count:]
            )
            rollout.snapshot_after = snapshot_after
            rollout.messages = all_messages
            rollout.step_cards = _build_step_cards(segment_events, 0)
            rollout.n_action_steps = sum(
                1 for c in rollout.step_cards if c.get("commands")
            )
            rollout.terminal_no_action_emitted = (rollout.n_action_steps == 0)
            # Format-error count is read off the live snapshot's assistant
            # messages — route_textbased_model.py stamps extra.format_error
            # on the assistant message before re-raising the FormatError.
            # These extras are preserved on rollout.messages here (the
            # stripping in _dump_record happens later, on the messages.json
            # copy only — messages_raw.json keeps them).
            rollout.n_assistant_turns = sum(
                1 for m in rollout.messages if m.get("role") == "assistant"
            )
            rollout.n_format_errors = sum(
                1 for m in rollout.messages
                if m.get("role") == "assistant"
                and bool((m.get("extra") or {}).get("format_error"))
            )
            rollout.workspace_meta = workspace_meta
            rollout.turns = self._extract_turn_token_info(snapshot_after, base_turn_count)
            rollout.total_tokens = {
                "prompt": sum(t.prompt_tokens for t in rollout.turns),
                "completion": sum(t.completion_tokens for t in rollout.turns),
            }
            rollout.status = result.get("status", "")
            rollout.terminated_early = result.get("exit_status") == "Submitted"
            if rollout.killed_stale_docker:
                # _extract_terminal_patch would shell out to the dead env
                # to fall back to `git diff`. Skip it: no patch survives a
                # stale-kill, so report empty + fallback-flagged.
                rollout.terminal_patch = ""
                rollout.terminal_patch_from_fallback = True
                rollout.status = "killed_stale_docker"
            else:
                rollout.terminal_patch, rollout.terminal_patch_from_fallback = (
                    self._extract_terminal_patch(result, session)
                )
            rollout._base_msg_count = base_msg_count  # type: ignore[attr-defined]
            logger.info(
                "[%s] naive r=%d done dt=%.1fs steps=%d submitted=%s status=%s "
                "patch_len=%d tokens_p=%d tokens_c=%d",
                self.task_id, rollout_index,
                time.perf_counter() - rollout.started_at,
                len(rollout.step_cards), rollout.terminated_early, rollout.status,
                len(rollout.terminal_patch),
                rollout.total_tokens.get("prompt", 0),
                rollout.total_tokens.get("completion", 0),
            )
        except Exception as exc:
            rollout.error = f"{type(exc).__name__}: {exc}"
            rollout.status = "error"
            logger.warning(
                "[%s] naive r=%d FAILED dt=%.1fs %s",
                self.task_id, rollout_index,
                time.perf_counter() - rollout.started_at, rollout.error,
            )
        finally:
            rollout.finished_at = time.perf_counter()
            self._cleanup_session(session)
        return rollout

    def _evaluate_gt(self, rollout: NaiveRollout) -> None:
        """Same recipe as TrajectorySearchParallelRunner._evaluate_gt: run the
        swebench harness on the rollout's terminal patch, apply fallback
        penalty when the patch came from `git diff` instead of formal submit.
        """
        t_gt = time.perf_counter()
        raw = (rollout.terminal_patch or "").rstrip()
        patch = (raw + "\n") if raw else ""
        if not patch:
            rollout.gt_score = 0.0
            rollout.gt_payload = {"reward": 0.0, "note": "empty_patch"}
            logger.info(
                "[%s] gt_done node=%s dt=0.0s reward=0.0 note=empty_patch",
                self.task_id, rollout.node_id,
            )
            return
        try:
            payload = evaluate_swebench_instance_patches(
                instance=self.instance,
                patches_by_key={rollout.node_id: patch},
                model_name=self.policy_model_name,
                max_workers=1,
                namespace=self.harness_namespace,
                work_dir=self.run_dir / "rollouts",
            )
            rollout_payload = (
                payload.get(rollout.node_id, {}) if isinstance(payload, dict) else {}
            )
            raw_reward = float(rollout_payload.get("reward", 0.0))
            # base_score = soft score on the unmodified repo: all p2p pass,
            # all f2p fail -> p2p_total / (p2p_total + f2p_total). Subtracting
            # it floors at 0, so a rollout that merely preserves baseline
            # earns no reward; only NEW pass rate gets signal.
            f2p_total = int(rollout_payload.get("f2p_total") or 0)
            p2p_total = int(rollout_payload.get("p2p_total") or 0)
            denom = f2p_total + p2p_total
            base_score = (p2p_total / denom) if denom > 0 else 0.0
            delta_reward = max(0.0, raw_reward - base_score)
            penalty = float(self.config.fallback_patch_penalty)
            no_action_penalty = float(self.config.no_action_patch_penalty)
            reward_kind = (self.config.reward_kind or "delta").lower()
            if reward_kind == "soft":
                scored_reward = raw_reward
            else:
                scored_reward = delta_reward
            notes: list[str] = []
            multipliers: dict[str, float] = {}
            if rollout.terminal_patch_from_fallback and penalty != 1.0:
                scored_reward = scored_reward * penalty
                multipliers["fallback_penalty"] = penalty
                notes.append("patch_from_git_diff_fallback")
            if rollout.terminal_no_action_emitted and no_action_penalty != 1.0:
                scored_reward = scored_reward * no_action_penalty
                multipliers["no_action_penalty"] = no_action_penalty
                notes.append("no_action_emitted")
            # Fix 1: linear per-turn format-error penalty. Multiplicative,
            # capped at 0 so we never flip sign. Default k=0 → no-op.
            fe_k = float(self.config.format_error_per_step_penalty)
            fe_count = int(rollout.n_format_errors)
            if fe_k > 0.0 and fe_count > 0:
                fe_multiplier = max(0.0, 1.0 - fe_count * fe_k)
                scored_reward = scored_reward * fe_multiplier
                multipliers["format_error_penalty"] = fe_multiplier
                notes.append(f"format_errors={fe_count}")
            # Fix 2: hard gate. If format-error rate over assistant turns
            # exceeds the threshold, zero reward outright. Catches the
            # 58062 pathology where the model emits ~90% malformed turns.
            fe_gate = float(self.config.format_ok_gate_threshold)
            n_asst = int(rollout.n_assistant_turns)
            fe_rate = (fe_count / n_asst) if n_asst > 0 else 0.0
            if fe_gate > 0.0 and fe_rate > fe_gate:
                scored_reward = 0.0
                multipliers["format_ok_gate"] = 0.0
                notes.append(f"format_ok_gate_tripped:rate={fe_rate:.2f}>{fe_gate:.2f}")
            rollout.gt_score = scored_reward
            rollout.gt_payload = {
                **rollout_payload,
                "raw_reward": raw_reward,
                "base_score": base_score,
                "delta_reward": delta_reward,
                "reward_kind": reward_kind,
                "n_format_errors": fe_count,
                "n_assistant_turns": n_asst,
                "format_error_rate": fe_rate,
                **multipliers,
                **({"note": "+".join(notes)} if notes else {}),
            }
            logger.info(
                "[%s] gt_done node=%s dt=%.1fs reward=%.3f",
                self.task_id, rollout.node_id, time.perf_counter() - t_gt,
                rollout.gt_score if rollout.gt_score is not None else -1.0,
            )
        except Exception as exc:
            rollout.gt_payload = {"error": f"{type(exc).__name__}: {exc}"}
            rollout.gt_score = None
            logger.warning(
                "[%s] gt_FAILED node=%s dt=%.1fs %s",
                self.task_id, rollout.node_id, time.perf_counter() - t_gt, exc,
            )

    # -- main loop -----------------------------------------------------------

    async def run(self) -> NaiveRecord:
        record = NaiveRecord(
            instance_id=self.task_id,
            run_dir=str(self.run_dir),
            task_id=self.task_id,
            config=asdict(self.config),
        )
        (self.run_dir / "config.json").write_text(
            json.dumps(asdict(self.config), indent=2)
        )
        started = time.perf_counter()
        cfg = self.config
        # Show distinct URLs in use (deduped) so per-trial routing is visible.
        distinct_urls = sorted(set(self.policy_base_urls))
        logger.info(
            "[%s] naive.run.start m=%d step_limit=%d policy_urls=%s",
            self.task_id, cfg.m, cfg.step_limit, distinct_urls,
        )

        loop = asyncio.get_running_loop()
        rollout_pool = ThreadPoolExecutor(
            max_workers=cfg.rollout_pool_size or cfg.m,
            thread_name_prefix=f"naive-{self.task_id[:12]}",
        )
        gt_pool = ThreadPoolExecutor(
            max_workers=cfg.gt_eval_workers,
            thread_name_prefix=f"naive-gt-{self.task_id[:12]}",
        )
        try:
            # Launch all M rollouts in parallel.
            rollout_futures: list[asyncio.Future] = []
            for ri in range(cfg.m):
                ctx = contextvars.copy_context()

                def _wrapped(idx=ri, c=ctx):
                    return c.run(self._run_one_rollout, idx)

                rollout_futures.append(loop.run_in_executor(rollout_pool, _wrapped))
            rollouts_or_excs = await asyncio.gather(
                *rollout_futures, return_exceptions=True
            )
            rollouts: list[NaiveRollout] = []
            for ri, item in enumerate(rollouts_or_excs):
                if isinstance(item, BaseException):
                    err_rollout = NaiveRollout(
                        rollout_index=ri,
                        node_id=f"naive-{self.task_id[:20]}-r{ri:02d}-err",
                        error=f"executor: {type(item).__name__}: {item}",
                        status="error",
                    )
                    rollouts.append(err_rollout)
                else:
                    rollouts.append(item)
            record.rollouts = rollouts

            # GT-score in parallel (CPU + docker, independent across rollouts).
            gt_futures: list[asyncio.Future] = []
            for r in rollouts:
                if r.error is not None:
                    r.gt_score = 0.0
                    r.gt_payload = {"reward": 0.0, "note": "rollout_error"}
                    continue
                if r.killed_stale_docker:
                    # No patch is recoverable from a stale-killed rollout
                    # — env was torn down mid-flight. Skip the harness
                    # spin-up and tag with a structured note so the
                    # collector can count it for the wandb metric.
                    r.gt_score = 0.0
                    r.gt_payload = {
                        "reward": 0.0,
                        "note": "killed_stale_docker",
                        "killed_stale_lag": r.killed_stale_lag,
                    }
                    continue
                ctx = contextvars.copy_context()
                gt_futures.append(
                    loop.run_in_executor(
                        gt_pool, lambda x=r, c=ctx: c.run(self._evaluate_gt, x)
                    )
                )
            if gt_futures:
                await asyncio.gather(*gt_futures, return_exceptions=True)

            record.completed = all(r.error is None for r in record.rollouts)
            if not record.completed and record.error is None:
                record.error = "one or more naive rollouts errored"
        except Exception as exc:
            record.error = f"runner exception: {exc}\n{traceback.format_exc()}"
            record.completed = False
        finally:
            record.seconds = time.perf_counter() - started
            self._dump_record(record)
            for pool in (rollout_pool, gt_pool):
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            self._cleanup_all()
        return record

    # -- persistence ---------------------------------------------------------

    def _dump_record(self, record: NaiveRecord) -> None:
        """Persist per-rollout artifacts + a top-level instance_record.json."""
        for r in record.rollouts:
            rdir = self.run_dir / "rollouts" / f"rollout_{r.rollout_index:02d}"
            rdir.mkdir(parents=True, exist_ok=True)
            stamped, _ = _stamp_steps(r.messages, start_step=0)
            (rdir / "messages.json").write_text(
                json.dumps(_strip_token_fields(stamped), indent=2, default=str)
            )
            (rdir / "messages_raw.json").write_text(
                json.dumps(stamped, indent=2, default=str)
            )
            (rdir / "terminal_patch.txt").write_text(r.terminal_patch or "")
            (rdir / "gt.json").write_text(
                json.dumps(
                    {"gt_score": r.gt_score, "gt_payload": r.gt_payload},
                    indent=2,
                    default=str,
                )
            )
            (rdir / "summary.json").write_text(
                json.dumps(
                    {
                        "node_id": r.node_id,
                        "status": r.status,
                        "terminated_early": r.terminated_early,
                        "terminal_patch_from_fallback": r.terminal_patch_from_fallback,
                        "terminal_no_action_emitted": r.terminal_no_action_emitted,
                        "n_action_steps": r.n_action_steps,
                        "n_assistant_turns": r.n_assistant_turns,
                        "n_format_errors": r.n_format_errors,
                        "killed_stale_docker": r.killed_stale_docker,
                        "killed_stale_lag": r.killed_stale_lag,
                        "terminal_patch_chars": len(r.terminal_patch or ""),
                        "error": r.error,
                        "n_messages": len(r.messages),
                        "n_step_cards": len(r.step_cards),
                        "total_tokens": r.total_tokens,
                        "started_at": r.started_at,
                        "finished_at": r.finished_at,
                    },
                    indent=2,
                    default=str,
                )
            )
        (self.run_dir / "instance_record.json").write_text(
            json.dumps(
                {
                    "instance_id": record.instance_id,
                    "run_dir": record.run_dir,
                    "config": record.config,
                    "completed": record.completed,
                    "error": record.error,
                    "seconds": record.seconds,
                    "n_rollouts": len(record.rollouts),
                    "rollouts": [
                        {
                            "rollout_index": r.rollout_index,
                            "node_id": r.node_id,
                            "status": r.status,
                            "gt_score": r.gt_score,
                            "n_step_cards": len(r.step_cards),
                            "terminated_early": r.terminated_early,
                        }
                        for r in record.rollouts
                    ],
                },
                indent=2,
                default=str,
            )
        )
