"""Naive GRPO baseline runner — M independent linear rollouts per instance.

No search. No Lane A / B / C. No docker commit. No rubric/judge. Each of the
M rollouts is a fresh agent session that starts from the base image, runs
linearly to termination or step_limit, gets GT-scored by the swebench
harness, and contributes one ExportSample to a single ExportGroup.

This module is intentionally parallel to (not coupled with)
``trajectory_search_parallel``: it imports a few stateless helpers from
``trajectory_search`` / ``parallel_utils`` (for terminal patch extraction,
GT eval, step-card serialization, token-info extraction)
but does NOT import the lanes runner. Editing the lanes runner does not
affect this file and vice-versa.

API mirrors TrajectorySearchParallelRunner.run():

    runner = NaiveSearchRunner(...)
    record = await runner.run()  # NaiveRecord

NaiveRecord has a single ``group`` field (one ExportGroup per instance,
M=8 samples) plus per-rollout bookkeeping.
"""

from __future__ import annotations

import contextvars
import copy
import json
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_rl import RolloutSessionSpec
from swe_agent.backend import SWEAgentRolloutBackend
from swe_agent.exceptions import PolicyVersionMismatch

from swe_agent.parallel_utils import (
    TurnTokenInfo,
    _ensure_litellm_prefix,
    exact_rollout_token_error,
    extract_terminal_patch_from_session,
    normalize_terminal_patch_text,
)
from swe_agent.run.run_swe_agent import (
    EvaluationRewardConfig,
    build_messages,
    evaluate_swebench_instance_patches,
    make_evaluation_payload,
)
from swe_agent.trajectory_search import _build_step_cards
from swe_agent.usage import usage_context


logger = logging.getLogger("swe_agent.naive_search")


# ---------------------------------------------------------------------------
# Config & data model
# ---------------------------------------------------------------------------


@dataclass
class NaiveSearchConfig:
    """Knobs for naive M-independent-rollout baseline."""

    m: int = 8  # number of independent rollouts per instance
    step_limit: int = 120  # hard cap per rollout

    policy_temperature: float = 1.0
    policy_top_p: float = 0.95

    gt_eval_workers: int = 8
    gt_eval_timeout: int = 600
    evaluate_gt: bool = True
    rollout_pool_size: int | None = None  # default = m
    rollout_max_attempts: int = 8

    # Same fallback-patch penalty recipe as v0/lanes.
    fallback_patch_penalty: float = 0.5
    # Additive adjustment for trajectories that execute no environment action.
    no_action_patch_penalty: float = -0.1

    reward_kind: str = "joint"
    joint_alpha: float = 1.0
    # Keep a fully solved rollout on the same unit scale as judge rewards.
    all_pass_reward: float = 1.0

@dataclass
class NaiveRollout:
    """One independent linear rollout of an instance.

    No fork point — each rollout starts from the base docker image with a
    fresh session and runs to termination or step_limit.
    """

    rollout_index: int  # 0..m-1
    node_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    step_cards: list[dict[str, Any]] = field(default_factory=list)
    turns: list[TurnTokenInfo] = field(default_factory=list)
    total_tokens: dict[str, int] = field(default_factory=dict)
    status: str = ""
    terminated_early: bool = False  # True iff agent invoked formal submit
    terminal_patch: str = ""
    terminal_patch_from_fallback: bool = False
    n_action_steps: int = 0  # count of step_cards with at least one command
    error: str | None = None
    gt_score: float | None = None
    evaluation_payload: dict[str, Any] | None = None
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
    """Run M independent fresh-session rollouts per instance and GT-score them."""

    def __init__(
        self,
        *,
        instance: dict[str, Any],
        backend: SWEAgentRolloutBackend,
        run_dir: Path,
        policy_model_name: str,
        policy_version: str | None = None,
        enforce_policy_version: bool = False,
        config: NaiveSearchConfig,
        harness_namespace: str | None,
        policy_base_url: str,
        api_key: str = "EMPTY",
        # Per-trial URL list; trial ri uses policy_base_urls[ri % len(...)].
        # Falls back to policy_base_url for all trials if not provided.
        policy_base_urls: list[str] | None = None,
    ) -> None:
        self.instance = copy.deepcopy(instance)
        self.backend = backend
        self.run_dir = Path(run_dir)
        self.policy_model_name = policy_model_name
        self.policy_version = policy_version or policy_model_name
        self.enforce_policy_version = bool(enforce_policy_version)
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

        self.run_timestamp = time.strftime("%Y%m%d-%H%M%S")

        self.run_dir.mkdir(parents=True, exist_ok=True)
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
            # Training uses slime's original stale-window semantics: the
            # dispatch version is retained in the exported record/sample, but
            # an in-flight trajectory is not aborted when the next update is
            # committed. Validation opts into the strict request-level guard
            # so unfinished policy work can be discarded on an update.
            policy_version=(
                self.policy_version if self.enforce_policy_version else None
            ),
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

    def _rollout_run_dir(self, rollout_index: int) -> Path:
        return (
            self.run_dir
            / self.task_id
            / f"{self.run_timestamp}-r{rollout_index:02d}"
        )

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
            # Snapshot the empty session so later token/event extraction only
            # reads data produced by this rollout.
            try:
                snap0 = session.snapshot().model_dump(mode="json")
                base_turn_count = len(snap0.get("metadata", {}).get("model_turns", []))
                base_event_count = len(snap0.get("metadata", {}).get("events", []))
            except Exception:
                base_turn_count = 0
                base_event_count = 0

            result = self._step_session(
                session, max_steps=self.config.step_limit
            )
            snapshot_after = session.snapshot().model_dump(mode="json")
            # The training sample wants the FULL chat (sys+user+all asst+tool)
            # so we keep messages from index 0 — the lane-to-grpo path will
            # treat the sys+user prefix as the masked-out parent.
            all_messages = copy.deepcopy(
                snapshot_after.get("agent", {}).get("state", {}).get("messages", [])
            )
            segment_events = copy.deepcopy(
                snapshot_after.get("metadata", {}).get("events", [])[base_event_count:]
            )
            rollout.messages = all_messages
            rollout.error = exact_rollout_token_error(all_messages)
            rollout.step_cards = _build_step_cards(segment_events, 0)
            rollout.n_action_steps = sum(
                1 for c in rollout.step_cards if c.get("commands")
            )
            rollout.turns = self._extract_turn_token_info(snapshot_after, base_turn_count)
            rollout.total_tokens = {
                "prompt": sum(t.prompt_tokens for t in rollout.turns),
                "completion": sum(t.completion_tokens for t in rollout.turns),
            }
            rollout.status = result.get("status", "")
            if result.get("exit_status") in {
                "CompletionLengthExceeded",
                "ContextWindowExceeded",
            }:
                rollout.status = "policy_overlength"
            if rollout.error is not None:
                rollout.status = "error"
            rollout.terminated_early = result.get("exit_status") == "Submitted"
            rollout.terminal_patch, rollout.terminal_patch_from_fallback = (
                extract_terminal_patch_from_session(result, session)
            )
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
        except PolicyVersionMismatch as exc:
            # The next policy update invalidates only unfinished validation
            # trajectories. Normal session cleanup remains unchanged from the
            # TTS path.
            rollout.error = f"{type(exc).__name__}: {exc}"
            rollout.status = "error"
            logger.warning(
                "[%s] naive r=%d STALE_POLICY dt=%.1fs %s",
                self.task_id,
                rollout_index,
                time.perf_counter() - rollout.started_at,
                rollout.error,
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
        """Evaluate one terminal patch with the shared reward definition."""
        t_gt = time.perf_counter()
        patch = normalize_terminal_patch_text(rollout.terminal_patch)
        reward_config = EvaluationRewardConfig(
            kind=self.config.reward_kind,
            joint_alpha=self.config.joint_alpha,
            all_pass_reward=self.config.all_pass_reward,
            fallback_patch_penalty=(
                self.config.fallback_patch_penalty
                if rollout.terminal_patch_from_fallback
                else 1.0
            ),
            no_action_patch_penalty=(
                self.config.no_action_patch_penalty
                if rollout.n_action_steps == 0
                else 0.0
            ),
        )
        if not patch:
            rollout.evaluation_payload = make_evaluation_payload(
                "empty", reward_config=reward_config
            )
            rollout.gt_score = float(rollout.evaluation_payload["reward"])
            logger.info(
                "[%s] gt_done node=%s dt=0.0s reward=%.3f note=empty_patch",
                self.task_id, rollout.node_id, rollout.gt_score,
            )
            return
        try:
            rollout_run_dir = self._rollout_run_dir(rollout.rollout_index)
            # Unlike the search runners, the naive runner does not write its
            # per-rollout artifacts until every GT evaluation has finished.
            # The shared Singularity evaluator needs its work directory to
            # exist before it can create a temporary evaluation directory.
            rollout_run_dir.mkdir(parents=True, exist_ok=True)
            eval_key = str(rollout_run_dir)
            payload = evaluate_swebench_instance_patches(
                instance=self.instance,
                patches_by_key={eval_key: patch},
                model_name=self.policy_model_name,
                max_workers=1,
                timeout=self.config.gt_eval_timeout,
                namespace=self.harness_namespace,
                work_dir=rollout_run_dir,
                reward_config=reward_config,
            )
            if not isinstance(payload, dict) or eval_key not in payload:
                raise RuntimeError(f"missing evaluation payload for {eval_key}")
            rollout_payload = payload[eval_key]
            rollout.evaluation_payload = copy.deepcopy(rollout_payload)
            if rollout_payload.get("metainfo", {}).get("infrastructure_error"):
                rollout.gt_score = None
            else:
                rollout.gt_score = float(rollout_payload["reward"])
            logger.info(
                "[%s] gt_done node=%s dt=%.1fs reward=%.3f",
                self.task_id, rollout.node_id, time.perf_counter() - t_gt,
                rollout.gt_score if rollout.gt_score is not None else -1.0,
            )
        except Exception as exc:
            rollout.evaluation_payload = make_evaluation_payload(
                "error", error=exc, reward_config=reward_config, infrastructure_error=True
            )
            rollout.gt_score = None
            logger.warning(
                "[%s] gt_FAILED node=%s dt=%.1fs %s",
                self.task_id, rollout.node_id, time.perf_counter() - t_gt, exc,
            )

    # -- main loop -----------------------------------------------------------

    async def run(self, on_rollout_done=None) -> NaiveRecord:
        record = NaiveRecord(
            instance_id=self.task_id,
            run_dir=str(self.run_dir),
            task_id=self.task_id,
            config=asdict(self.config),
        )
        started = time.perf_counter()
        cfg = self.config
        # Show distinct URLs in use (deduped) so per-trial routing is visible.
        distinct_urls = sorted(set(self.policy_base_urls))
        logger.info(
            "[%s] naive.run.start m=%d step_limit=%d policy_urls=%s",
            self.task_id, cfg.m, cfg.step_limit, distinct_urls,
        )

        rollout_pool = ThreadPoolExecutor(
            max_workers=cfg.rollout_pool_size or cfg.m,
            thread_name_prefix=f"naive-{self.task_id[:12]}",
        )
        gt_slots = threading.Semaphore(max(1, int(cfg.gt_eval_workers)))
        try:
            def _run_valid_rollout(rollout_index: int) -> NaiveRollout:
                last_rollout: NaiveRollout | None = None
                attempts = max(1, int(cfg.rollout_max_attempts))
                for attempt in range(1, attempts + 1):
                    with usage_context(
                        suppress_accounting=attempt > 1
                    ):
                        rollout = self._run_one_rollout(rollout_index)
                    last_rollout = rollout
                    if (
                        rollout.error is not None
                        and "PolicyVersionMismatch" in rollout.error
                    ):
                        return rollout
                    if rollout.error is None:
                        if on_rollout_done is not None:
                            try:
                                on_rollout_done(rollout_index)
                            except Exception as exc:
                                logger.warning(
                                    "[%s] on_rollout_done r=%d raised: %s",
                                    self.task_id,
                                    rollout_index,
                                    exc,
                                )
                        break
                    if attempt < attempts:
                        logger.warning(
                            "[%s] naive r=%d retrying policy attempt=%d/%d "
                            "rollout_error=%s",
                            self.task_id,
                            rollout_index,
                            attempt + 1,
                            attempts,
                            rollout.error or "",
                        )
                assert last_rollout is not None
                if last_rollout.error is not None:
                    last_rollout.error = (
                        f"rollout_retry_exhausted after {attempts} attempts: "
                        f"{last_rollout.error}"
                    )
                    last_rollout.status = "error"
                    last_rollout.gt_score = None
                    return last_rollout

                if not cfg.evaluate_gt:
                    return last_rollout

                # Policy capacity is released before terminal evaluation. An
                # evaluator infrastructure failure retries this same terminal
                # patch rather than wasting another policy rollout or holding
                # an inference slot idle.
                for eval_attempt in range(1, attempts + 1):
                    with gt_slots:
                        self._evaluate_gt(last_rollout)
                    if last_rollout.gt_score is not None:
                        return last_rollout
                    if eval_attempt < attempts:
                        logger.warning(
                            "[%s] naive r=%d retrying evaluator "
                            "attempt=%d/%d",
                            self.task_id,
                            rollout_index,
                            eval_attempt + 1,
                            attempts,
                        )
                last_rollout.error = (
                    "evaluator_retry_exhausted after "
                    f"{attempts} attempts"
                )
                last_rollout.status = "error"
                return last_rollout

            rollout_futures = []
            for rollout_index in range(cfg.m):
                rollout_context = contextvars.copy_context()

                def _wrapped(
                    idx=rollout_index,
                    context=rollout_context,
                ):
                    return context.run(_run_valid_rollout, idx)

                rollout_futures.append(
                    rollout_pool.submit(_wrapped)
                )
            rollouts_or_excs: list[NaiveRollout | Exception] = []
            for future in rollout_futures:
                try:
                    rollouts_or_excs.append(future.result())
                except Exception as exc:
                    rollouts_or_excs.append(exc)
            rollouts: list[NaiveRollout] = []
            for rollout_index, item in enumerate(rollouts_or_excs):
                if not isinstance(item, Exception):
                    rollouts.append(item)
                    continue
                err_rollout = NaiveRollout(
                    rollout_index=rollout_index,
                    node_id=(
                        f"naive-{self.task_id[:20]}-r{rollout_index:02d}-err"
                    ),
                    error=(
                        "rollout_retry_exhausted: executor: "
                        f"{type(item).__name__}: {item}"
                    ),
                    status="error",
                )
                err_rollout.evaluation_payload = make_evaluation_payload(
                    "error",
                    error=err_rollout.error,
                )
                rollouts.append(err_rollout)
            record.rollouts = rollouts

            record.completed = all(r.error is None for r in record.rollouts)
            if not record.completed and record.error is None:
                record.error = "one or more naive rollouts errored"
        except Exception as exc:
            record.error = f"runner exception: {exc}\n{traceback.format_exc()}"
            record.completed = False
        finally:
            record.seconds = time.perf_counter() - started
            self._dump_record(record)
            try:
                rollout_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._cleanup_all()
        return record

    # -- persistence ---------------------------------------------------------

    def _dump_record(self, record: NaiveRecord) -> None:
        """Persist each rollout in the same file shape as run_swe_agent."""
        for r in record.rollouts:
            rdir = self._rollout_run_dir(r.rollout_index)
            rdir.mkdir(parents=True, exist_ok=True)
            (rdir / "messages.json").write_text(
                json.dumps(
                    build_messages(
                        r.messages,
                        model_name=self.policy_model_name,
                        trajectory_format="mini-swe-agent-1.1",
                    ),
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            (rdir / "model_patch.json").write_text(
                json.dumps(
                    {
                        self.task_id: {
                            "model_name_or_path": self.policy_model_name,
                            "instance_id": self.task_id,
                            "model_patch": r.terminal_patch or "",
                        }
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            evaluation_payload = r.evaluation_payload
            if evaluation_payload is None:
                if not self.config.evaluate_gt:
                    continue
                if r.error is None:
                    raise RuntimeError(
                        f"missing evaluation payload for rollout {r.node_id}"
                    )
                # A policy/session failure can exhaust its bounded rollout
                # retries before terminal evaluation is reached.  Preserve
                # that original failure as an infrastructure-error artifact;
                # do not let persistence mask it with a second, unrelated
                # "missing evaluation payload" exception.
                evaluation_payload = make_evaluation_payload(
                    "error",
                    error=r.error,
                    infrastructure_error=True,
                )
            (rdir / "evaluation.json").write_text(
                json.dumps(
                    evaluation_payload,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
