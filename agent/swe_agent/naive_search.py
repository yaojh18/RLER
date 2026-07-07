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

import asyncio
import contextvars
import copy
import json
import logging
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
    extract_terminal_patch_from_session,
)
from swe_agent.run.run_swe_agent import (
    EvaluationRewardConfig,
    build_messages,
    evaluate_swebench_instance_patches,
    make_evaluation_payload,
)
from swe_agent.trajectory_search import _build_step_cards


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
    # Additive adjustment for trajectories that execute no environment action.
    no_action_patch_penalty: float = -0.1

    reward_kind: str = "joint"
    joint_alpha: float = 1.0
    all_pass_reward: float = 2.0


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
        self.run_timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.base_image = str(self.backend.environment_config.get("image", ""))
        self.docker_executable = str(
            self.backend.environment_config.get("executable", "docker")
        )

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

            result = self._step_session(session, max_steps=self.config.step_limit)
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
        raw = (rollout.terminal_patch or "").rstrip()
        patch = (raw + "\n") if raw else ""
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
            eval_key = str(rollout_run_dir)
            payload = evaluate_swebench_instance_patches(
                instance=self.instance,
                patches_by_key={eval_key: patch},
                model_name=self.policy_model_name,
                max_workers=1,
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

    async def run(self) -> NaiveRecord:
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
                    err_rollout.evaluation_payload = make_evaluation_payload(
                        "error", error=err_rollout.error,
                    )
                    rollouts.append(err_rollout)
                else:
                    rollouts.append(item)
            record.rollouts = rollouts

            # GT-score in parallel (CPU + docker, independent across rollouts).
            gt_futures: list[asyncio.Future] = []
            for r in rollouts:
                if r.error is not None:
                    r.gt_score = None
                    if r.evaluation_payload is None:
                        r.evaluation_payload = make_evaluation_payload(
                            "error", error=r.error,
                        )
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
                raise RuntimeError(f"missing evaluation payload for rollout {r.node_id}")
            (rdir / "evaluation.json").write_text(
                json.dumps(
                    evaluation_payload,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
