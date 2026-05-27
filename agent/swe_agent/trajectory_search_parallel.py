"""Lane-based parallel trajectory search.

Flat Lane A / Lane B / Lane C scheme, three concurrent lanes per instance:

* Lane A (spine, NOT trained)
    A single agent runs the problem linearly to terminal/submit. Never blocks.
    Every `steps_per_round` assistant turns: `docker commit` the container,
    emit a MidCp = {idx, image_tag, snapshot, asst_step}. Purely a fork-point
    provider — its trajectory is NOT included in any training group.

* Lane B (M = 8 forks per mid_cp, sampling temperature = 1.0)
    For each mid_cp Lane A emits, fork M agents from that docker snapshot.
    Each Lane B branch runs independently to its own termination. The 8
    branches forked from mid_cp_i form fork-group i.

* Lane C (async rubric + judge per fork-group)
    Trigger: when all 8 Lane B branches in group i have completed at least
    `steps_per_round` assistant turns past their fork point (OR terminated
    earlier). Lane C dispatches summary/rubric_gen/judge ASYNCHRONOUSLY on
    the shared parent state + 8 × first-`steps_per_round` continuations.
    Lane C does NOT block Lane B from continuing.

GRPO bundle per fork-group: 8 trajectories sharing prompt = Lane A history
up to mid_cp_i, each with its own continuation to termination. Per-branch
reward = gt_score * fallback_penalty + rubric_judge_score; advantage = reward
- mean across the group (standard GRPO baseline).
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import json
import logging
import os
import random
import subprocess
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import pvariance
from typing import Any, Literal

import openai
from agent_rl import (
    RolloutSessionSpec,
    RolloutSnapshot,
    register_model_service,
)
from agent_rl.run_utils import ModelRouteConfig, configure_model_route
from swe_agent.backend import SWEAgentRolloutBackend

# Reuse v0's proven helpers verbatim — they are independent of the search
# topology (tree vs lanes). Keeping a single implementation also means the
# rubric/judge/PSU prompts stay byte-equal across v0 and v1, which matters
# for A/B comparisons.
from swe_agent.parallel_utils import (
    _build_rubric_prompt,
    _do_judge_one,
    _do_rubric_gen,
    _ensure_litellm_prefix,
    _make_async_client,
    _stamp_steps,
    _strip_litellm_prefix,
    _strip_token_fields,
    TurnTokenInfo,
)
from swe_agent.prompt import EMPTY_PERSISTENT_STATE, EMPTY_WORKSPACE_META
from swe_agent.run.run_swe_agent import evaluate_swebench_instance_patches
from swe_agent.rubric_bank import (
    ExperienceRubricBank,
    RubricRecord,
    ScoreRubricBank,
)
# Reuse the sequential helpers verbatim — they handle prompt construction,
# JSON-schema-validated rubric generation, judge scoring, and PSU. The
# parallel runner registers global routes for "rubric_generation" /
# "rubric_judge" in __init__ so these helpers (which call
# run_chat_with_route_completion_async with route names) end up hitting
# self.rubric_base_url. This keeps the rubric/judge prompts byte-equal
# with the sequential runner so trained rubric models transfer 1:1.
from swe_agent.trajectory_search import (
    JUDGE_ERROR_REWARD,
    _build_step_cards,
    _collect_workspace_meta,
    _docker_commit,
    _generate_round_rubrics,
    _redundancy_reward,
    _score_parent_round,
    _score_round,
    _update_persistent_state,
)


logger = logging.getLogger("swe_agent.trajectory_search_parallel")


# Module-level cache: per (base_url, model_name) tuple, the registered
# service_name. The full-parity rubric pipeline relies on global routes
# "rubric_generation" / "rubric_judge"; if N parallel runner instances
# all register their own service, the LAST one wins and earlier
# instances' in-flight calls get re-routed. They all point at the same
# DSv4 anyway, but make registration idempotent so the route stays
# stable across instance lifetimes.
_RUBRIC_ROUTE_LOCK = __import__("threading").Lock()
_RUBRIC_ROUTE_REGISTRY: dict[tuple[str, str], str] = {}


# ---------------------------------------------------------------------------
# Configuration & data model
# ---------------------------------------------------------------------------


@dataclass
class ParallelSearchConfig:
    """Same name as v0 config so call sites can swap with minimal diff,
    but with v1-specific fields (max_mid_cps replaces max_rounds; lane_b
    temperature is separate from policy temperature)."""

    m: int = 8  # forks per mid_cp (NOT M-1)
    max_mid_cps: int = 6  # cap on Lane A round emissions per instance
    steps_per_round: int = 20  # asst turns between mid_cp commits
    step_limit: int = 120  # hard cap per Lane B branch
    seed: int | None = None

    # Lane A sampling — runs deterministic-ish spine; lower temp is fine.
    policy_temperature: float = 1.0
    policy_top_p: float = 0.95
    # Lane B sampling — higher for intra-group diversity (we WANT 8 different
    # solution attempts from the same parent state).
    lane_b_temperature: float = 1.0
    lane_b_top_p: float = 0.95

    # Lane C (rubric/judge) reuses v0 defaults.
    rubric_temperature: float = 1.0
    rubric_top_p: float = 0.95
    rubric_max_tokens: int = 4096
    judge_temperature: float = 0.1
    judge_top_p: float = 0.95
    judge_max_tokens: int = 1024
    psu_max_tokens: int = 4096

    gt_eval_workers: int = 8
    lane_a_pool_size: int = 1  # Lane A is a single agent
    lane_b_pool_size: int | None = None  # default = m * max_mid_cps

    keep_images: bool = False
    return_logprobs: bool = True
    # Same recipe as v0: multiplicative penalty when the agent never invoked
    # the formal submit command and we fell back to `git diff` of the
    # working copy.
    fallback_patch_penalty: float = 0.5

    # ---- Full-parity rubric machinery (ported from sequential) ----
    # n=1 gives single-sample rubric_gen per ForkGroup (one shot). Set >1
    # for the sequential-style N-sample fan-out where the bank update
    # picks one of N drafts via _redundancy_reward + variance scoring.
    n: int = 1
    max_active_rubrics: int = 6
    rubric_bank_strategy: Literal["score", "experience", "both"] = "score"
    # When rubric_bank_strategy is experience/both, the on-disk experience
    # bank is persisted under run_dir/experience_bank.json. Set None to
    # use the in-memory seed and never persist (e.g. eval runs).
    experience_bank_path: str | None = None
    # If True, register global "rubric_generation" / "rubric_judge" routes
    # in __init__ so the sequential helpers can be reused. Set False to
    # bypass and fall back to the legacy one-shot _do_rubric_gen path.
    enable_full_parity_rubric: bool = True
    # Skip Lane C entirely (no rubric generation, no judge scoring). When
    # True, _run_fork_group only awaits GT eval; group.judge_response stays
    # empty and the bundler must use a GT-only reward path. Saves the rubric
    # token spend per ForkGroup and the bank-update serialization barrier.
    disable_rubric: bool = False


@dataclass
class MidCp:
    """Snapshot Lane A emits every `steps_per_round` assistant turns.

    Used as the fork point for `m` Lane B branches. The docker image_tag
    captures the container filesystem state; the snapshot dict captures the
    agent's full session state (messages, model_turns, events).
    """

    idx: int  # 0-indexed mid_cp position along Lane A
    asst_step: int  # cumulative Lane A asst step at emission
    image_tag: str  # docker image committed from Lane A's container
    snapshot: dict[str, Any] = field(default_factory=dict)
    # Token-level position markers — for verifying prefix invariance later.
    parent_message_count: int = 0  # len(snapshot.messages) at emission
    parent_event_count: int = 0
    parent_turn_count: int = 0
    emitted_at: float = 0.0  # perf_counter timestamp


@dataclass
class LaneBBranch:
    """A single Lane B fork. Runs to termination after being forked from a
    MidCp. Records its own messages, token info, terminal patch, GT score.

    Flattened vs v0's BranchState — no separate short/terminal phase since
    Lane B doesn't have round boundaries. We just record everything that
    happened between fork and termination.
    """

    group_index: int  # which fork-group (= which MidCp idx)
    branch_index: int  # 0..m-1 within the group
    node_id: str
    parent_image_tag: str
    snapshot_after: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    step_cards: list[dict[str, Any]] = field(default_factory=list)
    workspace_meta: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(EMPTY_WORKSPACE_META)
    )
    turns: list[TurnTokenInfo] = field(default_factory=list)
    total_tokens: dict[str, int] = field(default_factory=dict)
    status: str = ""  # session status string at exit
    terminated_early: bool = False  # True iff agent invoked formal submit
    terminal_patch: str = ""
    terminal_patch_from_fallback: bool = False
    error: str | None = None
    gt_score: float | None = None
    gt_payload: dict[str, Any] | None = None
    # Lane C barrier accounting. round1_event fires when this branch has
    # completed >= steps_per_round asst turns past mid_cp OR terminated
    # earlier. round1_messages = the truncated trajectory used as Lane C input.
    round1_done_at: float = 0.0
    round1_step_cards: list[dict[str, Any]] = field(default_factory=list)
    round1_workspace_meta: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0


@dataclass
class ForkGroup:
    """8 Lane B branches forked from the same MidCp + per-group Lane C output.

    Ready-for-bundle when: all `m` branches have terminated (or hit step_limit)
    AND Lane C has returned. Lane A's status is irrelevant for readiness.
    """

    group_index: int  # = mid_cp.idx
    mid_cp: MidCp
    branches: list[LaneBBranch] = field(default_factory=list)
    rubric_prompt: str = ""
    rubric_model_response: dict[str, Any] = field(default_factory=dict)
    judge_response: dict[str, Any] = field(default_factory=dict)
    persisted_parent_state: dict[str, Any] = field(default_factory=dict)
    lane_c_started_at: float = 0.0
    lane_c_done_at: float = 0.0
    bundled_at: float = 0.0
    # Full-parity rubric pipeline outputs (populated by _run_lane_c when
    # enable_full_parity_rubric=True). rubric_samples mirrors the
    # sequential runner's per-sample payloads (N=cfg.n entries); the bank
    # snapshot after this group's update is what the next group (along
    # the Lane A spine) loads as its starting state.
    rubric_samples: list[dict[str, Any]] = field(default_factory=list)
    selected_rubric_sample_index: int | None = None
    active_bank_after: list[dict[str, Any]] = field(default_factory=list)
    inactive_bank_after: list[dict[str, Any]] = field(default_factory=list)
    persistent_state_after: dict[str, Any] = field(default_factory=dict)


@dataclass
class LaneAState:
    """Lane A's overall record — for inspection / debug. NOT trained on."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    mid_cps: list[MidCp] = field(default_factory=list)
    terminal_patch: str = ""
    terminal_patch_from_fallback: bool = False
    terminated_early: bool = False
    status: str = ""
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    total_tokens: dict[str, int] = field(default_factory=dict)


@dataclass
class InstanceRecord:
    """Top-level record persisted to disk per instance. Conceptually replaces
    v0's InstanceRecord; the slime data-buffer hookup reads `groups`
    (and converts each ForkGroup to a GRPOExportBundle)."""

    instance_id: str
    run_dir: str
    task_id: str
    config: dict[str, Any]
    lane_a: LaneAState = field(default_factory=LaneAState)
    groups: list[ForkGroup] = field(default_factory=list)
    completed: bool = False
    error: str | None = None
    seconds: float = 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TrajectorySearchParallelRunner:
    """Lane-based parallel trajectory search per instance.

    Public API mirrors v0's ParallelDataSearchRunner:
        runner = TrajectorySearchParallelRunner(...)
        record = await runner.run(on_group_done=callback)

    `on_group_done(fork_group) -> None` fires once per ForkGroup as soon as
    it is ready-for-bundle (all m branches done + Lane C complete). The
    slime rollout fn converts the ForkGroup into a GRPOExportBundle inside
    the callback for streaming to training.
    """

    def __init__(
        self,
        *,
        instance: dict[str, Any],
        backend: SWEAgentRolloutBackend,
        run_dir: Path,
        policy_model_name: str,
        rubric_model_name: str | None,
        judge_model_name: str | None,
        config: ParallelSearchConfig,
        harness_namespace: str | None,
        policy_base_url: str,
        rubric_base_url: str,
        api_key: str = "EMPTY",
    ) -> None:
        self.instance = copy.deepcopy(instance)
        self.backend = backend
        self.run_dir = Path(run_dir)
        self.policy_model_name = policy_model_name
        self.rubric_model_name = rubric_model_name or policy_model_name
        self.judge_model_name = judge_model_name or self.rubric_model_name
        self.config = config
        self.harness_namespace = harness_namespace
        self.policy_base_url = policy_base_url.rstrip("/")
        self.rubric_base_url = rubric_base_url.rstrip("/")
        self.api_key = api_key

        self.task = instance["problem_statement"]
        self.task_id = instance["instance_id"]
        self.served_rubric_model_name = _strip_litellm_prefix(self.rubric_model_name)
        self.served_judge_model_name = _strip_litellm_prefix(self.judge_model_name)
        self.run_id = f"{self.task_id}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.base_image = str(self.backend.environment_config.get("image", ""))
        self.docker_executable = str(
            self.backend.environment_config.get("executable", "docker")
        )
        # Image-tag namespace per instance, same convention as v0.
        self.image_repository = f"rler-pds/{self.task_id.replace('__', '-').lower()}"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "groups").mkdir(parents=True, exist_ok=True)
        self._created_image_tags: list[str] = []
        self._live_sessions: list[Any] = []
        self._rng = random.Random(
            config.seed if config.seed is not None else hash(self.task_id) & 0xFFFFFFFF
        )

        # Captured by Lane A startup and reused by Lane C prompt construction.
        self.system_prompt = ""
        self.user_prompt = ""

        # ---- Full-parity rubric machinery (task 2) ----
        # Mirrors the sequential runner's wiring: an in-memory ScoreRubricBank
        # carries active/inactive rubrics across ForkGroups along the Lane A
        # spine; an optional ExperienceRubricBank takes a single
        # `update_after_instance` call at end-of-run with the accumulated
        # rubric payloads.
        uses_experience_bank = (
            self.config.enable_full_parity_rubric
            and self.config.rubric_bank_strategy in {"experience", "both"}
        )
        uses_score_bank = (
            self.config.enable_full_parity_rubric
            and self.config.rubric_bank_strategy in {"score", "both"}
        )
        if uses_score_bank:
            self.rubric_bank: ScoreRubricBank | None = ScoreRubricBank(
                max_active_rubrics=self.config.max_active_rubrics,
            )
        else:
            self.rubric_bank = None
        if uses_experience_bank:
            bank_path = (
                Path(self.config.experience_bank_path)
                if self.config.experience_bank_path
                else self.run_dir / "experience_bank.json"
            )
            self.experience_bank: ExperienceRubricBank | None = ExperienceRubricBank(
                bank_path=bank_path,
                write_artifacts=True,
            )
        else:
            self.experience_bank = None
        # Per-group bank snapshots — group i's _run_lane_c loads from
        # _group_bank_snapshots[i-1] and saves into [i]. The synthetic
        # "-1" entry is seeded by _initialize_rubric_bank() right after
        # Lane A captures system+user prompts (since the bank seed depends
        # on the task text).
        self._group_bank_snapshots: dict[int, dict[str, Any]] = {}
        # Ordered events ensure bank carry-forward serializes by
        # group_index even though Lane Cs run concurrently with each
        # other (and concurrently with later Lane B branches). Group i
        # awaits _group_done_events[i-1] before touching the bank.
        self._group_done_events: dict[int, asyncio.Event] = {}
        # Accumulated for ExperienceRubricBank.update_after_instance at
        # end of run; one entry per (group, rubric_sample).
        self._rubric_update_records: list[dict[str, Any]] = []
        self._rubric_bank_initialized = False

        # Register global model routes for the sequential helpers. The
        # parallel runner is normally launched inside its own subprocess
        # (collect_lanes_rollout_async uses ProcessPoolExecutor), so one
        # set of routes per process is exactly what we want — no risk of
        # cross-instance route collisions. We register both
        # rubric_generation and rubric_judge against the same
        # SGLangChatService at self.rubric_base_url.
        # Register global rubric routes IDEMPOTENTLY per (base_url, model).
        # Assumes this process serves a SINGLE endpoint — for multi-endpoint
        # scale-out, run K collector processes (1 per endpoint) so each has
        # clean global state. See docs/parallel_sft_pipeline_runbook.md for
        # the K-shard launcher.
        if self.config.enable_full_parity_rubric:
            try:
                from swe_agent.serving import SGLangChatService
                key = (self.rubric_base_url, self.served_rubric_model_name)
                with _RUBRIC_ROUTE_LOCK:
                    service_name = _RUBRIC_ROUTE_REGISTRY.get(key)
                    if service_name is None:
                        service_name = f"rubric-svc-{uuid.uuid4().hex[:8]}"
                        register_model_service(
                            service_name,
                            SGLangChatService(
                                base_url=self.rubric_base_url,
                                api_key=self.api_key,
                                default_model_name=self.served_rubric_model_name,
                            ),
                        )
                        configure_model_route(
                            "rubric_generation",
                            ModelRouteConfig(
                                backend="service",
                                service_name=service_name,
                                model_name=self.served_rubric_model_name,
                            ),
                        )
                        configure_model_route(
                            "rubric_judge",
                            ModelRouteConfig(
                                backend="service",
                                service_name=service_name,
                                model_name=self.served_judge_model_name,
                            ),
                        )
                        _RUBRIC_ROUTE_REGISTRY[key] = service_name
                        logger.info(
                            "[%s] registered rubric routes: service=%s base_url=%s model=%s",
                            self.task_id, service_name, self.rubric_base_url,
                            self.served_rubric_model_name,
                        )
            except Exception as exc:
                logger.warning(
                    "[%s] full-parity rubric route registration FAILED — "
                    "falling back to legacy one-shot Lane C: %s",
                    self.task_id, exc,
                )
                self.config = copy.copy(self.config)
                self.config.enable_full_parity_rubric = False
                self.rubric_bank = None
                self.experience_bank = None

    # -- session/docker plumbing ---------------------------------------------
    #
    # Copied verbatim from v0's ParallelDataSearchRunner. We don't import
    # them as bound methods because they reference v0-specific attributes;
    # but the logic is stable and proven so we keep the implementations
    # identical to avoid drift.

    def _make_initial_session(self) -> Any:
        spec = RolloutSessionSpec(
            session_id=f"lane-a-root-{uuid.uuid4().hex}",
            task=self.task,
            task_id=self.task_id,
            sample_index=0,
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
        # Pin model calls to the policy URL with the Lane A sampling params.
        try:
            mk = session.agent.model.config.model_kwargs
            mk["api_base"] = (
                self.policy_base_url + "/v1"
                if not self.policy_base_url.endswith("/v1")
                else self.policy_base_url
            )
            mk.setdefault("api_key", self.api_key)
            mk["temperature"] = float(self.config.policy_temperature)
            mk["top_p"] = float(self.config.policy_top_p)
            session.agent.model.config.model_name = _ensure_litellm_prefix(
                session.agent.model.config.model_name
            )
        except Exception:
            logger.warning("Could not pin api_base / model_name on Lane A session")
        self._live_sessions.append(session)
        return session

    def _step_session(self, session: Any, max_steps: int) -> dict[str, Any]:
        return session.run_until_pause(max_steps=max_steps).model_dump(mode="json")

    def _extract_turn_token_info(
        self, snapshot_dict: dict[str, Any], starting_turn_index: int
    ) -> list[TurnTokenInfo]:
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
        """(patch_text, is_fallback). Same recipe as v0."""

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

    def _would_fork_overflow(self, session: Any) -> bool:
        """Predict whether a Lane B fork from current Lane A state would
        fail with sglang context-length 400 on its first /generate call.

        Lane B's first call input = tokenize(messages, add_gen_prompt=True)
        (same pure-splice path the agent uses). Compare against the
        served sglang context length minus the per-call max_new_tokens
        budget; if it doesn't fit, skip emitting the mid_cp.

        Returns True if fork would overflow. Returns False on tokenizer
        errors (fail-open: don't block emission if we can't measure)."""
        try:
            from swe_agent.tokenization import tokenize_messages_with_template
            snap = session.snapshot().model_dump(mode="json")
            msgs = snap.get("agent", {}).get("state", {}).get("messages", [])
            if not msgs:
                return False
            # Lane B's first call includes ALL of Lane A's messages so far
            # (which Lane B sees as the conversation prefix) + the new
            # `<|im_start|>assistant\n<think>\n` generation prompt.
            input_ids = tokenize_messages_with_template(
                msgs, add_generation_prompt=True,
            )
            # sglang context length: read from session's model_kwargs.
            # max_new_tokens: how much output room Lane B will request.
            try:
                mk = session.agent.model.config.model_kwargs
            except Exception:
                mk = {}
            max_new_tokens = int(mk.get("max_tokens") or mk.get("max_completion_tokens") or 4096)
            # 80960 = current --sglang-context-length in grpo.sh
            sglang_ctx = int(os.environ.get("SWE_AGENT_LANES_SGLANG_CTX", "80960"))
            need = len(input_ids) + max_new_tokens
            return need > sglang_ctx
        except Exception as exc:
            logger.warning(
                "[%s] _would_fork_overflow check FAILED, defaulting to no-skip: %s",
                self.task_id, exc,
            )
            return False

    def _commit_container(self, session: Any, tag_kind: str) -> str | None:
        """`docker commit` the session's container to a fresh image tag.
        Returns the new image tag, or None if the container is gone."""
        container_id = getattr(session.agent.env, "container_id", None) or getattr(
            session.agent.env, "_container_id", None
        )
        if not container_id:
            return None
        tag = self._new_image_tag(tag_kind)
        try:
            _docker_commit(self.docker_executable, container_id, tag)
            return tag
        except Exception as exc:
            logger.warning(
                "[%s] docker commit failed for tag=%s: %s", self.task_id, tag, exc
            )
            return None

    def _new_image_tag(self, tag_kind: str) -> str:
        suffix = uuid.uuid4().hex[:6]
        tag = f"{self.image_repository}:{tag_kind}-{suffix}"
        self._created_image_tags.append(tag)
        return tag

    def _delete_image(self, image_tag: str | None) -> None:
        if not image_tag or self.config.keep_images:
            return
        if image_tag == self.base_image:
            return
        try:
            subprocess.run(
                [self.docker_executable, "image", "rm", "-f", image_tag],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception as exc:
            logger.warning("Failed to delete image %s: %s", image_tag, exc)

    def _cleanup_all(self) -> None:
        for session in self._live_sessions:
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
        self._live_sessions = []
        for tag in list(self._created_image_tags):
            self._delete_image(tag)
        self._created_image_tags = []

    # -- Lane B: per-mid_cp forks --------------------------------------------

    def _override_model_kwargs(
        self,
        snapshot_dict: dict[str, Any],
        base_url: str,
        temperature: float,
        top_p: float,
    ) -> None:
        """Pin a resumed snapshot's model calls to a specific sglang base_url
        and override sampling params (typically temp=1.0 for Lane B)."""
        model_section = snapshot_dict.setdefault("model", {})
        config_section = model_section.setdefault("config", {})
        kwargs = config_section.setdefault("model_kwargs", {})
        kwargs["api_base"] = (
            base_url + "/v1" if not base_url.endswith("/v1") else base_url
        )
        kwargs.setdefault("api_key", self.api_key)
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = float(top_p)
        if "model_name" in config_section:
            config_section["model_name"] = _ensure_litellm_prefix(
                config_section["model_name"]
            )

    def _fork_lane_b(self, *, mid_cp: MidCp, branch_index: int) -> Any:
        """Fork a single Lane B branch from a MidCp. Returns the resumed
        session pinned to a fresh docker container (instantiated from
        mid_cp.image_tag) with Lane B sampling temperature.

        Mirrors v0's _fork_branch container-isolation contract:
        reuse_container_id=None + container_id=None + owns_container=False
        force DockerEnvironment.__init__ down the _start_container path
        so each Lane B gets its own container. Without this, all m=8
        siblings would share Lane A's container and the first cleanup()
        would kill the rest."""
        node_id = (
            f"lane-b-g{mid_cp.idx:03d}-b{branch_index:02d}-{uuid.uuid4().hex[:6]}"
        )
        resumed = copy.deepcopy(mid_cp.snapshot)
        resumed["session_id"] = f"{node_id}-session"
        resumed["spec"]["session_id"] = resumed["session_id"]
        for index, event in enumerate(resumed.get("metadata", {}).get("events", [])):
            event["session_id"] = resumed["session_id"]
            event["event_id"] = f"{resumed['session_id']}:{index}"
        for turn in resumed.get("metadata", {}).get("model_turns", []):
            turn["session_id"] = resumed["session_id"]
        self._override_model_kwargs(
            resumed,
            self.policy_base_url,
            temperature=self.config.lane_b_temperature,
            top_p=self.config.lane_b_top_p,
        )
        env_section = resumed.setdefault("environment", {})
        env_config = env_section.setdefault("config", {})
        env_state = env_section.setdefault("state", {})
        env_config["image"] = mid_cp.image_tag
        env_config["reuse_container_id"] = None
        env_state["container_id"] = None
        env_state["owns_container"] = False
        session = self.backend.resume_session(RolloutSnapshot(**resumed))
        self._live_sessions.append(session)
        return session

    def _run_lane_b_branch(
        self,
        *,
        mid_cp: MidCp,
        branch_index: int,
        budget_steps: int,
    ) -> LaneBBranch:
        """Run a single Lane B branch end-to-end (blocking, called from
        the executor): fork, run up to budget_steps, capture terminal
        patch. Returns the populated LaneBBranch.

        GT eval is dispatched separately by `_run_fork_group` so that the
        gt_pool can be shared across all groups."""
        node_id = (
            f"lane-b-g{mid_cp.idx:03d}-b{branch_index:02d}-{uuid.uuid4().hex[:6]}"
        )
        branch = LaneBBranch(
            group_index=mid_cp.idx,
            branch_index=branch_index,
            node_id=node_id,
            parent_image_tag=mid_cp.image_tag,
            started_at=time.perf_counter(),
        )
        session = None
        try:
            session = self._fork_lane_b(mid_cp=mid_cp, branch_index=branch_index)
            before_event_count = mid_cp.parent_event_count
            before_message_count = mid_cp.parent_message_count
            before_turn_count = mid_cp.parent_turn_count
            result = self._step_session(session, max_steps=budget_steps)
            snapshot_after = session.snapshot().model_dump(mode="json")
            workspace_meta = _collect_workspace_meta(session.agent.env)
            segment_events = copy.deepcopy(
                snapshot_after.get("metadata", {})
                .get("events", [])[before_event_count:]
            )
            messages = copy.deepcopy(
                snapshot_after.get("agent", {}).get("state", {})
                .get("messages", [])[before_message_count:]
            )
            branch.snapshot_after = snapshot_after
            branch.messages = messages
            branch.step_cards = _build_step_cards(segment_events, 0)
            branch.workspace_meta = workspace_meta
            branch.turns = self._extract_turn_token_info(snapshot_after, before_turn_count)
            branch.total_tokens = {
                "prompt": sum(t.prompt_tokens for t in branch.turns),
                "completion": sum(t.completion_tokens for t in branch.turns),
            }
            branch.status = result.get("status", "")
            branch.terminated_early = result.get("exit_status") == "Submitted"
            branch.terminal_patch, branch.terminal_patch_from_fallback = (
                self._extract_terminal_patch(result, session)
            )
            spr = self.config.steps_per_round
            branch.round1_step_cards = list(branch.step_cards[:spr])
            branch.round1_workspace_meta = workspace_meta
            branch.round1_done_at = time.perf_counter()
            logger.info(
                "[%s] lane_b g=%d b=%d done dt=%.1fs steps=%d submitted=%s "
                "status=%s patch_len=%d tokens_p=%d tokens_c=%d",
                self.task_id, mid_cp.idx, branch_index,
                time.perf_counter() - branch.started_at,
                len(branch.step_cards), branch.terminated_early, branch.status,
                len(branch.terminal_patch),
                branch.total_tokens.get("prompt", 0),
                branch.total_tokens.get("completion", 0),
            )
        except Exception as exc:
            branch.error = f"{type(exc).__name__}: {exc}"
            branch.status = "error"
            logger.warning(
                "[%s] lane_b g=%d b=%d FAILED dt=%.1fs %s",
                self.task_id, mid_cp.idx, branch_index,
                time.perf_counter() - branch.started_at, branch.error,
            )
        finally:
            branch.finished_at = time.perf_counter()
            if session is not None:
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
        return branch

    def _evaluate_gt(self, branch: LaneBBranch) -> None:
        """Score a Lane B branch's terminal patch via swebench harness.
        Same recipe as v0's _evaluate_gt (with fallback-patch penalty)."""
        t_gt = time.perf_counter()
        raw = (branch.terminal_patch or "").rstrip()
        patch = (raw + "\n") if raw else ""
        if not patch:
            branch.gt_score = 0.0
            branch.gt_payload = {"reward": 0.0, "note": "empty_patch"}
            logger.info(
                "[%s] gt_done node=%s dt=0.0s reward=0.0 note=empty_patch",
                self.task_id, branch.node_id,
            )
            return
        try:
            payload = evaluate_swebench_instance_patches(
                instance=self.instance,
                patches_by_key={branch.node_id: patch},
                model_name=self.policy_model_name,
                max_workers=1,
                namespace=self.harness_namespace,
                work_dir=self.run_dir / "groups",
            )
            branch_payload = (
                payload.get(branch.node_id, {}) if isinstance(payload, dict) else {}
            )
            raw_reward = float(branch_payload.get("reward", 0.0))
            penalty = float(self.config.fallback_patch_penalty)
            if branch.terminal_patch_from_fallback and penalty != 1.0:
                branch.gt_score = raw_reward * penalty
                branch.gt_payload = {
                    **branch_payload,
                    "raw_reward": raw_reward,
                    "fallback_penalty": penalty,
                    "note": "patch_from_git_diff_fallback",
                }
            else:
                branch.gt_payload = branch_payload
                branch.gt_score = raw_reward
            logger.info(
                "[%s] gt_done node=%s dt=%.1fs reward=%.3f",
                self.task_id, branch.node_id, time.perf_counter() - t_gt,
                branch.gt_score if branch.gt_score is not None else -1.0,
            )
        except Exception as exc:
            branch.gt_payload = {"error": f"{type(exc).__name__}: {exc}"}
            branch.gt_score = None
            logger.warning(
                "[%s] gt_FAILED node=%s dt=%.1fs %s",
                self.task_id, branch.node_id, time.perf_counter() - t_gt, exc,
            )

    # -- Lane C: rubric + judge per fork-group -------------------------------

    def _build_continuation_view_lane_b(self, branch: LaneBBranch) -> dict[str, Any]:
        """Lane C input shape per branch — mirrors v0's _build_continuation_view
        but reads from LaneBBranch's round1_* fields (first steps_per_round
        of execution past mid_cp), so Lane C judges partial trajectories
        of consistent length across the group."""
        workspace = branch.round1_workspace_meta or branch.workspace_meta or {}
        return {
            "node_id": branch.node_id,
            "summary": {
                "step_count": len(branch.round1_step_cards),
                "changed_files": list(workspace.get("changed_files", []))[:8],
                "untracked_files": list(workspace.get("untracked_files", []))[:8],
                "diff_stat": workspace.get("diff_stat", ""),
                "current_patch_chars": int(workspace.get("current_patch_chars", 0) or 0),
                "result_status": branch.status,
                "exit_status": "submitted" if branch.terminated_early else "",
            },
            "trajectory_continuation": {
                "step_cards": branch.round1_step_cards,
                "segment_step_range": [0, len(branch.round1_step_cards)],
            },
        }

    def _initialize_rubric_bank(self) -> None:
        """Seed the score rubric bank using the task text and stash the
        synthetic "-1" snapshot that group 0 will load. Called once per
        instance, right after Lane A captures system_prompt + user_prompt
        (since the bank seed depends on the task text)."""
        if self._rubric_bank_initialized:
            return
        self._rubric_bank_initialized = True
        if not self.config.enable_full_parity_rubric:
            return
        task_text = "\n\n".join(
            part for part in [self.system_prompt, self.user_prompt] if part
        ) or self.task
        if self.rubric_bank is not None:
            self.rubric_bank.initialize(task_text)
            self._group_bank_snapshots[-1] = {
                "active_bank": [asdict(r) for r in self.rubric_bank.active_bank],
                "inactive_bank": [],
                "persistent_state": copy.deepcopy(EMPTY_PERSISTENT_STATE),
                "recent_segments": [],
            }
        else:
            # No score bank, but downstream still wants a snapshot dict so
            # the wait/load logic doesn't KeyError. Empty bank → empty
            # extra_prompt_sections in the gen prompt.
            self._group_bank_snapshots[-1] = {
                "active_bank": [],
                "inactive_bank": [],
                "persistent_state": copy.deepcopy(EMPTY_PERSISTENT_STATE),
                "recent_segments": [],
            }
        # Sentinel event: group 0 waits on -1 which is set immediately so
        # group 0 starts processing as soon as its branches are ready.
        ev = asyncio.Event()
        ev.set()
        self._group_done_events[-1] = ev

    async def _run_lane_c_full_parity(
        self,
        *,
        group: ForkGroup,
    ) -> None:
        """Sequential-parity rubric pipeline for one ForkGroup.

        Mirrors trajectory_search.TrajectorySearchRunner._prepare_round_judging:
        loads the prior group's bank snapshot (chained via _group_done_events),
        runs PSU on evicted segments, fans out N rubric_gen samples in
        parallel, two-phase scores them (existing rubrics once + per-sample
        new rubrics), aggregates per-rubric rewards (variance + redundancy +
        judge errors), updates the score bank, picks one valid sample
        randomly, accumulates the experience-bank record, and persists the
        per-group bank snapshot + judge_response shape needed by
        lane_to_grpo_bundle.

        ForkGroups along Lane A serialize on the rubric_bank via ordered
        asyncio events keyed by group_index: group i awaits done[i-1]
        before loading bank state, then sets done[i] when finished. Lane
        B branches and concurrent GT eval are unaffected.
        """
        cfg = self.config
        t_c = time.perf_counter()
        # Wait for the prior group's bank update to land. Sentinel "-1"
        # is set in _initialize_rubric_bank so group 0 doesn't block.
        prior_idx = group.group_index - 1
        if prior_idx not in self._group_done_events:
            self._group_done_events.setdefault(prior_idx, asyncio.Event())
        # Ensure THIS group's event exists before we await prior (so a
        # later group queueing up can find ours):
        self._group_done_events.setdefault(group.group_index, asyncio.Event())
        prior_event = self._group_done_events[prior_idx]
        try:
            await prior_event.wait()
        except Exception:
            pass

        # Load bank state from prior group's snapshot. Group 0 reads the
        # synthetic "-1" snapshot seeded by _initialize_rubric_bank.
        prior_snapshot = self._group_bank_snapshots.get(
            prior_idx, self._group_bank_snapshots.get(-1, {})
        )
        if self.rubric_bank is not None:
            active_bank = [
                RubricRecord(**r) if isinstance(r, dict) else r
                for r in prior_snapshot.get("active_bank", [])
            ]
            inactive_bank = [
                RubricRecord(**r) if isinstance(r, dict) else r
                for r in prior_snapshot.get("inactive_bank", [])
            ]
            self.rubric_bank.set_state(
                active_bank=active_bank, inactive_bank=inactive_bank,
            )

        # Filter to branches that produced at least one round1 step. The
        # rubric judge needs real continuations to score; empty branches
        # (errored / overflow on first call) are dropped from the rubric
        # pipeline but still get a 0-reward dummy in the policy bundle.
        valid_branches = [
            b for b in group.branches
            if (b.error is None) and (b.round1_step_cards or b.step_cards)
        ]
        compare_parent = group.group_index > 0
        try:
            if len(valid_branches) < 2:
                # Not enough valid branches to give the judge any signal
                # — match the sequential runner's "stop_search" path by
                # leaving rubric/judge empty. lane_to_grpo_bundle treats
                # missing rubric as rubric=None → 0 reward (task 4).
                group.rubric_model_response = {
                    "error": "fewer_than_two_valid_branches",
                    "n_valid_branches": len(valid_branches),
                }
                group.judge_response = {}
                logger.info(
                    "[%s] lane_c g=%d SKIP (only %d valid branches)",
                    self.task_id, group.group_index, len(valid_branches),
                )
                return

            # Build the shared-parent context. Lane A's segment up to
            # this group's MidCp is the "latest agent trajectory" the
            # rubric prompt shows alongside each branch's continuation.
            try:
                events = (
                    group.mid_cp.snapshot.get("metadata", {})
                    .get("events", [])[: group.mid_cp.parent_event_count]
                )
                shared_step_cards = _build_step_cards(events, 0)
            except Exception:
                shared_step_cards = []
            latest_shared_segment = (
                {
                    "step_cards": shared_step_cards,
                    "segment_step_range": [0, len(shared_step_cards)],
                }
                if shared_step_cards
                else None
            )

            # PSU: if the prior group's "recent_segments" has 2+ entries,
            # evict the oldest into the persistent_state via the rubric
            # model. Matches sequential's behavior where every round's
            # judging starts by folding evicted segments into PSU.
            previous_state = prior_snapshot.get(
                "persistent_state", copy.deepcopy(EMPTY_PERSISTENT_STATE)
            )
            updated_parent_state = copy.deepcopy(previous_state)
            prior_recent = prior_snapshot.get("recent_segments", [])
            if len(prior_recent) >= 2:
                evicted = prior_recent[0].get("step_cards", [])
                if evicted:
                    try:
                        updated_parent_state = await _update_persistent_state(
                            system_prompt=self.system_prompt,
                            user_prompt=self.task,
                            previous_state=previous_state,
                            evicted_step_cards=evicted,
                            workspace_meta=copy.deepcopy(EMPTY_WORKSPACE_META),
                            model_name=self.rubric_model_name,
                            temperature=cfg.rubric_temperature,
                            top_p=cfg.rubric_top_p,
                            max_tokens=cfg.psu_max_tokens,
                            model_kwargs=None,
                        )
                    except Exception:
                        updated_parent_state = copy.deepcopy(previous_state)

            question = {
                "system_prompt": self.system_prompt,
                "user_prompt": self.task,
            }
            shared_context = {
                "previous_persistent_state": updated_parent_state,
                "latest_agent_trajectory": latest_shared_segment,
            }
            continuations = [
                self._build_continuation_view_lane_b(b) for b in valid_branches
            ]

            # Existing bank context: pulls active rubrics + the rendered
            # "## Existing Rubrics:" section appended to the gen prompt.
            existing_rubrics: list[RubricRecord] = []
            extra_prompt_sections: list[str] = []
            if self.rubric_bank is not None and self.rubric_bank.active_bank:
                ctx = self.rubric_bank.build_generation_context()
                existing_rubrics = ctx.existing_rubrics
                extra_prompt_sections.extend(ctx.extra_prompt_sections)

            # Experience-bank retrieval (if enabled): asks the rubric
            # model to pick relevant past experiences which become
            # additional prompt context for this round.
            retrieved_experiences = []
            retrieve_messages: list[dict[str, Any]] = []
            if self.experience_bank is not None and self.experience_bank.experiences:
                try:
                    exp_ctx = await self.experience_bank.build_generation_context(
                        question={**question, "instance_id": self.task_id},
                        previous_state=updated_parent_state,
                        latest_shared_segment=latest_shared_segment,
                        continuations=continuations,
                        model_name=self.rubric_model_name,
                        temperature=cfg.rubric_temperature,
                        top_p=cfg.rubric_top_p,
                        max_tokens=cfg.rubric_max_tokens,
                    )
                    extra_prompt_sections.extend(exp_ctx.extra_prompt_sections)
                    retrieved_experiences = copy.deepcopy(exp_ctx.retrieved)
                    retrieve_messages = copy.deepcopy(exp_ctx.retrieve_messages)
                except Exception as exc:
                    logger.warning(
                        "[%s] experience_bank.build_generation_context FAILED: %s",
                        self.task_id, exc,
                    )

            # Fan out N rubric generations in parallel — sequential's
            # _prepare_round_judging:1232-1240. Each draft is independent
            # so asyncio.gather is correct; the per-sample bank update
            # picks one of the N drafts based on per-rubric rewards.
            rubric_generation_kwargs = {
                "question": {**question, "instance_id": self.task_id},
                "previous_state": updated_parent_state,
                "latest_shared_segment": latest_shared_segment,
                "continuations": continuations,
                "model_name": self.rubric_model_name,
                "temperature": cfg.rubric_temperature,
                "top_p": cfg.rubric_top_p,
                "max_tokens": cfg.rubric_max_tokens,
                "round_index": group.group_index,
                "model_kwargs": None,
                "extra_prompt_sections": extra_prompt_sections,
            }
            t_rub = time.perf_counter()
            generated_samples = await asyncio.gather(
                *[
                    _generate_round_rubrics(
                        **rubric_generation_kwargs, sample_index=i,
                    )
                    for i in range(cfg.n)
                ],
            )
            logger.info(
                "[%s] lane_c g=%d rubric_gen dt=%.1fs n=%d total_rubrics=%d",
                self.task_id, group.group_index, time.perf_counter() - t_rub,
                cfg.n, sum(len(s.generated) for s in generated_samples),
            )

            # Two-phase scoring: existing rubrics judged once across all
            # branches (+ parent if compare_parent), then per-sample for
            # the newly generated rubrics. Saves N×|existing| judge
            # calls vs. the naive per-sample-full-rescoring approach.
            t_judge = time.perf_counter()
            active_continuation_scores, active_continuation_errors = await _score_round(
                question=question,
                shared_context=shared_context,
                continuations=continuations,
                rubrics=existing_rubrics,
                model_name=self.judge_model_name,
                temperature=cfg.judge_temperature,
                top_p=cfg.judge_top_p,
                max_tokens=cfg.judge_max_tokens,
                model_kwargs=None,
            )
            active_parent_scores: list[dict[str, Any]] = []
            active_parent_errors: list[dict[str, str]] = []
            parent_node_id = f"g{group.group_index - 1:03d}" if compare_parent else None
            if compare_parent and existing_rubrics:
                active_parent_scores, active_parent_errors = await _score_parent_round(
                    question=question,
                    shared_context=shared_context,
                    rubrics=existing_rubrics,
                    model_name=self.judge_model_name,
                    temperature=cfg.judge_temperature,
                    top_p=cfg.judge_top_p,
                    max_tokens=cfg.judge_max_tokens,
                    node_id=parent_node_id,
                    model_kwargs=None,
                )

            active_ids = {r.rubric_id for r in existing_rubrics}
            rubric_samples_payload: list[dict[str, Any]] = []
            valid_rubric_samples: list[dict[str, Any]] = []

            for gen_sample in generated_samples:
                # De-dup against existing + within-sample.
                seen: set[str] = set()
                generated_rubrics: list[RubricRecord] = []
                for r in gen_sample.generated:
                    if r.rubric_id in active_ids or r.rubric_id in seen:
                        continue
                    seen.add(r.rubric_id)
                    generated_rubrics.append(r)
                scoring_rubrics = existing_rubrics + generated_rubrics

                if generated_rubrics:
                    gen_cont_scores, gen_cont_errs = await _score_round(
                        question=question,
                        shared_context=shared_context,
                        continuations=continuations,
                        rubrics=generated_rubrics,
                        model_name=self.judge_model_name,
                        temperature=cfg.judge_temperature,
                        top_p=cfg.judge_top_p,
                        max_tokens=cfg.judge_max_tokens,
                        model_kwargs=None,
                    )
                else:
                    gen_cont_scores = [[] for _ in continuations]
                    gen_cont_errs = []
                scored_continuations = [
                    copy.deepcopy(a) + g
                    for a, g in zip(active_continuation_scores, gen_cont_scores)
                ]
                parent_scores = copy.deepcopy(active_parent_scores)
                judge_errors_payload = copy.deepcopy(
                    active_continuation_errors + active_parent_errors + gen_cont_errs
                )
                if compare_parent and generated_rubrics:
                    gp_scores, gp_errs = await _score_parent_round(
                        question=question,
                        shared_context=shared_context,
                        rubrics=generated_rubrics,
                        model_name=self.judge_model_name,
                        temperature=cfg.judge_temperature,
                        top_p=cfg.judge_top_p,
                        max_tokens=cfg.judge_max_tokens,
                        node_id=parent_node_id,
                        model_kwargs=None,
                    )
                    parent_scores = copy.deepcopy(active_parent_scores) + gp_scores
                    judge_errors_payload.extend(gp_errs)

                # Aggregate per-rubric: child×branch scores, variance,
                # redundancy, judge_error. Identical math to sequential.
                child_score_lookup = {b.node_id: {} for b in valid_branches}
                for branch, score_records in zip(valid_branches, scored_continuations):
                    for record in score_records:
                        child_score_lookup[branch.node_id][record["rubric_id"]] = float(
                            record["score_normalized"]
                        )
                parent_score_lookup = {
                    r["rubric_id"]: float(r["score_normalized"]) for r in parent_scores
                }

                parent_score_by_rubric: dict[str, float] = {}
                child_score_by_rubric: dict[str, dict[str, float]] = {}
                variance_by_rubric: dict[str, float] = {}
                redundency_by_rubric: dict[str, float] = {}
                judge_error_by_rubric: dict[str, float] = {}
                reward_by_rubric: dict[str, float] = {}
                previous_score_vectors: list[list[float]] = []
                for err in judge_errors_payload:
                    rid = err.get("rubric_id")
                    if rid:
                        judge_error_by_rubric[rid] = (
                            judge_error_by_rubric.get(rid, 0.0) + JUDGE_ERROR_REWARD
                        )
                for rubric in scoring_rubrics:
                    vector: list[float] = []
                    p_score = parent_score_lookup.get(rubric.rubric_id, 0.0)
                    parent_score_by_rubric[rubric.rubric_id] = p_score
                    if compare_parent:
                        vector.append(p_score)
                    cscores: dict[str, float] = {}
                    for branch in valid_branches:
                        s = child_score_lookup[branch.node_id].get(rubric.rubric_id, 0.0)
                        cscores[branch.node_id] = s
                        vector.append(s)
                    child_score_by_rubric[rubric.rubric_id] = cscores
                    variance_by_rubric[rubric.rubric_id] = (
                        0.0 if len(vector) <= 1 else float(pvariance(vector))
                    )
                    redundancy_reward = _redundancy_reward(vector, previous_score_vectors)
                    redundency_by_rubric[rubric.rubric_id] = redundancy_reward
                    reward_by_rubric[rubric.rubric_id] = (
                        variance_by_rubric[rubric.rubric_id]
                        + redundancy_reward
                        + judge_error_by_rubric.get(rubric.rubric_id, 0.0)
                    )
                    previous_score_vectors.append(vector)

                # Bank update: candidates = existing + this sample's
                # generated, ranked by reward, truncated to max_active.
                if self.rubric_bank is not None:
                    bank_update = self.rubric_bank.update_after_round(
                        generated=gen_sample.generated,
                        rewards=reward_by_rubric,
                    )
                    bank_scoring_rubrics = bank_update.rubrics
                    active_after_payload = [
                        asdict(r) for r in bank_update.active_after
                    ]
                    inactive_after_payload = [
                        asdict(r) for r in bank_update.inactive_after
                    ]
                else:
                    bank_scoring_rubrics = gen_sample.generated
                    active_after_payload = []
                    inactive_after_payload = []

                parent_reward = 0.0
                if compare_parent and bank_scoring_rubrics:
                    parent_reward = sum(
                        parent_score_lookup.get(r.rubric_id, 0.0)
                        for r in bank_scoring_rubrics
                    ) / len(bank_scoring_rubrics)
                child_rewards: dict[str, float] = {}
                for branch in valid_branches:
                    sl = child_score_lookup[branch.node_id]
                    child_rewards[branch.node_id] = (
                        sum(
                            sl.get(r.rubric_id, 0.0) * (getattr(r, "weight", 1.0) or 1.0)
                            for r in bank_scoring_rubrics
                        ) / len(bank_scoring_rubrics)
                        if bank_scoring_rubrics else 0.0
                    )

                generated_ids = {r.rubric_id for r in gen_sample.generated}
                sample_payload = {
                    "sample_index": gen_sample.sample_index,
                    "rubric_list_id": gen_sample.rubric_list_id,
                    "generated": [asdict(r) for r in gen_sample.generated],
                    "messages": gen_sample.messages,
                    "format_errors": copy.deepcopy(gen_sample.format_errors or []),
                    "terminal_error": gen_sample.terminal_error,
                    "generated_titles": [r.title for r in gen_sample.generated],
                    "child_score_by_rubric": {
                        rid: cs for rid, cs in child_score_by_rubric.items()
                        if rid in generated_ids
                    },
                    "parent_score_by_rubric": {
                        rid: s for rid, s in parent_score_by_rubric.items()
                        if rid in generated_ids
                    },
                    "child_rewards": child_rewards,
                    "parent_reward": parent_reward,
                    "variance_by_rubric": {
                        rid: v for rid, v in variance_by_rubric.items()
                        if rid in generated_ids
                    },
                    "redundency_by_rubric": {
                        rid: r for rid, r in redundency_by_rubric.items()
                        if rid in generated_ids
                    },
                    "judge_error_by_rubric": {
                        rid: judge_error_by_rubric.get(rid, 0.0)
                        for rid in generated_ids
                    },
                    "reward_by_rubric": {
                        rid: r for rid, r in reward_by_rubric.items()
                        if rid in generated_ids
                    },
                    "judge_errors": judge_errors_payload,
                    "selected": False,
                    "active_before": [asdict(r) for r in existing_rubrics],
                    "active_after": active_after_payload,
                    "inactive_after": inactive_after_payload,
                }
                if self.experience_bank is not None:
                    sample_payload.update({
                        "retrieved": [asdict(e) for e in retrieved_experiences],
                        "retrieve_messages": copy.deepcopy(retrieve_messages),
                    })
                rubric_samples_payload.append(sample_payload)
                valid_rubric_samples.append(sample_payload)

            if not valid_rubric_samples:
                group.rubric_model_response = {
                    "error": "no_valid_rubric_samples",
                    "rubric_samples": rubric_samples_payload,
                }
                group.judge_response = {}
                logger.warning(
                    "[%s] lane_c g=%d no_valid_rubric_samples",
                    self.task_id, group.group_index,
                )
                return

            selected = random.choice(valid_rubric_samples)
            selected["selected"] = True

            # Build judge_response in the shape lane_to_grpo_bundle's
            # _branch_overall_rubric_score expects: judge_response[rubric_key]
            # [f"branch_{branch_index:02d}"] = {"score_normalized": float}.
            # We use branch.branch_index from the ORIGINAL group.branches
            # ordering so a dummy/errored branch at position 2 doesn't
            # alias with valid_branches[2]'s scores.
            judge_response: dict[str, dict[str, Any]] = {}
            for rubric_idx, r_payload in enumerate(selected["generated"]):
                rid = r_payload.get("rubric_id")
                key = f"rubric_{rubric_idx:02d}"
                judge_response.setdefault(key, {})
                for branch in valid_branches:
                    s = selected["child_score_by_rubric"].get(rid, {}).get(
                        branch.node_id
                    )
                    if s is None:
                        continue
                    judge_response[key][f"branch_{branch.branch_index:02d}"] = {
                        "score_normalized": float(s),
                    }
            group.judge_response = judge_response
            group.rubric_samples = rubric_samples_payload
            group.selected_rubric_sample_index = selected["sample_index"]
            group.rubric_model_response = {
                "n_rubric_samples": len(rubric_samples_payload),
                "selected_sample_index": selected["sample_index"],
                "n_generated_total": sum(
                    len(s["generated"]) for s in rubric_samples_payload
                ),
            }
            group.active_bank_after = selected["active_after"]
            group.inactive_bank_after = selected["inactive_after"]
            group.persistent_state_after = updated_parent_state

            # Stash for the next group along Lane A.
            self._group_bank_snapshots[group.group_index] = {
                "active_bank": selected["active_after"],
                "inactive_bank": selected["inactive_after"],
                "persistent_state": updated_parent_state,
                # recent_segments rolls a 2-entry window: prior tail +
                # this group's shared segment. PSU evicts the prior tail
                # on the NEXT group.
                "recent_segments": (
                    list(prior_snapshot.get("recent_segments", []))
                    + ([{
                        "step_cards": shared_step_cards,
                        "segment_step_range": [0, len(shared_step_cards)],
                    }] if shared_step_cards else [])
                )[-2:],
            }

            # Record for ExperienceRubricBank.update_after_instance at
            # end of run. Mirrors sequential's `rubric_update_records`
            # accumulation in run() (trajectory_search.py:869, 883).
            if self.experience_bank is not None:
                for sample in valid_rubric_samples:
                    self._rubric_update_records.append({
                        "rubric_payload": {
                            "sample_index": sample["sample_index"],
                            "rubric_list_id": sample["rubric_list_id"],
                            "generated": sample["generated"],
                            "child_rewards": sample["child_rewards"],
                            "parent_reward": sample["parent_reward"],
                            "reward_by_rubric": sample["reward_by_rubric"],
                            "group_index": group.group_index,
                        },
                        "messages": sample["messages"],
                    })

            logger.info(
                "[%s] lane_c g=%d full_parity dt=%.1fs judge_dt=%.1fs samples=%d "
                "active_after=%d inactive_after=%d",
                self.task_id, group.group_index,
                time.perf_counter() - t_c, time.perf_counter() - t_judge,
                len(rubric_samples_payload),
                len(selected["active_after"]), len(selected["inactive_after"]),
            )
        except Exception as exc:
            logger.warning(
                "[%s] lane_c g=%d full_parity FAILED: %s\n%s",
                self.task_id, group.group_index, exc, traceback.format_exc(),
            )
            group.rubric_model_response = {
                "error": f"{type(exc).__name__}: {exc}",
            }
            group.judge_response = {}
            # Even on failure, propagate the prior snapshot so the next
            # group along Lane A doesn't see a broken chain.
            self._group_bank_snapshots[group.group_index] = copy.deepcopy(
                prior_snapshot
            )
        finally:
            group.lane_c_done_at = time.perf_counter()
            # Always release downstream waiters — never deadlock the
            # chain on a single group's failure.
            ev = self._group_done_events.setdefault(
                group.group_index, asyncio.Event(),
            )
            ev.set()

    async def _run_lane_c(
        self,
        *,
        group: ForkGroup,
        rubric_client: openai.AsyncOpenAI,
    ) -> None:
        """Legacy one-shot Lane C for the v1 release. Run when
        config.enable_full_parity_rubric=False.

        Builds rubric prompt from shared parent state (= Lane A state at
        MidCp) + each branch's first steps_per_round of continuation; calls
        rubric_gen → judge. Persists rubric_prompt, rubric_model_response,
        judge_response onto the ForkGroup. Persisted parent state stays
        empty (no PSU chain across mid_cps)."""
        cfg = self.config
        t_c = time.perf_counter()
        # Shared parent context: latest agent trajectory up to mid_cp.
        # Use Lane A's step cards up to mid_cp.asst_step as the "latest"
        # shared segment seen by all branches in this group.
        # (Build cheaply from MidCp's snapshot events.)
        try:
            events = (
                group.mid_cp.snapshot.get("metadata", {})
                .get("events", [])[: group.mid_cp.parent_event_count]
            )
            shared_step_cards = _build_step_cards(events, 0)
        except Exception:
            shared_step_cards = []
        latest_shared_segment = (
            {
                "step_cards": shared_step_cards,
                "segment_step_range": [0, len(shared_step_cards)],
            }
            if shared_step_cards
            else None
        )

        continuations = [
            self._build_continuation_view_lane_b(b) for b in group.branches
        ]
        prompt = _build_rubric_prompt(
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            previous_state=copy.deepcopy(EMPTY_PERSISTENT_STATE),
            latest_shared_segment=latest_shared_segment,
            continuations=continuations,
        )
        group.rubric_prompt = prompt
        group.lane_c_started_at = t_c
        logger.info(
            "[%s] lane_c g=%d rubric_start branches=%d",
            self.task_id, group.group_index, len(group.branches),
        )
        try:
            gen = await _do_rubric_gen(
                client=rubric_client,
                model_name=self.served_rubric_model_name,
                prompt=prompt,
                temperature=cfg.rubric_temperature,
                top_p=cfg.rubric_top_p,
                max_tokens=cfg.rubric_max_tokens,
            )
            group.rubric_model_response = gen
            logger.info(
                "[%s] lane_c g=%d rubric_done dt=%.1fs rubrics=%d term_err=%s",
                self.task_id, group.group_index, time.perf_counter() - t_c,
                len(gen.get("generated_rubrics", [])), gen.get("terminal_error"),
            )
        except Exception as exc:
            group.rubric_model_response = {"error": f"{type(exc).__name__}: {exc}"}
            logger.warning(
                "[%s] lane_c g=%d rubric_FAILED %s",
                self.task_id, group.group_index, exc,
            )
            group.lane_c_done_at = time.perf_counter()
            return

        rubrics = group.rubric_model_response.get("generated_rubrics", []) or []
        if not rubrics:
            group.judge_response = {}
            group.lane_c_done_at = time.perf_counter()
            return

        t_judge = time.perf_counter()
        judge_calls = []
        mapping = []
        for rubric_idx, rubric in enumerate(rubrics):
            for branch_idx, continuation in enumerate(continuations):
                judge_calls.append(
                    _do_judge_one(
                        client=rubric_client,
                        model_name=self.served_judge_model_name,
                        system_prompt=self.system_prompt,
                        user_prompt=self.user_prompt,
                        previous_state=copy.deepcopy(EMPTY_PERSISTENT_STATE),
                        latest_shared_segment=latest_shared_segment,
                        continuation=continuation,
                        rubric=rubric,
                        temperature=cfg.judge_temperature,
                        top_p=cfg.judge_top_p,
                        max_tokens=cfg.judge_max_tokens,
                    )
                )
                mapping.append((rubric_idx, branch_idx))
        results = await asyncio.gather(*judge_calls, return_exceptions=True)
        n_err = sum(
            1 for r in results
            if isinstance(r, BaseException)
            or (isinstance(r, dict) and "error" in r)
        )
        judge_response: dict[str, dict[str, Any]] = {}
        for (rubric_idx, branch_idx), res in zip(mapping, results):
            key = f"rubric_{rubric_idx:02d}"
            judge_response.setdefault(key, {})
            if isinstance(res, BaseException):
                judge_response[key][f"branch_{branch_idx:02d}"] = {"error": str(res)}
            else:
                judge_response[key][f"branch_{branch_idx:02d}"] = res
        group.judge_response = judge_response
        group.lane_c_done_at = time.perf_counter()
        logger.info(
            "[%s] lane_c g=%d judge_done dt=%.1fs calls=%d err=%d (rubrics=%d branches=%d)",
            self.task_id, group.group_index, time.perf_counter() - t_judge,
            len(results), n_err, len(rubrics), len(continuations),
        )

    async def _run_fork_group(
        self,
        *,
        mid_cp: MidCp,
        loop: asyncio.AbstractEventLoop,
        lane_b_pool: ThreadPoolExecutor,
        gt_pool: ThreadPoolExecutor,
        rubric_client: openai.AsyncOpenAI | None = None,
    ) -> ForkGroup:
        """Fork m=8 Lane B branches from mid_cp, run them all in parallel,
        evaluate GT scores concurrently. Returns the populated ForkGroup
        ready for Lane C dispatch in step 4."""
        cfg = self.config
        t_group = time.perf_counter()
        group = ForkGroup(group_index=mid_cp.idx, mid_cp=mid_cp)
        logger.info(
            "[%s] fork_group g=%d start m=%d parent_image=%s",
            self.task_id, mid_cp.idx, cfg.m, mid_cp.image_tag,
        )
        branch_futures: list[asyncio.Future] = []
        for bi in range(cfg.m):
            ctx = contextvars.copy_context()

            def _wrapped(mc=mid_cp, b=bi, c=ctx):
                return c.run(
                    self._run_lane_b_branch,
                    mid_cp=mc, branch_index=b, budget_steps=cfg.step_limit,
                )

            branch_futures.append(loop.run_in_executor(lane_b_pool, _wrapped))
        branches_or_excs = await asyncio.gather(
            *branch_futures, return_exceptions=True
        )
        branches: list[LaneBBranch] = []
        for bi, item in enumerate(branches_or_excs):
            if isinstance(item, BaseException):
                err_branch = LaneBBranch(
                    group_index=mid_cp.idx,
                    branch_index=bi,
                    node_id=f"lane-b-g{mid_cp.idx:03d}-b{bi:02d}-err",
                    parent_image_tag=mid_cp.image_tag,
                    error=f"executor: {type(item).__name__}: {item}",
                    status="error",
                )
                branches.append(err_branch)
            else:
                branches.append(item)
        group.branches = branches

        # GT eval and Lane C run CONCURRENTLY — neither depends on the
        # other. GT touches the swebench harness (CPU+filesystem), Lane C
        # hits sglang for rubric+judge — they don't conflict.
        gt_futures: list[asyncio.Future] = []
        for br in branches:
            if br.error is not None:
                br.gt_score = 0.0
                br.gt_payload = {"reward": 0.0, "note": "branch_error"}
                continue
            ctx = contextvars.copy_context()
            gt_futures.append(
                loop.run_in_executor(
                    gt_pool, lambda b=br, c=ctx: c.run(self._evaluate_gt, b)
                )
            )
        lane_c_task: asyncio.Task | None = None
        if cfg.disable_rubric:
            # GT-only training: skip rubric/judge entirely. We still need to
            # release any downstream group waiting on this group's bank-
            # carry-forward event (full-parity Lane C is the only writer
            # under normal config) so the chain doesn't deadlock.
            ev = self._group_done_events.setdefault(
                group.group_index, asyncio.Event(),
            )
            ev.set()
        elif self.config.enable_full_parity_rubric:
            # Full-parity Lane C reuses sequential's helpers via global
            # route registry; no client param needed. Bank carry-forward
            # is serialized by group_index via ordered events inside.
            lane_c_task = asyncio.create_task(
                self._run_lane_c_full_parity(group=group)
            )
        elif rubric_client is not None:
            lane_c_task = asyncio.create_task(
                self._run_lane_c(group=group, rubric_client=rubric_client)
            )

        pending: list[asyncio.Future | asyncio.Task] = []
        pending.extend(gt_futures)
        if lane_c_task is not None:
            pending.append(lane_c_task)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        # Free the mid_cp image — all branches forked from it have terminated,
        # been GT-scored, and Lane C has read what it needs from the snapshot.
        self._delete_image(mid_cp.image_tag)
        logger.info(
            "[%s] fork_group g=%d done dt=%.1fs gt_scores=%s lane_c_dt=%.1fs",
            self.task_id, mid_cp.idx, time.perf_counter() - t_group,
            [
                round(b.gt_score, 3) if b.gt_score is not None else None
                for b in branches
            ],
            (group.lane_c_done_at - group.lane_c_started_at)
            if group.lane_c_started_at else 0.0,
        )
        return group

    # -- Lane A: linear spine ------------------------------------------------

    def _start_lane_a(self, lane_a: LaneAState) -> Any:
        """Spin up Lane A's root session, capture system+user prompts.
        Returns the session. Records started_at on `lane_a`."""
        lane_a.started_at = time.perf_counter()
        session = self._make_initial_session()
        snapshot = session.snapshot().model_dump(mode="json")
        msgs = snapshot.get("agent", {}).get("state", {}).get("messages", [])
        if msgs:
            self.system_prompt = msgs[0].get("content", "") if len(msgs) >= 1 else ""
            self.user_prompt = msgs[1].get("content", "") if len(msgs) >= 2 else ""
        # Seed the full-parity rubric bank now that we have task text. The
        # sentinel _group_done_events[-1] gets set so group 0's Lane C can
        # proceed immediately. Safe no-op when full-parity is disabled.
        self._initialize_rubric_bank()
        return session

    def _run_lane_a_chunk(
        self,
        session: Any,
        *,
        max_steps: int,
        prev_message_count: int,
        prev_turn_count: int,
        prev_event_count: int,
    ) -> dict[str, Any]:
        """Run Lane A for up to `max_steps` agent steps. Returns a dict with:
        result            (the run_until_pause return)
        snapshot          (full session snapshot after)
        new_messages      (messages added in this chunk)
        new_step_cards    (step cards from this chunk's events)
        new_turns         (TurnTokenInfo extracted for this chunk)
        workspace_meta    (current workspace state)
        """
        result = self._step_session(session, max_steps=max_steps)
        snapshot_after = session.snapshot().model_dump(mode="json")
        all_msgs = snapshot_after.get("agent", {}).get("state", {}).get("messages", [])
        new_messages = copy.deepcopy(all_msgs[prev_message_count:])
        segment_events = copy.deepcopy(
            snapshot_after.get("metadata", {}).get("events", [])[prev_event_count:]
        )
        new_step_cards = _build_step_cards(segment_events, 0)
        new_turns = self._extract_turn_token_info(snapshot_after, prev_turn_count)
        workspace_meta = _collect_workspace_meta(session.agent.env)
        return {
            "result": result,
            "snapshot": snapshot_after,
            "new_messages": new_messages,
            "new_step_cards": new_step_cards,
            "new_turns": new_turns,
            "workspace_meta": workspace_meta,
        }

    async def _lane_a_loop(
        self,
        *,
        lane_a: LaneAState,
        loop: asyncio.AbstractEventLoop,
        lane_a_pool: ThreadPoolExecutor,
        on_mid_cp=None,
    ) -> None:
        """Drive Lane A: run in chunks of steps_per_round, commit + emit
        a MidCp at each boundary, until the agent terminates or step_limit
        is reached. Lane A runs in a background thread (its agent calls
        block on sglang) so this coroutine just orchestrates."""
        cfg = self.config
        t_start = time.perf_counter()
        try:
            ctx = contextvars.copy_context()
            session = await loop.run_in_executor(
                lane_a_pool, lambda c=ctx: c.run(self._start_lane_a, lane_a)
            )
        except Exception as exc:
            lane_a.error = f"lane_a_start: {type(exc).__name__}: {exc}"
            lane_a.status = "error"
            lane_a.finished_at = time.perf_counter()
            logger.warning("[%s] lane_a_start FAILED: %s", self.task_id, lane_a.error)
            return

        all_msgs_count = 0
        all_turns_count = 0
        all_events_count = 0
        # The initial snapshot already contains the system+user prompts but
        # no asst turns yet. Count baseline:
        try:
            snap0 = session.snapshot().model_dump(mode="json")
            all_msgs_count = len(
                snap0.get("agent", {}).get("state", {}).get("messages", [])
            )
            all_turns_count = len(snap0.get("metadata", {}).get("model_turns", []))
            all_events_count = len(snap0.get("metadata", {}).get("events", []))
        except Exception:
            pass

        chunks_emitted = 0
        cumulative_steps = 0
        last_result: dict[str, Any] | None = None

        # Always emit an EXTRA MidCp at idx=0 from the bare initial state
        # (system+user only, zero asst turns) so Lane B forks m branches
        # from the problem statement itself, in addition to the Lane A
        # chunked MidCps (which then get idx=1, 2, ...). Motivated by 52826
        # where ~17% of rubric calls 400'd because Lane A's prefix at late
        # mid_cps pushed the rubric input past sglang's 80960 context:
        # a root-fork branch keeps that input bounded to system+user + 8
        # short Lane B tails.
        mid_cp_idx_offset = 0
        ctx_root_snap = contextvars.copy_context()
        try:
            root_snap = await loop.run_in_executor(
                lane_a_pool,
                lambda s=session, c=ctx_root_snap: c.run(
                    lambda: s.snapshot().model_dump(mode="json")
                ),
            )
            ctx_root_commit = contextvars.copy_context()
            root_image_tag = await loop.run_in_executor(
                lane_a_pool,
                lambda s=session, c=ctx_root_commit: c.run(
                    self._commit_container, s, "mid-root"
                ),
            )
        except Exception as exc:
            logger.warning(
                "[%s] root fork emit FAILED (skipping root branch): %s",
                self.task_id, exc,
            )
            root_image_tag = None
        if root_image_tag is not None:
            root_mid_cp = MidCp(
                idx=0,
                asst_step=0,
                image_tag=root_image_tag,
                snapshot=copy.deepcopy(root_snap),
                parent_message_count=all_msgs_count,
                parent_event_count=all_events_count,
                parent_turn_count=all_turns_count,
                emitted_at=time.perf_counter(),
            )
            lane_a.mid_cps.append(root_mid_cp)
            logger.info(
                "[%s] lane_a mid_cp emitted idx=0 image=%s asst_step=0 (ROOT)",
                self.task_id, root_image_tag,
            )
            if on_mid_cp is not None:
                try:
                    on_mid_cp(root_mid_cp)
                except Exception as exc:
                    logger.warning(
                        "[%s] on_mid_cp(root) raised: %s", self.task_id, exc,
                    )
            mid_cp_idx_offset = 1

        try:
            while cumulative_steps < cfg.step_limit:
                remaining = cfg.step_limit - cumulative_steps
                chunk_size = min(cfg.steps_per_round, remaining)
                t_chunk = time.perf_counter()
                ctx_c = contextvars.copy_context()
                try:
                    chunk = await loop.run_in_executor(
                        lane_a_pool,
                        lambda s=session, ms=chunk_size, pmc=all_msgs_count, ptc=all_turns_count, pec=all_events_count, c=ctx_c: c.run(
                            self._run_lane_a_chunk,
                            s,
                            max_steps=ms,
                            prev_message_count=pmc,
                            prev_turn_count=ptc,
                            prev_event_count=pec,
                        ),
                    )
                except Exception as exc:
                    lane_a.error = f"lane_a_chunk: {type(exc).__name__}: {exc}"
                    lane_a.status = "error"
                    break
                result = chunk["result"]
                snap = chunk["snapshot"]
                new_msgs = chunk["new_messages"]
                new_step_cards = chunk["new_step_cards"]
                new_turns = chunk["new_turns"]
                last_result = result

                # Append to Lane A record.
                lane_a.messages.extend(new_msgs)
                tp = lane_a.total_tokens.get("prompt", 0)
                tc = lane_a.total_tokens.get("completion", 0)
                lane_a.total_tokens = {
                    "prompt": tp + sum(t.prompt_tokens for t in new_turns),
                    "completion": tc + sum(t.completion_tokens for t in new_turns),
                }

                # Update cumulative position trackers.
                all_msgs_count = len(
                    snap.get("agent", {}).get("state", {}).get("messages", [])
                )
                all_turns_count = len(snap.get("metadata", {}).get("model_turns", []))
                all_events_count = len(snap.get("metadata", {}).get("events", []))
                cumulative_steps += len(new_step_cards)
                lane_a.status = result.get("status", "")
                terminated = result.get("exit_status") == "Submitted"
                logger.info(
                    "[%s] lane_a chunk=%d dt=%.1fs new_steps=%d cumulative=%d status=%s submitted=%s",
                    self.task_id,
                    chunks_emitted,
                    time.perf_counter() - t_chunk,
                    len(new_step_cards),
                    cumulative_steps,
                    lane_a.status,
                    terminated,
                )

                if terminated:
                    lane_a.terminated_early = True
                    try:
                        lane_a.terminal_patch, lane_a.terminal_patch_from_fallback = (
                            self._extract_terminal_patch(result, session)
                        )
                    except Exception:
                        pass
                    break

                # Emit a MidCp at this boundary IFF we still have room and
                # the chunk produced at least one new step. Empty chunks
                # mean the agent stalled inside its loop — no point forking
                # from a no-progress state.
                #
                # NEW: also skip if Lane B fork from this state would
                # immediately fail with sglang context-overflow. We
                # tokenize the current Lane A messages with the agent's
                # gen-prompt suffix and check if input + max_new_tokens
                # exceeds sglang's served context length. 51963 had 113
                # Lane B branches (28% of all branches) die DOA from
                # this overflow because Lane A's prefix at late mid_cps
                # was already too close to 80960. Skipping the emit
                # saves ~docker_run + ~retry-loop wasted cost per branch.
                if chunks_emitted < cfg.max_mid_cps and len(new_step_cards) > 0:
                    ctx_o = contextvars.copy_context()
                    would_overflow = await loop.run_in_executor(
                        lane_a_pool,
                        lambda s=session, c=ctx_o: c.run(
                            self._would_fork_overflow, s
                        ),
                    )
                    # Shifted idx: when fork_from_root emitted a root MidCp at
                    # idx=0, chunked MidCps start at idx=1. max_mid_cps still
                    # caps chunk emissions independently.
                    mid_cp_idx = chunks_emitted + mid_cp_idx_offset
                    if would_overflow:
                        logger.info(
                            "[%s] lane_a mid_cp emit SKIPPED idx=%d: "
                            "Lane B fork would overflow sglang context "
                            "(asst_step=%d)",
                            self.task_id, mid_cp_idx, cumulative_steps,
                        )
                        # Count this as a chunk emission so max_mid_cps cap
                        # advances; this also means Lane A keeps growing
                        # past the overflow point but no further mid_cps
                        # will be emitted (the overflow check keeps firing).
                        chunks_emitted += 1
                        continue
                    image_tag = await loop.run_in_executor(
                        lane_a_pool,
                        lambda s=session, tk=f"mid-{mid_cp_idx:03d}", c=contextvars.copy_context(): c.run(
                            self._commit_container, s, tk
                        ),
                    )
                    if image_tag is None:
                        logger.warning(
                            "[%s] lane_a mid_cp idx=%d: docker commit returned None, "
                            "skipping fork-group emit",
                            self.task_id,
                            mid_cp_idx,
                        )
                    else:
                        mid_cp = MidCp(
                            idx=mid_cp_idx,
                            asst_step=cumulative_steps,
                            image_tag=image_tag,
                            snapshot=copy.deepcopy(snap),
                            parent_message_count=all_msgs_count,
                            parent_event_count=all_events_count,
                            parent_turn_count=all_turns_count,
                            emitted_at=time.perf_counter(),
                        )
                        lane_a.mid_cps.append(mid_cp)
                        logger.info(
                            "[%s] lane_a mid_cp emitted idx=%d image=%s asst_step=%d",
                            self.task_id,
                            mid_cp.idx,
                            mid_cp.image_tag,
                            mid_cp.asst_step,
                        )
                        if on_mid_cp is not None:
                            try:
                                on_mid_cp(mid_cp)
                            except Exception as exc:
                                logger.warning(
                                    "[%s] on_mid_cp callback raised: %s",
                                    self.task_id,
                                    exc,
                                )

                # Soft cap: if we've already emitted max_mid_cps, finishing
                # the run is fine but no more forks will be scheduled.
                chunks_emitted += 1

            # Loop exit: if not terminated_early, capture best-effort terminal patch.
            if not lane_a.terminated_early:
                try:
                    lane_a.terminal_patch, lane_a.terminal_patch_from_fallback = (
                        self._extract_terminal_patch(last_result or {}, session)
                    )
                except Exception:
                    pass
        finally:
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
            lane_a.finished_at = time.perf_counter()
            logger.info(
                "[%s] lane_a_done dt=%.1fs status=%s mid_cps=%d cum_steps=%d patch_len=%d",
                self.task_id,
                time.perf_counter() - t_start,
                lane_a.status,
                len(lane_a.mid_cps),
                cumulative_steps,
                len(lane_a.terminal_patch),
            )

    # -- main loop -----------------------------------------------------------

    async def run(self, *, on_group_done=None) -> InstanceRecord:
        """Run the full lane-based pipeline.

        on_group_done: optional sync callback invoked once per ForkGroup
            AFTER (a) all m Lane B branches terminated or hit step_limit,
            and (b) Lane C judge/rubric returned. Callback signature:
            (fork_group: ForkGroup) -> None. Used by slime's rollout fn to
            stream completed groups to the data buffer without waiting for
            the whole instance to finish.
        """
        instance_record = InstanceRecord(
            instance_id=self.task_id,
            run_dir=str(self.run_dir),
            task_id=self.task_id,
            config=asdict(self.config),
        )
        (self.run_dir / "config.json").write_text(
            json.dumps(asdict(self.config), indent=2)
        )
        started = time.perf_counter()
        logger.info(
            "[%s] run.start m=%d max_mid_cps=%d steps_per_round=%d step_limit=%d policy_url=%s rubric_url=%s",
            self.task_id,
            self.config.m,
            self.config.max_mid_cps,
            self.config.steps_per_round,
            self.config.step_limit,
            self.policy_base_url,
            self.rubric_base_url,
        )

        loop = asyncio.get_running_loop()
        cfg = self.config
        lane_a_pool = ThreadPoolExecutor(
            max_workers=cfg.lane_a_pool_size,
            thread_name_prefix=f"lane-a-{self.task_id[:12]}",
        )
        lane_b_pool = ThreadPoolExecutor(
            max_workers=cfg.lane_b_pool_size or (cfg.m * cfg.max_mid_cps),
            thread_name_prefix=f"lane-b-{self.task_id[:12]}",
        )
        gt_pool = ThreadPoolExecutor(
            max_workers=cfg.gt_eval_workers,
            thread_name_prefix=f"lane-gt-{self.task_id[:12]}",
        )
        rubric_client = _make_async_client(self.rubric_base_url, self.api_key)

        # Per-group on_group_done fires AS SOON AS each group is ready
        # (all m Lane B branches + GT + Lane C all done). Slime rollout
        # function reads these to stream completed groups to the data
        # buffer without waiting for the whole instance to finish.

        group_tasks: list[asyncio.Task] = []

        async def _await_group_then_callback(coro):
            try:
                grp = await coro
            except Exception as exc:
                logger.warning("[%s] fork-group raised: %s", self.task_id, exc)
                return None
            if on_group_done is not None:
                try:
                    on_group_done(grp)
                except Exception as cb_exc:
                    logger.warning(
                        "[%s] on_group_done raised: %s", self.task_id, cb_exc
                    )
            return grp

        def _on_mid_cp(mid_cp: MidCp) -> None:
            # Fire-and-track: each mid_cp emission spawns an async task that
            # forks m=8 Lane B branches, runs them to termination, then
            # GT + Lane C concurrently. We do NOT await here — Lane A
            # keeps producing more mid_cps.
            coro = self._run_fork_group(
                mid_cp=mid_cp,
                loop=loop,
                lane_b_pool=lane_b_pool,
                gt_pool=gt_pool,
                rubric_client=rubric_client,
            )
            group_tasks.append(asyncio.create_task(_await_group_then_callback(coro)))

        try:
            await self._lane_a_loop(
                lane_a=instance_record.lane_a,
                loop=loop,
                lane_a_pool=lane_a_pool,
                on_mid_cp=_on_mid_cp,
            )
            # Lane A done. Wait for all fork-groups it spawned to finish.
            if group_tasks:
                logger.info(
                    "[%s] lane_a finished, awaiting %d fork-groups",
                    self.task_id, len(group_tasks),
                )
                t_drain = time.perf_counter()
                groups_or_excs = await asyncio.gather(
                    *group_tasks, return_exceptions=True
                )
                for item in groups_or_excs:
                    if isinstance(item, BaseException):
                        logger.warning(
                            "[%s] fork-group raised: %s", self.task_id, item
                        )
                        continue
                    if item is None:
                        # _await_group_then_callback swallowed the exception
                        # and logged it; on_group_done still got skipped.
                        continue
                    instance_record.groups.append(item)
                # Sort by group_index for deterministic disk layout.
                instance_record.groups.sort(key=lambda g: g.group_index)
                logger.info(
                    "[%s] fork-groups drained dt=%.1fs n_groups=%d",
                    self.task_id,
                    time.perf_counter() - t_drain,
                    len(instance_record.groups),
                )
            instance_record.completed = (
                instance_record.lane_a.error is None
                and all(
                    all(b.error is None for b in g.branches)
                    for g in instance_record.groups
                )
            )
            if not instance_record.completed and instance_record.error is None:
                instance_record.error = (
                    instance_record.lane_a.error
                    or "one or more lane_b branches errored"
                )
        except Exception as exc:
            instance_record.error = f"runner exception: {exc}\n{traceback.format_exc()}"
            instance_record.completed = False
        finally:
            instance_record.seconds = time.perf_counter() - started
            # End-of-instance experience-bank update — analogue of the
            # sequential runner's run() epilogue at trajectory_search.py
            # lines 888-908. Accumulates all per-group rubric payloads
            # and asks the rubric model to fold them into the bank.
            if (
                self.experience_bank is not None
                and self._rubric_update_records
            ):
                try:
                    await self.experience_bank.update_after_instance(
                        run_dir=self.run_dir,
                        instance=self.instance,
                        rubric_payloads=[
                            {**rec["rubric_payload"], "messages": copy.deepcopy(rec["messages"])}
                            for rec in self._rubric_update_records
                            if isinstance(rec.get("rubric_payload"), dict)
                        ],
                        model_name=self.rubric_model_name,
                        temperature=self.config.rubric_temperature,
                        top_p=self.config.rubric_top_p,
                        max_tokens=self.config.rubric_max_tokens,
                        model_kwargs=None,
                    )
                except Exception as exc:
                    logger.warning(
                        "[%s] experience_bank.update_after_instance FAILED: %s",
                        self.task_id, exc,
                    )
            # Lane A is the spine; not trained, not dumped as its own dir.
            # Each group writes its own shared_parent.json (sliced from the
            # MidCp snapshot). Lane A's terminal patch + per-mid_cp markers
            # are summarized in instance_record.json.
            for grp in instance_record.groups:
                self._dump_group(grp)
            self._dump_instance_record(instance_record)
            for pool in (lane_a_pool, lane_b_pool, gt_pool):
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            try:
                await rubric_client.close()
            except Exception:
                pass
            self._cleanup_all()
        return instance_record

    # -- persistence ---------------------------------------------------------

    def _dump_group(self, group: ForkGroup) -> None:
        """Persist one fork-group under run_dir/groups/group_NNN/.

        Each branch gets its own subdir with:
          continuation.json / continuation_raw.json — Lane B messages added
            past the MidCp fork point. _raw retains per-message token fields
            (prompt_token_ids, token_ids, logprobs) needed for training
            sample assembly downstream.
          terminal_patch.txt — final patch
          gt.json — GT score + payload
          summary.json — small metadata blob

        Group-level files:
          members.json — index of branches and their parent MidCp
          rubric_prompt.txt, rubric_response.json, judge_response.json,
            psu_input.json — Lane C outputs (empty until step 4).
        """
        gdir = self.run_dir / "groups" / f"group_{group.group_index:03d}"
        gdir.mkdir(parents=True, exist_ok=True)
        # shared_parent: the Lane A history up to this group's MidCp, shared
        # across all M=8 branches. Sliced from MidCp.snapshot to avoid
        # storing Lane A messages elsewhere. _raw retains per-message token
        # fields (prompt_token_ids/token_ids/logprobs) so the bundle
        # converter doesn't need a separate read path.
        try:
            parent_messages = list(
                (group.mid_cp.snapshot.get("agent", {}).get("state", {}) or {})
                .get("messages", [])
            )
        except Exception:
            parent_messages = []
        parent_stamped, _ = _stamp_steps(parent_messages, start_step=0)
        (gdir / "shared_parent.json").write_text(
            json.dumps(_strip_token_fields(parent_stamped), indent=2, default=str)
        )
        (gdir / "shared_parent_raw.json").write_text(
            json.dumps(parent_stamped, indent=2, default=str)
        )
        for branch in group.branches:
            bdir = gdir / "branches" / f"branch_{branch.branch_index:02d}"
            bdir.mkdir(parents=True, exist_ok=True)
            stamped, _ = _stamp_steps(branch.messages, start_step=0)
            (bdir / "continuation.json").write_text(
                json.dumps(_strip_token_fields(stamped), indent=2, default=str)
            )
            (bdir / "continuation_raw.json").write_text(
                json.dumps(stamped, indent=2, default=str)
            )
            (bdir / "terminal_patch.txt").write_text(branch.terminal_patch or "")
            (bdir / "gt.json").write_text(
                json.dumps(
                    {"gt_score": branch.gt_score, "gt_payload": branch.gt_payload},
                    indent=2,
                    default=str,
                )
            )
            (bdir / "summary.json").write_text(
                json.dumps(
                    {
                        "node_id": branch.node_id,
                        "parent_image_tag": branch.parent_image_tag,
                        "status": branch.status,
                        "terminated_early": branch.terminated_early,
                        "terminal_patch_from_fallback": branch.terminal_patch_from_fallback,
                        "terminal_patch_chars": len(branch.terminal_patch or ""),
                        "error": branch.error,
                        "n_messages": len(branch.messages),
                        "n_step_cards": len(branch.step_cards),
                        "n_round1_step_cards": len(branch.round1_step_cards),
                        "total_tokens": branch.total_tokens,
                        "started_at": branch.started_at,
                        "round1_done_at": branch.round1_done_at,
                        "finished_at": branch.finished_at,
                    },
                    indent=2,
                    default=str,
                )
            )
        (gdir / "members.json").write_text(
            json.dumps(
                {
                    "group_index": group.group_index,
                    "mid_cp": {
                        "idx": group.mid_cp.idx,
                        "image_tag": group.mid_cp.image_tag,
                        "asst_step": group.mid_cp.asst_step,
                        "emitted_at": group.mid_cp.emitted_at,
                    },
                    "branches": [
                        {
                            "branch_index": b.branch_index,
                            "node_id": b.node_id,
                            "status": b.status,
                            "gt_score": b.gt_score,
                            "n_step_cards": len(b.step_cards),
                            "terminated_early": b.terminated_early,
                        }
                        for b in group.branches
                    ],
                },
                indent=2,
                default=str,
            )
        )
        (gdir / "rubric_prompt.txt").write_text(group.rubric_prompt or "")
        (gdir / "rubric_response.json").write_text(
            json.dumps(group.rubric_model_response, indent=2, default=str)
        )
        (gdir / "judge_response.json").write_text(
            json.dumps(group.judge_response, indent=2, default=str)
        )
        (gdir / "psu_input.json").write_text(
            json.dumps(group.persisted_parent_state, indent=2, default=str)
        )
        # Full-parity additions (task 2): dump per-sample rubric payloads
        # in the same shape the sequential runner uses, so the SFT data
        # exporter can read them via a small wrapper.
        if group.rubric_samples:
            (gdir / "rubric_samples.json").write_text(
                json.dumps(
                    {
                        "group_index": group.group_index,
                        "selected_sample_index": group.selected_rubric_sample_index,
                        "rubric_samples": group.rubric_samples,
                    },
                    indent=2,
                    default=str,
                )
            )
        if group.active_bank_after or group.inactive_bank_after:
            (gdir / "rubric_bank_after.json").write_text(
                json.dumps(
                    {
                        "active_bank_after": group.active_bank_after,
                        "inactive_bank_after": group.inactive_bank_after,
                        "persistent_state_after": group.persistent_state_after,
                    },
                    indent=2,
                    default=str,
                )
            )

    def _dump_instance_record(self, instance_record: InstanceRecord) -> None:
        path = self.run_dir / "instance_record.json"
        gt_scores_per_group: list[list[float | None]] = []
        for g in instance_record.groups:
            gt_scores_per_group.append([b.gt_score for b in g.branches])
        path.write_text(
            json.dumps(
                {
                    "instance_id": instance_record.instance_id,
                    "task_id": instance_record.task_id,
                    "run_dir": instance_record.run_dir,
                    "completed": instance_record.completed,
                    "error": instance_record.error,
                    "seconds": instance_record.seconds,
                    "num_groups": len(instance_record.groups),
                    "num_mid_cps": len(instance_record.lane_a.mid_cps),
                    "config": instance_record.config,
                    "gt_scores_per_group": gt_scores_per_group,
                },
                indent=2,
                default=str,
            )
        )
