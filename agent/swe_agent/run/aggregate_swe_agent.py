#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import re
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from statistics import pvariance
from typing import Any, Sequence

from agent_rl import RolloutSessionSpec
from agent_rl.run_utils import (
    ModelRouteConfig,
    configure_model_route,
    extract_last_json_object,
    freeform_thought_model_kwargs,
    route_completion_message,
)
from swe_agent.backend import SWEAgentRolloutBackend
from swe_agent.parallel_utils import (
    _atomic_write_json as _write_json,
    compact_workspace_meta,
    extract_terminal_patch_from_session,
    messages_from_first_assistant,
    rubric_score_record,
)
from swe_agent.prompt import (
    AGGREGATE_RUBRIC_GENERATION_PROMPT,
    AGGREGATE_RUBRIC_JUDGE_PROMPT,
    AGGREGATE_TRAJECTORY_SUMMARY_PROMPT,
    RUBRIC_JUDGE_FORMAT_CORRECTION_PROMPT,
    STRUCTURED_SUMMARY_FORMAT_CORRECTION_PROMPT,
)
from swe_agent.rubric_bank import _truncate_middle
from swe_agent.run.benchmarks.swebench import (
    build_swebench_config,
    get_swebench_harness_namespace,
    get_swebench_docker_image_name,
    get_swebench_singularity_image_name,
    load_swebench_instances_by_id,
    load_swebench_instances,
    select_container_environment_class,
)
from swe_agent.run.run_swe_agent import (
    DEFAULT_COMPLETION_MAX_TOKENS,
    DEFAULT_ENV_TIMEOUT,
    DEFAULT_LOG_ROOT,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MODEL_CLASS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PULL_TIMEOUT,
    DEFAULT_SERVE_MODEL,
    DEFAULT_SGLANG_IMAGE,
    DEFAULT_SPLIT,
    DEFAULT_STEP_LIMIT,
    DEFAULT_SUBSET,
    DEFAULT_VLLM_PORT,
    ParseInstanceIds,
    SLIME_API_BASE,
    SLIME_API_KEY,
    SLIME_SERVICE_NAME,
    SWE_AGENT_TEXTBASED_CONFIG,
    VLLM_SERVICE_NAME,
    _resolve_model_name,
    _openai_server_ready,
    _litellm_model_kwargs,
    _start_sglang_server,
    _stop_sglang_server,
    build_messages,
    choose_gpus,
    clear_policy_route,
    configure_policy_route,
    evaluate_swebench_instance_patches,
    find_free_port,
    format_swe_agent_observation,
    infer_litellm_api_env,
    make_evaluation_payload,
    tee_console,
    temporary_env,
    terminate_process,
    wait_for_openai_server,
    write_failure_artifacts,
)
from swe_agent.trajectory_search import (
    EMPTY_PERSISTENT_STATE,
    MAX_FORMAT_CORRECTION_ROUNDS,
    MAX_RUBRIC_GENERATION_ROUNDS,
    RubricRecord,
    SearchConfig,
    _avg_scores_from_rubrics,
    _build_step_cards,
    _collect_workspace_meta,
    _convert_rubric_item,
    _parse_judge_result,
    _parse_persistent_state_response,
    _render_step_cards,
    _rubric_judge_view,
)
from swe_agent.models.litellm_model import LitellmModel
from swe_agent.models.utils.retry import retry


logger = logging.getLogger("swe_agent.aggregate_swe_agent")
DEFAULT_AGGREGATE_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "aggregate_outputs"
AGGREGATE_RUBRIC_CONTINUE_PROMPT = (
    "Generate the next best aggregate-trajectory rubric or return an empty object. Follow the output format example above."
)
MAX_AGGREGATE_RUBRICS = 5
MAX_AGGREGATE_ASSISTANT_MESSAGE_CHARS = 3000


async def _summarize_aggregate_trajectory(
    *,
    question: dict[str, str],
    step_cards: list[dict[str, Any]],
    trajectory_metadata: dict[str, Any],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    user_prompt = "\n".join(
        [
            "## Question:",
            f"System Prompt:\n{question.get('system_prompt', '')}",
            "",
            f"User Prompt:\n{question.get('user_prompt', '')}",
            "",
            "## Trajectory Metadata:",
            json.dumps(trajectory_metadata, indent=2, ensure_ascii=False),
            "",
            "## Full Agent Trajectory:",
            _render_step_cards(step_cards),
        ]
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": AGGREGATE_TRAJECTORY_SUMMARY_PROMPT.strip()},
        {"role": "user", "content": user_prompt},
    ]
    last_error = ""
    try:
        for _ in range(MAX_FORMAT_CORRECTION_ROUNDS):
            async for attempt in retry(
                logger=logger,
                abort_exceptions=LitellmModel.abort_exceptions,
                model_name=model_name,
                async_retry=True,
            ):
                with attempt:
                    assistant_message = await route_completion_message(
                        route_name="rubric_generation",
                        model_name=model_name,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        model_kwargs=freeform_thought_model_kwargs(model_kwargs),
                    )
            response = assistant_message.get("content_no_thinking") or assistant_message.get("content") or ""
            messages.append(assistant_message)
            parsed = _parse_persistent_state_response(response)
            full_response = assistant_message.get("content") or ""
            if parsed is None and full_response != response:
                parsed = _parse_persistent_state_response(full_response)
            if parsed is not None:
                return {
                    "state": {**copy.deepcopy(EMPTY_PERSISTENT_STATE), **parsed},
                    "messages": messages,
                    "format_error": None,
                }
            last_error = "No JSON object could be parsed from the summary response."
            messages.append(
                {
                    "role": "user",
                    "content": STRUCTURED_SUMMARY_FORMAT_CORRECTION_PROMPT.strip(),
                }
            )
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {exc}"
    return {
        "state": copy.deepcopy(EMPTY_PERSISTENT_STATE),
        "messages": messages,
        "format_error": last_error or "Summary generation failed.",
    }


def _aggregate_rubric_user_prompt(
    *,
    question: dict[str, str],
    trajectory_summaries: list[dict[str, Any]],
) -> str:
    prompt_parts = [
        AGGREGATE_RUBRIC_GENERATION_PROMPT.strip(),
        "",
        "## Question:",
        f"System Prompt:\n{question.get('system_prompt', '')}",
        "",
        f"User Prompt:\n{question.get('user_prompt', '')}",
        "",
        "## Trajectory Summaries:",
    ]
    for index, trajectory_summary in enumerate(trajectory_summaries, start=1):
        prompt_parts.extend(
            [
                f"## Trajectory {index}:",
                json.dumps(trajectory_summary, indent=2, ensure_ascii=False),
                "",
            ]
        )
    return "\n".join(prompt_parts)


async def _generate_aggregate_rubrics(
    *,
    question: dict[str, str],
    trajectory_summaries: list[dict[str, Any]],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    round_index: int,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_text = "\n\n".join(part for part in [question.get("system_prompt", ""), question.get("user_prompt", "")] if part)
    generated: list[RubricRecord] = []
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": _aggregate_rubric_user_prompt(
                question=question,
                trajectory_summaries=trajectory_summaries,
            ),
        }
    ]
    format_errors: list[dict[str, Any]] = []
    terminal_error = None
    stop_reason = None
    turn_index = 1

    while turn_index <= MAX_RUBRIC_GENERATION_ROUNDS:
        parsed_rubric: RubricRecord | None = None
        last_error = ""
        assistant_message: dict[str, Any] | None = None
        try:
            async for attempt in retry(
                logger=logger,
                abort_exceptions=LitellmModel.abort_exceptions,
                model_name=model_name,
                async_retry=True,
            ):
                with attempt:
                    assistant_message = await route_completion_message(
                        route_name="rubric_generation",
                        model_name=model_name,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        model_kwargs=freeform_thought_model_kwargs(model_kwargs),
                    )
        except Exception as exc:
            terminal_error = f"{type(exc).__name__}: {exc}"
            stop_reason = "request_error"
            break
        if assistant_message is None:
            terminal_error = "Rubric generation request completed without an assistant message."
            stop_reason = "request_error"
            break

        messages.append(assistant_message)
        response = assistant_message.get("content_no_thinking") or assistant_message.get("content") or ""
        if os.getenv("AGGREGATE_DEBUG_RUBRICS"):
            print("\n===== AGGREGATE RUBRIC RESPONSE =====\n")
            print(response)
        parsed = extract_last_json_object(response)
        full_response = assistant_message.get("content") or ""
        if full_response != response:
            full_candidate = extract_last_json_object(full_response)
            if parsed is None or (
                isinstance(full_candidate, dict)
                and _convert_rubric_item(task_text, full_candidate, round_index) is not None
            ):
                parsed = full_candidate
        if parsed == {}:
            if generated:
                stop_reason = "empty_object"
                break
            last_error = "Expected at least one aggregate rubric before returning {}."
        elif not isinstance(parsed, dict):
            last_error = "Expected a JSON object or {}, but no JSON object could be parsed."
        else:
            parsed_rubric = _convert_rubric_item(task_text, parsed, round_index)
            if parsed_rubric is None:
                last_error = (
                    "Expected a rubric object with polarity, positive weight, title, description, metadata, "
                    "and a 1-5 scale."
                )
            elif re.sub(r"\s+", " ", parsed_rubric.title).strip().casefold() in {
                re.sub(r"\s+", " ", rubric.title).strip().casefold() for rubric in generated
            }:
                last_error = (
                    f"The rubric title {parsed_rubric.title!r} duplicates one already generated in this "
                    "aggregate conversation. Generate a criterion with a different title and judging focus, "
                    "or return {} if no useful non-redundant criterion remains."
                )
                parsed_rubric = None

        if parsed_rubric is None:
            format_errors.append({"turn_index": turn_index, "error": last_error})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{last_error} Follow the output format example above. The final JSON must contain polarity, "
                        "positive weight, title, description, metadata, and scale fields. Return {} only if at least one "
                        "complete rubric has already "
                        f"been generated and no useful non-redundant rubric remains. {AGGREGATE_RUBRIC_CONTINUE_PROMPT}"
                    ),
                }
            )
            turn_index += 1
            continue

        generated.append(parsed_rubric)
        if len(generated) >= MAX_AGGREGATE_RUBRICS:
            stop_reason = "max_rubrics"
            break
        messages.append({"role": "user", "content": AGGREGATE_RUBRIC_CONTINUE_PROMPT})
        turn_index += 1

    if stop_reason is None:
        stop_reason = "max_rounds"
        if not generated:
            terminal_error = f"Reached aggregate rubric generation max rounds={MAX_RUBRIC_GENERATION_ROUNDS} without a valid rubric."
    return {
        "generated": generated,
        "messages": messages,
        "format_errors": format_errors,
        "terminal_error": terminal_error,
        "stop_reason": stop_reason,
    }


async def _score_aggregate_summaries(
    *,
    question: dict[str, str],
    trajectory_summaries: list[dict[str, Any]],
    rubrics: list[RubricRecord],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[list[list[dict[str, Any]]], dict[str, float], dict[int, str]]:
    if not trajectory_summaries or not rubrics:
        return [[] for _ in trajectory_summaries], {}, {}
    calls = []
    mapping: list[tuple[int, RubricRecord]] = []
    question_text = f"System Prompt:\n{question.get('system_prompt', '')}\n\nUser Prompt:\n{question.get('user_prompt', '')}"
    for summary_index, trajectory_summary in enumerate(trajectory_summaries):
        summary_text = json.dumps(trajectory_summary, indent=2, ensure_ascii=False)
        for rubric in rubrics:
            judge_rubric = _rubric_judge_view(rubric)
            criterion = "\n".join(
                [
                    f"Title: {judge_rubric['title']}",
                    f"Type: {judge_rubric['direction']}",
                    f"Description: {judge_rubric['description']}",
                    "Scale:",
                    *[f"{score}: {judge_rubric['scale'][str(score)]}" for score in range(1, 6)],
                    f"Metadata: {json.dumps(judge_rubric['metadata'], ensure_ascii=False)}",
                ]
            )

            async def _judge_single(
                *,
                summary_text: str = summary_text,
                criterion: str = criterion,
            ) -> tuple[list[dict[str, Any]], int, str, str | None]:
                user_prompt = "\n".join(
                    [
                        "## Question:",
                        question_text,
                        "",
                        "## Complete Trajectory Summary:",
                        summary_text,
                        "",
                        "## Criterion:",
                        criterion,
                    ]
                )
                messages = [
                    {"role": "system", "content": AGGREGATE_RUBRIC_JUDGE_PROMPT.strip()},
                    {"role": "user", "content": user_prompt},
                ]
                try:
                    for _ in range(MAX_FORMAT_CORRECTION_ROUNDS):
                        async for attempt in retry(
                            logger=logger,
                            abort_exceptions=LitellmModel.abort_exceptions,
                            model_name=model_name,
                            async_retry=True,
                        ):
                            with attempt:
                                assistant_message = await route_completion_message(
                                    route_name="rubric_judge",
                                    model_name=model_name,
                                    messages=messages,
                                    temperature=temperature,
                                    top_p=top_p,
                                    max_tokens=max_tokens,
                                    model_kwargs=freeform_thought_model_kwargs(model_kwargs),
                                )
                        messages.append(assistant_message)
                        response = assistant_message.get("content_no_thinking") or assistant_message.get("content") or ""
                        judge_result = _parse_judge_result(response)
                        full_response = assistant_message.get("content") or ""
                        if judge_result is None and full_response != response:
                            judge_result = _parse_judge_result(full_response)
                        if judge_result is not None:
                            evidence, score_raw = judge_result
                            return (
                                messages_from_first_assistant(messages),
                                score_raw,
                                evidence,
                                None,
                            )
                        messages.append(
                            {
                                "role": "user",
                                "content": RUBRIC_JUDGE_FORMAT_CORRECTION_PROMPT.strip(),
                            }
                        )
                    return (
                        messages_from_first_assistant(messages),
                        1,
                        "",
                        "InvalidAggregateJudgeScore",
                    )
                except Exception as exc:
                    return (
                        messages_from_first_assistant(messages),
                        1,
                        "",
                        f"{type(exc).__name__}: {exc}",
                    )

            calls.append(_judge_single())
            mapping.append((summary_index, rubric))
    responses = await asyncio.gather(*calls, return_exceptions=True)
    per_view_scores: list[list[dict[str, Any]]] = [[] for _ in trajectory_summaries]
    rubric_values: dict[str, list[float]] = {}
    view_errors: dict[int, str] = {}
    for (summary_index, rubric), response in zip(mapping, responses):
        if isinstance(response, Exception):
            view_errors.setdefault(summary_index, f"{type(response).__name__}: {response}")
            continue
        judge_messages, score_raw, evidence, error = response
        if error is not None:
            view_errors.setdefault(summary_index, error)
        record = rubric_score_record(
            rubric,
            score_raw,
            judge_messages,
            evidence=evidence,
        )
        record["judge_error"] = error
        per_view_scores[summary_index].append(record)
        rubric_values.setdefault(rubric.rubric_id, []).append(float(record["score_normalized"]))
    variances = {
        rubric_id: (0.0 if len(values) <= 1 else float(pvariance(values)))
        for rubric_id, values in rubric_values.items()
    }
    return per_view_scores, variances, view_errors


def _source_message_text(message: dict[str, Any]) -> str:
    for key in ("content_no_thinking", "content", "message"):
        value = message.get(key)
        if isinstance(value, str):
            return value
    return ""


def _source_step_cards(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    current_card: dict[str, Any] | None = None
    saw_assistant = False
    for message in messages:
        role = message.get("role")
        content = _source_message_text(message)
        if role == "assistant":
            saw_assistant = True
            commands = [
                tool_call.get("command", "")
                for tool_call in message.get("tool_calls", [])
                if isinstance(tool_call, dict) and tool_call.get("command")
            ]
            current_card = {
                "step_index": len(cards),
                "assistant_message": _truncate_middle(content, MAX_AGGREGATE_ASSISTANT_MESSAGE_CHARS),
                "commands": commands,
                "observation": "",
            }
        elif role == "user" and saw_assistant:
            if current_card is None:
                continue
            current_card["observation"] = format_swe_agent_observation(content)
            cards.append(current_card)
            current_card = None
    if current_card is not None:
        cards.append(current_card)
    return cards


def _reset_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


class AggregateTrajectoryRunner:
    def __init__(
        self,
        *,
        instance: dict[str, Any],
        backend: SWEAgentRolloutBackend | None,
        run_dir: Path,
        policy_model_name: str,
        rubric_model_name: str,
        judge_model_name: str,
        search_config: SearchConfig,
        num_trajectories: int,
        harness_namespace: str | None,
        source_run_dirs: Sequence[Path] | None = None,
        rubric_model_kwargs: dict[str, Any] | None = None,
        judge_model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.instance = copy.deepcopy(instance)
        self.backend = backend
        self.run_dir = Path(run_dir)
        self.nodes_dir = self.run_dir / "nodes"
        self.policy_model_name = policy_model_name
        self.rubric_model_name = rubric_model_name or policy_model_name
        self.judge_model_name = judge_model_name or policy_model_name
        self.search_config = search_config
        self.num_trajectories = num_trajectories
        self.harness_namespace = harness_namespace
        self.source_run_dirs = [Path(path) for path in (source_run_dirs or [])]
        self.rubric_model_kwargs = copy.deepcopy(rubric_model_kwargs or {})
        self.judge_model_kwargs = copy.deepcopy(judge_model_kwargs or {})
        self.task = instance["problem_statement"]
        self.task_id = instance["instance_id"]
        self.system_prompt = ""
        self.user_prompt = self.task

    def _run_candidate(self, candidate_index: int) -> dict[str, Any]:
        if self.backend is None:
            raise RuntimeError("A rollout backend is required when source trajectories are not configured.")
        node_id = f"node_{candidate_index:03d}"
        node_dir = self.nodes_dir / node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        session = None
        try:
            spec = RolloutSessionSpec(
                session_id=f"{node_id}-{uuid.uuid4().hex}",
                task=self.task,
                task_id=self.task_id,
                sample_index=candidate_index,
                policy_ref=self.policy_model_name,
                policy_version=self.policy_model_name,
                dataset_name="swebench",
                ground_truth=self.instance.get("patch"),
                raw_user_query=self.task,
                limits={"step_limit": self.backend.agent_config.get("step_limit", 0)},
                metadata={"template_vars": copy.deepcopy(self.instance)},
            )
            session = self.backend.create_session(spec)
            initial_snapshot = session.snapshot().model_dump(mode="json")
            initial_messages = initial_snapshot["agent"]["state"].get("messages", [])
            system_prompt = initial_messages[0].get("content", "") if initial_messages else ""
            user_prompt = initial_messages[1].get("content", "") if len(initial_messages) > 1 else self.task
            terminal_error = None
            try:
                result = session.run_until_pause(max_steps=self.backend.agent_config.get("step_limit"))
                result_payload = result.model_dump(mode="json")
            except Exception as exc:
                terminal_error = f"{type(exc).__name__}: {exc}"
                result_payload = {
                    "status": "error",
                    "exit_status": type(exc).__name__,
                    "submission": "",
                    "metadata": {},
                }
            snapshot_payload = session.snapshot().model_dump(mode="json")
            try:
                workspace_meta = _collect_workspace_meta(session.agent.env)
            except Exception as exc:
                workspace_meta = {"error": f"{type(exc).__name__}: {exc}"}
            terminal_patch, patch_from_fallback = extract_terminal_patch_from_session(result_payload, session)
            slim_traj = build_messages(
                snapshot_payload["agent"]["state"].get("messages", []),
                model_name=self.policy_model_name,
                preserve_token_fields=True,
            )
            patch_payload = {
                self.task_id: {
                    "model_name_or_path": self.policy_model_name,
                    "instance_id": self.task_id,
                    "model_patch": terminal_patch,
                    "patch_from_fallback": patch_from_fallback,
                }
            }
            node_payload = {
                "node_id": node_id,
                "sample_index": candidate_index,
                "session_id": snapshot_payload.get("session_id", ""),
                "status": result_payload.get("status", ""),
                "exit_status": result_payload.get("exit_status", ""),
                "step_count": int(result_payload.get("metadata", {}).get("n_calls", 0) or 0),
                "terminal_patch_chars": len(terminal_patch),
                "terminal_patch_from_fallback": patch_from_fallback,
                "terminal_error": terminal_error,
                "policy_model_name": self.policy_model_name,
            }
            if terminal_patch.strip():
                try:
                    terminal_evaluation = evaluate_swebench_instance_patches(
                        instance=self.instance,
                        patches_by_key={node_id: terminal_patch},
                        model_name=self.policy_model_name,
                        max_workers=1,
                        namespace=self.harness_namespace,
                        work_dir=node_dir,
                    ).get(node_id, make_evaluation_payload("error", error="Missing node evaluation"))
                except Exception as exc:
                    terminal_evaluation = make_evaluation_payload("error", error=exc)
            else:
                terminal_evaluation = make_evaluation_payload("empty")
            _write_json(node_dir / "messages.json", slim_traj)
            _write_json(node_dir / "terminal_patch.json", patch_payload)
            _write_json(node_dir / "terminal_evaluation.json", terminal_evaluation)
            return {
                "node_id": node_id,
                "candidate_index": candidate_index,
                "node_dir": node_dir,
                "node_payload": node_payload,
                "messages_payload": slim_traj,
                "patch_payload": patch_payload,
                "snapshot": snapshot_payload,
                "result": result_payload,
                "workspace_meta": workspace_meta,
                "step_cards": _build_step_cards(snapshot_payload.get("metadata", {}).get("events", []), step_start=0),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "terminal_patch": terminal_patch,
                "terminal_patch_from_fallback": patch_from_fallback,
                "terminal_error": terminal_error,
                "terminal_evaluation": terminal_evaluation,
            }
        finally:
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

    def _load_source_candidate(self, candidate_index: int, source_run_dir: Path) -> dict[str, Any]:
        node_id = f"node_{candidate_index:03d}"
        node_dir = self.nodes_dir / node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        messages_payload = _load_json_object(source_run_dir / "messages.json")
        patch_payload = _load_json_object(source_run_dir / "model_patch.json")
        terminal_evaluation = _load_json_object(source_run_dir / "evaluation.json")
        messages = messages_payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Missing messages list in {source_run_dir / 'messages.json'}")
        patch_record = patch_payload.get(self.task_id)
        if not isinstance(patch_record, dict):
            raise ValueError(f"Missing patch for {self.task_id} in {source_run_dir / 'model_patch.json'}")
        terminal_patch = str(patch_record.get("model_patch") or "")
        terminal_patch_from_fallback = bool(patch_record.get("patch_from_fallback", False))
        step_cards = _source_step_cards(messages)
        system_prompt = next(
            (_source_message_text(message) for message in messages if message.get("role") == "system"),
            "",
        )
        user_prompt = next(
            (_source_message_text(message) for message in messages if message.get("role") == "user"),
            self.task,
        )
        evaluation_status = str(terminal_evaluation.get("status") or "")
        result_payload = {
            "status": "finished" if terminal_patch.strip() else "empty",
            "exit_status": evaluation_status,
            "submission": terminal_patch,
            "metadata": {"n_calls": len(step_cards)},
        }
        terminal_error = None
        if evaluation_status == "error":
            terminal_error = str((terminal_evaluation.get("metainfo") or {}).get("output") or "")
        changed_files = re.findall(r"^\+\+\+ b/(.+)$", terminal_patch, flags=re.MULTILINE)
        workspace_meta = {
            "changed_files": changed_files,
            "current_patch_chars": len(terminal_patch),
        }
        node_payload = {
            "node_id": node_id,
            "sample_index": candidate_index,
            "status": result_payload["status"],
            "exit_status": evaluation_status,
            "step_count": len(step_cards),
            "terminal_patch_chars": len(terminal_patch),
            "terminal_patch_from_fallback": terminal_patch_from_fallback,
            "terminal_error": terminal_error,
            "policy_model_name": self.policy_model_name,
            "source_run_dir": str(source_run_dir),
        }
        _write_json(node_dir / "messages.json", messages_payload)
        _write_json(node_dir / "terminal_patch.json", patch_payload)
        _write_json(node_dir / "terminal_evaluation.json", terminal_evaluation)
        return {
            "node_id": node_id,
            "candidate_index": candidate_index,
            "node_dir": node_dir,
            "node_payload": node_payload,
            "messages_payload": messages_payload,
            "patch_payload": patch_payload,
            "result": result_payload,
            "workspace_meta": workspace_meta,
            "step_cards": step_cards,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "terminal_patch": terminal_patch,
            "terminal_patch_from_fallback": terminal_patch_from_fallback,
            "terminal_error": terminal_error,
            "terminal_evaluation": terminal_evaluation,
        }

    def _run_candidates(self) -> list[dict[str, Any]]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        if self.source_run_dirs:
            if len(self.source_run_dirs) != self.num_trajectories:
                raise ValueError(
                    f"Expected {self.num_trajectories} source trajectories for {self.task_id}, "
                    f"found {len(self.source_run_dirs)}."
                )
            candidates = [
                self._load_source_candidate(index, source_run_dir)
                for index, source_run_dir in enumerate(self.source_run_dirs)
            ]
            self.system_prompt = candidates[0].get("system_prompt") or self.system_prompt
            self.user_prompt = candidates[0].get("user_prompt") or self.user_prompt
            return candidates
        candidates: list[dict[str, Any]] = []
        max_workers = max(1, self.num_trajectories)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aggregate-rollout") as executor:
            future_map = {executor.submit(self._run_candidate, index): index for index in range(self.num_trajectories)}
            for future in as_completed(future_map):
                try:
                    candidates.append(future.result())
                except Exception:
                    pass
        candidates.sort(key=lambda candidate: candidate["candidate_index"])
        if len(candidates) != self.num_trajectories:
            raise RuntimeError(
                f"Expected {self.num_trajectories} completed candidate trajectories for "
                f"{self.task_id}, found {len(candidates)}."
            )
        if candidates:
            self.system_prompt = candidates[0].get("system_prompt") or self.system_prompt
            self.user_prompt = candidates[0].get("user_prompt") or self.user_prompt
        return candidates

    async def _compress_and_score(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        question = {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
        }
        summary_calls = [
            _summarize_aggregate_trajectory(
                question=question,
                step_cards=candidate["step_cards"],
                trajectory_metadata={
                    "step_count": len(candidate["step_cards"]),
                    "result_status": candidate["result"].get("status", ""),
                    "exit_status": candidate["result"].get("exit_status", ""),
                },
                model_name=self.rubric_model_name,
                temperature=self.search_config.rubric_temperature,
                top_p=self.search_config.rubric_top_p,
                max_tokens=self.search_config.rubric_max_tokens,
                model_kwargs=self.rubric_model_kwargs,
            )
            for candidate in candidates
        ]
        summary_results = await asyncio.gather(*summary_calls)
        for candidate, summary_result in zip(candidates, summary_results):
            state = summary_result["state"]
            candidate["summary_messages"] = messages_from_first_assistant(summary_result["messages"])
            candidate["summary_format_error"] = summary_result.get("format_error")
            candidate["compressed_view"] = {
                "compressed_trajectory": state,
                "submited_patch": candidate.get("terminal_patch"),
                "workspace_meta": compact_workspace_meta(candidate["workspace_meta"], file_limit=12),
            }
            _write_json(
                candidate["node_dir"] / "summary.json",
                {
                    "node_id": candidate["node_id"],
                    "summary": candidate["compressed_view"],
                    "format_error": candidate.get("summary_format_error"),
                },
            )
            _write_json(
                candidate["node_dir"] / "summary_message.json",
                candidate.get("summary_messages", []),
            )

        rubric_sample = await _generate_aggregate_rubrics(
            question=question,
            trajectory_summaries=[candidate["compressed_view"] for candidate in candidates],
            model_name=self.rubric_model_name,
            temperature=self.search_config.rubric_temperature,
            top_p=self.search_config.rubric_top_p,
            max_tokens=self.search_config.rubric_max_tokens,
            round_index=1,
            model_kwargs=self.rubric_model_kwargs,
        )
        scoring_rubrics = list(rubric_sample["generated"])
        if rubric_sample.get("terminal_error"):
            raise RuntimeError(f"Aggregate rubric generation failed: {rubric_sample['terminal_error']}")
        if not scoring_rubrics:
            raise RuntimeError(
                "Aggregate rubric generation produced no valid rubric: "
                f"{rubric_sample.get('terminal_error') or rubric_sample.get('stop_reason')}"
            )
        score_records, variances, judge_errors = await _score_aggregate_summaries(
            question=question,
            trajectory_summaries=[candidate["compressed_view"] for candidate in candidates],
            rubrics=scoring_rubrics,
            model_name=self.judge_model_name,
            temperature=self.search_config.judge_temperature,
            top_p=self.search_config.judge_top_p,
            max_tokens=self.search_config.judge_max_tokens,
            model_kwargs=self.judge_model_kwargs,
        )
        score_lookup_by_node = {
            candidate["node_id"]: {
                record["rubric_id"]: float(record.get("score_normalized", 0.0))
                for record in records
            }
            for candidate, records in zip(candidates, score_records)
        }
        average_scores = _avg_scores_from_rubrics(
            node_ids=[candidate["node_id"] for candidate in candidates],
            score_lookup_by_node=score_lookup_by_node,
            rubrics=scoring_rubrics,
        )
        for index, candidate in enumerate(candidates):
            candidate["score_records"] = score_records[index]
            candidate["judge_error"] = judge_errors.get(index)
            candidate["reward"] = average_scores.get(candidate["node_id"], 0.0)
        return {
            "rubric_sample": rubric_sample,
            "scoring_rubrics": scoring_rubrics,
            "judge_errors": judge_errors,
            "average_rubric_judged_scores": average_scores,
            "variances": variances,
        }

    def run(self) -> None:
        candidates = self._run_candidates()
        if not candidates:
            raise RuntimeError("No aggregate candidate trajectories completed successfully.")

        judged = asyncio.run(self._compress_and_score(candidates))

        best_candidate = max(
            candidates,
            key=lambda candidate: (float(candidate.get("reward", 0.0)), candidate["result"].get("status") == "finished"),
        )
        score_by_rubric: dict[str, dict[str, float]] = {}
        for candidate in candidates:
            for record in candidate.get("score_records", []):
                score_by_rubric.setdefault(record["rubric_id"], {})[candidate["node_id"]] = float(record.get("score_normalized", 0.0))
        average_rubric_judged_scores = judged.get("average_rubric_judged_scores", {})
        judge_messages = [
            {
                "node_id": candidate["node_id"],
                "rubric_id": record["rubric_id"],
                "messages": copy.deepcopy(record.get("judge_message") or []),
                "error": record.get("judge_error"),
            }
            for candidate in candidates
            for record in candidate.get("score_records", [])
        ]

        _write_json(self.run_dir / "rubric_message.json", {"messages": judged["rubric_sample"]["messages"]})
        _write_json(self.run_dir / "judge_message.json", judge_messages)
        _write_json(
            self.run_dir / "rubric.json",
            {
                "generated": [asdict(rubric) for rubric in judged["rubric_sample"]["generated"]],
                "format_errors": judged["rubric_sample"]["format_errors"],
                "terminal_error": judged["rubric_sample"]["terminal_error"],
                "stop_reason": judged["rubric_sample"]["stop_reason"],
                "selected_node_id": best_candidate["node_id"],
                "selected_node_dir": str(best_candidate["node_dir"]),
                "rubric_score": float(best_candidate.get("reward", 0.0)),
                "terminal_evaluation_reward": float((best_candidate.get("terminal_evaluation") or {}).get("reward", 0.0)),
                "terminal_evaluation_status": (best_candidate.get("terminal_evaluation") or {}).get("status", ""),
                "model_patch_chars": len(best_candidate.get("terminal_patch") or ""),
                "patch_from_fallback": bool(best_candidate.get("terminal_patch_from_fallback")),
                "score_by_rubric": score_by_rubric,
                "average_rubric_judged_scores": average_rubric_judged_scores,
                "variance_by_rubric": judged["variances"],
                "judge_errors": judged["judge_errors"],
                "judge_scores": [
                    {
                        "node_id": candidate["node_id"],
                        "overall_score": candidate.get("reward", 0.0),
                        "judge_error": candidate.get("judge_error"),
                        "status": candidate["result"].get("status", ""),
                        "exit_status": candidate["result"].get("exit_status", ""),
                        "terminal_patch_chars": len(candidate.get("terminal_patch") or ""),
                        "terminal_patch_from_fallback": bool(candidate.get("terminal_patch_from_fallback")),
                        "terminal_evaluation": candidate.get("terminal_evaluation"),
                        "summary_format_error": candidate.get("summary_format_error"),
                        "scores": [
                            {key: copy.deepcopy(value) for key, value in record.items() if key != "judge_message"}
                            for record in candidate.get("score_records", [])
                        ],
                    }
                    for candidate in candidates
                ],
            },
        )
        _write_json(self.run_dir / "patch.json", best_candidate["patch_payload"])
        _write_json(self.run_dir / "evaluation.json", best_candidate.get("terminal_evaluation") or make_evaluation_payload("error", error="Missing selected node evaluation"))


def _find_source_run_dirs(args: argparse.Namespace, instance_id: str) -> list[Path]:
    if args.source_rollout_root is None:
        return []
    instance_root = args.source_rollout_root / instance_id
    if not instance_root.is_dir():
        raise FileNotFoundError(f"Source rollout instance directory does not exist: {instance_root}")
    by_lane: dict[int, Path] = {}
    for run_dir in instance_root.glob(args.source_run_pattern):
        if not run_dir.is_dir():
            continue
        match = re.search(r"-r(\d+)$", run_dir.name)
        if match is None:
            continue
        lane = int(match.group(1))
        if lane >= args.num_trajectories:
            continue
        missing = [
            filename
            for filename in ("messages.json", "model_patch.json", "evaluation.json")
            if not (run_dir / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Source rollout {run_dir} is missing: {', '.join(missing)}")
        if lane in by_lane:
            raise ValueError(f"Multiple source rollouts matched lane r{lane:02d} for {instance_id}")
        by_lane[lane] = run_dir
    expected_lanes = set(range(args.num_trajectories))
    if set(by_lane) != expected_lanes:
        missing_lanes = sorted(expected_lanes - set(by_lane))
        raise ValueError(
            f"Expected source lanes 0..{args.num_trajectories - 1} for {instance_id}; "
            f"missing {missing_lanes} under {instance_root} with pattern {args.source_run_pattern!r}."
        )
    return [by_lane[lane] for lane in sorted(by_lane)]


def run_aggregate(
    args: argparse.Namespace,
    instance_ids: Sequence[str] | None,
) -> None:
    model_name = _resolve_model_name(args)
    selected_ids = list(instance_ids or [])
    instances = (
        load_swebench_instances_by_id(args.subset, args.split, selected_ids)
        if selected_ids
        else load_swebench_instances(args.subset, args.split)
    )
    timestamp = args.timestamp or time.strftime("%Y%m%d-%H%M%S")
    run_root = args.output_root / (
        f"{re.sub(r'[^A-Za-z0-9._-]+', '_', args.subset.replace('/', '__'))}_"
        f"{re.sub(r'[^A-Za-z0-9._-]+', '_', args.split.replace('/', '__'))}_"
        f"{re.sub(r'[^A-Za-z0-9._-]+', '_', model_name.replace('/', '__'))}"
    )
    run_log_path = DEFAULT_LOG_ROOT / f"aggregate-{timestamp}.log"
    run_log_path.parent.mkdir(parents=True, exist_ok=True)

    vllm_handle = None
    sglang_server = None
    shared_model_kwargs: dict[str, Any]
    if args.backend == "vllm":
        from agent_rl.vllm_server import launch_vllm_server_handle

        gpu_ids = choose_gpus(args.gpu_id)
        gpu_id = gpu_ids[0] if gpu_ids else None
        if gpu_id is None:
            raise RuntimeError("vLLM backend requires a GPU; got gpu_id=none")
        vllm_handle = launch_vllm_server_handle(
            model_name=args.vllm_model,
            port=find_free_port(args.vllm_port),
            gpu_id=gpu_id,
            gpu_ids=gpu_ids,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            allow_long_max_model_len=args.allow_long_max_model_len,
        )
        configure_policy_route(backend_name=args.backend, model_name=model_name, base_url=vllm_handle.base_url)
        shared_model_kwargs = {"api_base": vllm_handle.base_url, "api_key": "EMPTY", **_litellm_model_kwargs(model_name)}
    elif args.backend == "sglang":
        if args.use_existing_sglang_server:
            wait_for_openai_server(args.sglang_api_base, args.sglang_api_key, timeout=args.server_timeout)
        elif not _openai_server_ready(args.sglang_api_base, args.sglang_api_key):
            sglang_server = _start_sglang_server(args, DEFAULT_LOG_ROOT / f"aggregate-{timestamp}-sglang.log")
        configure_policy_route(
            backend_name=args.backend,
            model_name=model_name,
            base_url=args.sglang_api_base,
            api_key=args.sglang_api_key,
        )
        shared_model_kwargs = {
            "api_base": args.sglang_api_base,
            "api_key": args.sglang_api_key,
            **_litellm_model_kwargs(model_name),
        }
    else:
        required_env = infer_litellm_api_env(args.openai_model)
        if required_env and not os.getenv(required_env):
            raise RuntimeError(f"{required_env} is not set for model {args.openai_model}")
        configure_policy_route(backend_name=args.backend, model_name=model_name)
        shared_model_kwargs = _litellm_model_kwargs(model_name)

    if args.backend == "vllm":
        configure_model_route("rubric_generation", ModelRouteConfig(backend="service", service_name=VLLM_SERVICE_NAME, model_name=model_name))
        configure_model_route("rubric_judge", ModelRouteConfig(backend="service", service_name=VLLM_SERVICE_NAME, model_name=model_name))
    elif args.backend == "sglang":
        configure_model_route("rubric_generation", ModelRouteConfig(backend="service", service_name=SLIME_SERVICE_NAME, model_name=model_name))
        configure_model_route("rubric_judge", ModelRouteConfig(backend="service", service_name=SLIME_SERVICE_NAME, model_name=model_name))
    else:
        configure_model_route("rubric_generation", ModelRouteConfig(backend="litellm", model_name=model_name))
        configure_model_route("rubric_judge", ModelRouteConfig(backend="litellm", model_name=model_name))
    errors: list[str] = []
    try:
        agent_model_class = "litellm_textbased" if args.backend == "openai" else DEFAULT_MODEL_CLASS
        config = None
        if args.source_rollout_root is None:
            config = build_swebench_config(
                config_spec=[str(SWE_AGENT_TEXTBASED_CONFIG)],
                model=model_name,
                model_class=agent_model_class,
                extra_overrides={
                    "agent": {
                        "step_limit": args.step_limit,
                        "cost_limit": 0,
                    },
                    "environment": {
                        "timeout": args.environment_timeout,
                        "pull_timeout": args.pull_timeout,
                    },
                    "model": {
                        "model_kwargs": {
                            "temperature": args.policy_temperature,
                            "top_p": args.policy_top_p,
                            "max_tokens": args.completion_max_tokens,
                            **shared_model_kwargs,
                        },
                        "cost_tracking": "ignore_errors",
                    },
                },
            )
        search_config = SearchConfig(
            max_active_rubrics=MAX_AGGREGATE_RUBRICS,
            policy_temperature=args.policy_temperature,
            policy_top_p=args.policy_top_p,
            rubric_temperature=args.rubric_temperature,
            rubric_top_p=args.rubric_top_p,
            rubric_max_tokens=args.rubric_max_tokens,
            judge_temperature=args.judge_temperature,
            judge_top_p=args.judge_top_p,
            judge_max_tokens=args.judge_max_tokens,
        )
        rubric_model_name = args.rubric_model or model_name
        judge_model_name = args.judge_model or model_name
        rubric_model_kwargs = dict(shared_model_kwargs)
        judge_model_kwargs = dict(shared_model_kwargs)

        def run_instance(instance: dict[str, Any]) -> None:
            run_dir = run_root / instance["instance_id"] / timestamp
            _reset_run_dir(run_dir)
            source_run_dirs = _find_source_run_dirs(args, instance["instance_id"])
            backend = None
            if config is not None:
                instance_config = copy.deepcopy(config)
                environment_config = instance_config.setdefault("environment", {})
                environment_config["environment_class"] = select_container_environment_class(
                    environment_config.get("environment_class", "docker")
                )
                if environment_config.get("environment_class") == "docker":
                    environment_config["image"] = get_swebench_docker_image_name(instance)
                elif environment_config.get("environment_class") == "singularity":
                    environment_config["image"] = get_swebench_singularity_image_name(instance)
                if instance.get("expected_output_json"):
                    environment_config["cwd"] = "/testbed"
                    environment_config.setdefault("dataset_name", "r2egym")
                else:
                    environment_config["cwd"] = (
                        instance.get("swebench_workdir")
                        or environment_config.get("cwd")
                        or "/testbed"
                    )
                backend = SWEAgentRolloutBackend(
                    model=instance_config.get("model", {}),
                    environment=instance_config.get("environment", {}),
                    agent=instance_config.get("agent", {}),
                    default_agent_type="default",
                    default_environment_type=instance_config.get("environment", {}).get(
                        "environment_class", "docker"
                    ),
                )
            runner = AggregateTrajectoryRunner(
                instance=instance,
                backend=backend,
                run_dir=run_dir,
                policy_model_name=model_name,
                rubric_model_name=rubric_model_name,
                judge_model_name=judge_model_name,
                search_config=search_config,
                num_trajectories=args.num_trajectories,
                harness_namespace=get_swebench_harness_namespace(instance),
                source_run_dirs=source_run_dirs,
                rubric_model_kwargs=rubric_model_kwargs,
                judge_model_kwargs=judge_model_kwargs,
            )
            runner.run()

        def record_failure(instance: dict[str, Any], exc: Exception) -> None:
            run_dir = run_root / instance["instance_id"] / timestamp
            errors.append(f"{instance['instance_id']}: {exc}")
            write_failure_artifacts(
                instance_id=instance["instance_id"],
                run_dir=run_dir,
                error=exc,
                log_path=run_log_path,
            )

        with temporary_env({"LITELLM_LOG": "ERROR"}):
            with tee_console(run_log_path):
                if args.workers == 1:
                    for instance in instances:
                        try:
                            run_instance(instance)
                        except Exception as exc:
                            record_failure(instance, exc)
                else:
                    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="aggregate-instance") as executor:
                        future_to_instance = {
                            executor.submit(run_instance, instance): instance
                            for instance in instances
                        }
                        for future in as_completed(future_to_instance):
                            instance = future_to_instance[future]
                            try:
                                future.result()
                            except Exception as exc:
                                record_failure(instance, exc)
        if errors:
            raise RuntimeError("; ".join(errors))
    finally:
        clear_policy_route()
        _stop_sglang_server(sglang_server)
        terminate_process(vllm_handle.process if vllm_handle else None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run aggregate rubric-selection SWE-agent on one or more SWE-bench instances.")
    parser.add_argument("--backend", choices=["vllm", "openai", "sglang"], default="vllm")
    parser.add_argument("--instance-id", action=ParseInstanceIds, nargs="+", default=None)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_AGGREGATE_OUTPUT_ROOT)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--source-rollout-root", type=Path, default=None)
    parser.add_argument("--source-run-pattern", default="*-r??")
    parser.add_argument("--step-limit", "--max-steps", dest="step_limit", type=int, default=DEFAULT_STEP_LIMIT)
    parser.add_argument("--environment-timeout", type=int, default=DEFAULT_ENV_TIMEOUT)
    parser.add_argument("--pull-timeout", type=int, default=DEFAULT_PULL_TIMEOUT)
    parser.add_argument("--gpu-id", default="auto:2")
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--allow-long-max-model-len", action="store_true")
    parser.add_argument("--vllm-model", default=DEFAULT_SERVE_MODEL)
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--sglang-model", default=DEFAULT_SERVE_MODEL)
    parser.add_argument("--sglang-api-base", default=SLIME_API_BASE)
    parser.add_argument("--sglang-api-key", default=SLIME_API_KEY)
    parser.add_argument("--sglang-image", default=os.environ.get("SEARCH_SWE_SLIME_IMAGE", DEFAULT_SGLANG_IMAGE))
    parser.add_argument("--use-existing-sglang-server", action="store_true")
    parser.add_argument("--server-timeout", type=int, default=600)
    parser.add_argument("--completion-max-tokens", type=int, default=DEFAULT_COMPLETION_MAX_TOKENS)
    parser.add_argument("--num-trajectories", "--m", dest="num_trajectories", type=int, default=4)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--policy-top-p", type=float, default=0.95)
    parser.add_argument("--rubric-temperature", type=float, default=0.0)
    parser.add_argument("--rubric-top-p", type=float, default=1.0)
    parser.add_argument("--rubric-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-top-p", type=float, default=1.0)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--rubric-model", default=None)
    parser.add_argument("--judge-model", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_aggregate(args, args.instance_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
