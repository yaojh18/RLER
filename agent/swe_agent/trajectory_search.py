from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import random
import re
import subprocess
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import pvariance
from typing import Any, Literal
import numpy as np

from agent_rl.run_utils import (
    extract_json_from_response,
    run_chat_with_route_async,
    run_chat_with_route_completion_async,
)

from agent_rl import RolloutSessionSpec, RolloutSnapshot
from swe_agent.backend import SWEAgentRolloutBackend
from swe_agent.parallel_utils import (
    ArtifactWriter,
    GRPOCollector,
    NodeArtifactBundle,
    PatchEvalManager,
    RubricArtifactBundle,
    _atomic_write_json,
    gap_redundancy
)
from swe_agent.run.run_swe_agent import build_messages, evaluate_swebench_instance_patches
from swe_agent.prompt import *


RETURN_CODE_RE = re.compile(r"<returncode>(.*?)</returncode>", re.DOTALL)
EXCEPTION_RE = re.compile(r"<exception>(.*?)</exception>", re.DOTALL)
OUTPUT_RE = re.compile(r"<output>\s*(.*?)</output>", re.DOTALL)
PR_DESCRIPTION_RE = re.compile(r"<pr_description>\s*(.*?)\s*</pr_description>", re.DOTALL)
OBSERVATION_TRUNCATION_MARKER = "\n[... Observation truncated due to length ...]\n"
MAX_OBSERVATION_CHARS = 1024
MIN_OBSERVATION_SECTION_CHARS = 256
EVALUATOR_MAX_RETRIES = 4
RUBRIC_LIST_FORMAT_MAX_RETRIES = 4
JUDGE_ERROR_REWARD = -0.2


@dataclass
class SearchConfig:
    m: int = 2
    n: int = 1
    k: int = 20
    p: int = 1
    max_rounds: int = 5
    step_limit: int = 100
    max_active_rubrics: int = 6
    policy_temperature: float = 1.0
    policy_top_p: float = 1.0
    rubric_temperature: float = 0.0
    rubric_top_p: float = 1.0
    rubric_max_tokens: int = 1024
    judge_temperature: float = 0.0
    judge_top_p: float = 1.0
    judge_max_tokens: int = 1024
    regression_margin: float = 0.0
    calculate_gt_reward: bool = False
    gt_reward_workers: int = 1
    evaluate_final_patch: bool = True
    export_grpo_bundles: bool = True
    write_artifacts: bool = True
    strategy: Literal["best", "probability", "random"] = "best"


@dataclass
class RubricRecord:
    rubric_id: str
    title: str
    direction: Literal["positive", "negative"]
    description: str
    scale: dict[str, str]
    weight: int
    source_round: int
    reward: float | None = None


@dataclass
class RubricGenerationSample:
    sample_index: int
    rubric_list_id: str
    generated: list[RubricRecord]
    messages: list[dict[str, Any]]
    format_errors: list[dict[str, Any]] | None = None


@dataclass
class SearchNode:
    node_id: str
    parent_id: str | None
    round_index: int
    depth: int
    session_id: str
    status: str
    step_start: int = 0
    step_end: int = 0
    submission: str = ""
    exit_status: str = ""
    policy_source: Literal["teacher", "student"] = "teacher"
    policy_model_name: str = ""


def _build_evaluation_payload(
    *,
    instance_id: str,
    completed: bool,
    resolved: bool,
    empty_patch: bool,
    error: bool,
) -> dict[str, Any]:
    return {
        "completed_ids": [instance_id] if completed else [],
        "incomplete_ids": [],
        "empty_patch_ids": [instance_id] if empty_patch else [],
        "submitted_ids": [instance_id],
        "resolved_ids": [instance_id] if resolved else [],
        "unresolved_ids": [instance_id] if completed and not resolved else [],
        "error_ids": [instance_id] if error else [],
        "schema_version": 2,
    }


def _truncate_middle(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(OBSERVATION_TRUNCATION_MARKER):
        return text[:limit]
    remaining = limit - len(OBSERVATION_TRUNCATION_MARKER)
    head = max(1, remaining // 2)
    tail = max(1, remaining - head)
    if head + tail >= len(text):
        return text
    return text[:head] + OBSERVATION_TRUNCATION_MARKER + text[-tail:]


def _render_structured_observation(
    *,
    exception: str | None,
    returncode: str | None,
    output: str | None,
) -> str:
    parts: list[str] = []
    if exception is not None:
        parts.append(f"<exception>{exception}</exception>")
    if returncode is not None:
        parts.append(f"<returncode>{returncode}</returncode>")
    if output is not None:
        parts.append(f"<output>\n{output}</output>")
    return "\n".join(parts)


def _truncate_structured_observation(observation: str) -> str:
    if len(observation) <= MAX_OBSERVATION_CHARS:
        return observation
    exception_match = EXCEPTION_RE.search(observation or "")
    returncode_match = RETURN_CODE_RE.search(observation or "")
    output_match = OUTPUT_RE.search(observation or "")
    if returncode_match is None and output_match is None and exception_match is None:
        return _truncate_middle(observation, MAX_OBSERVATION_CHARS)

    exception = exception_match.group(1) if exception_match is not None else None
    returncode = returncode_match.group(1).strip() if returncode_match is not None else None
    output = output_match.group(1) if output_match is not None else None
    sections = {}
    if exception is not None:
        sections["exception"] = exception
    if output is not None:
        sections["output"] = output

    base_length = len(
        _render_structured_observation(
            exception="" if exception is not None else None,
            returncode=returncode,
            output="" if output is not None else None,
        )
    )

    fixed_sections = {
        name: content for name, content in sections.items() if len(content) <= MIN_OBSERVATION_SECTION_CHARS
    }
    truncatable_sections = {
        name: content for name, content in sections.items() if len(content) > MIN_OBSERVATION_SECTION_CHARS
    }
    fixed_budget = sum(len(content) for content in fixed_sections.values())
    content_budget = MAX_OBSERVATION_CHARS - base_length - fixed_budget

    section_budgets = {name: len(content) for name, content in fixed_sections.items()}
    truncatable_lengths = {name: len(content) for name, content in truncatable_sections.items()}
    if truncatable_lengths and content_budget > 0:
        total_truncatable_length = sum(truncatable_lengths.values())
        shrink_ratio = content_budget / total_truncatable_length
        for name, length in truncatable_lengths.items():
            section_budgets[name] = min(length, max(int(length * shrink_ratio), MIN_OBSERVATION_SECTION_CHARS))

    if exception is not None:
        exception = _truncate_middle(exception, section_budgets.get("exception", len(exception)))
    if output is not None:
        output = _truncate_middle(output, section_budgets.get("output", len(output)))

    truncated = _render_structured_observation(exception=exception, returncode=returncode, output=output)
    return truncated


def _build_step_cards(segment_events: list[dict[str, Any]], step_start: int) -> list[dict[str, Any]]:
    cards: dict[int, dict[str, Any]] = {}
    step_index = step_start
    for event in segment_events:
        card = cards.setdefault(
            step_index,
            {
                "step_index": step_index,
                "assistant_message": "",
                "commands": [],
                "observation": "",
            },
        )
        if event.get("kind") == "model_response":
            message = event.get("payload", {}).get("message", {})
            assistant_message = message.get("content_no_thinking", message.get("content", ""))
            if assistant_message:
                card["assistant_message"] = _truncate_middle(assistant_message, MAX_OBSERVATION_CHARS)
            continue
        if event.get("kind") == "environment_action":
            commands = [action.get("command", "") for action in (event.get("payload") or {}).get("actions", [])]
            card["commands"] = commands
            continue
        if event.get("kind") not in {"environment_result", "agent_interrupt"}:
            continue
        messages = (event.get("payload") or {}).get("messages", [])
        if not messages:
            continue
        observation = messages[0].get("content", "")
        card["observation"] = _truncate_structured_observation(observation or "")
        step_index += 1
    return [cards[index] for index in sorted(cards)]


def _collect_workspace_meta(environment: Any) -> dict[str, Any]:
    cwd = getattr(getattr(environment, "config", None), "cwd", "") or "/testbed"
    command = f"""python - <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess

cwd = {json.dumps(cwd)}
os.chdir(cwd)

def run(cmd):
    completed = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()

payload = {{
    "cwd": cwd,
    "git_repo": False,
    "head_commit": "",
    "changed_files": [],
    "untracked_files": [],
    "status": [],
    "diff_stat": "",
    "current_patch_chars": 0,
    "workspace_fingerprint": None,
}}
git_ok, _, _ = run("git rev-parse --is-inside-work-tree")
payload["git_repo"] = git_ok == 0

if payload["git_repo"]:
    _, head_commit, _ = run("git rev-parse HEAD")
    _, changed_files, _ = run("git diff --name-only --diff-filter=ACMRTUXB")
    _, untracked_files, _ = run("git ls-files --others --exclude-standard")
    _, status, _ = run("git status --porcelain=v1")
    _, diff_stat, _ = run("git diff --stat --compact-summary")
    _, full_diff, _ = run("git diff --no-ext-diff")
    changed = [line for line in changed_files.splitlines() if line.strip()]
    untracked = [line for line in untracked_files.splitlines() if line.strip()]
    status_lines = [line for line in status.splitlines() if line.strip()]
    payload["head_commit"] = head_commit
    payload["changed_files"] = changed
    payload["untracked_files"] = untracked
    payload["status"] = status_lines
    payload["diff_stat"] = diff_stat
    payload["current_patch_chars"] = len(full_diff)
    fingerprints = {{}}
    for rel_path in changed + untracked:
        path = pathlib.Path(rel_path)
        if path.is_file():
            try:
                fingerprints[rel_path] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                pass
    payload["workspace_fingerprint"] = hashlib.sha256(
        json.dumps(
            {{
                "head_commit": head_commit,
                "changed_files": changed,
                "untracked_files": untracked,
                "fingerprints": fingerprints,
            }},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

print(json.dumps(payload))
PY"""
    result = environment.execute({"command": command})
    if result.get("returncode") != 0 or not result.get("output", "").strip():
        return {**EMPTY_WORKSPACE_META, "cwd": cwd}
    parsed = extract_json_from_response(result["output"])
    if not isinstance(parsed, dict):
        return {**EMPTY_WORKSPACE_META, "cwd": cwd}
    workspace_meta = {**EMPTY_WORKSPACE_META, **parsed}
    workspace_meta["cwd"] = cwd
    workspace_meta["changed_files"] = list(workspace_meta.get("changed_files", []))
    workspace_meta["untracked_files"] = list(workspace_meta.get("untracked_files", []))
    workspace_meta["status"] = list(workspace_meta.get("status", []))
    workspace_meta["current_patch_chars"] = int(workspace_meta.get("current_patch_chars", 0) or 0)
    return workspace_meta


async def _update_persistent_state(
    *,
    system_prompt: str,
    user_prompt: str,
    previous_state: dict[str, Any],
    evicted_step_cards: list[dict[str, Any]] | None,
    workspace_meta: dict[str, Any],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if evicted_step_cards is None:
        return copy.deepcopy(previous_state)
    prompt = "\n\n".join(
        [
            PERSISTENT_STATE_UPDATE_PROMPT.strip(),
            f"\n\n## Question:\n System Prompt:\n{system_prompt}\n User Prompt:\n{user_prompt}",
            f"## Previous Persistent State:\n{json.dumps(previous_state, indent=2, ensure_ascii=False)}",
            f"## Evicted Older Trajectory:\n{json.dumps(evicted_step_cards, indent=2, ensure_ascii=False)}",
            f"## Workspace Metadata:\n{json.dumps(workspace_meta, indent=2, ensure_ascii=False)}",
        ]
    )
    for _ in range(EVALUATOR_MAX_RETRIES):
        response = await run_chat_with_route_async(
            "rubric_judge",
            model_name=model_name,
            user_prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format=copy.deepcopy(PERSISTENT_STATE_RESPONSE_FORMAT),
            enable_json_schema_validation=True,
            **(model_kwargs or {}),
        )
        parsed = extract_json_from_response(response)
        if isinstance(parsed, dict):
            state = copy.deepcopy(previous_state)
            for key in state:
                if isinstance(parsed.get(key), str):
                    state[key] = parsed[key]
            return state
    return copy.deepcopy(previous_state)


def _convert_generated_rubric(task: str, payload: dict[str, Any], round_index: int) -> RubricRecord | None:
    item = payload.get("rubric", None)
    if item is None or not isinstance(item, dict):
        return None
    direction = item.get("polarity", None)
    if direction not in {"positive", "negative"}:
        return None
    title = item.get("title", None)
    if title is None or not isinstance(title, str):
        return None
    description = item.get("description", None)
    if description is None or not isinstance(description, str):
        return None
    scale = {str(score): text for score, text in (item.get("scale") or {}).items()}
    if set(scale) != {"1", "2", "3", "4", "5"}:
        return None
    scale_text = {isinstance(text, str) for _, text in (item.get("scale") or {}).items()}
    if not scale_text or not all(scale_text):
        return None
    rubric_id = hashlib.md5(
        json.dumps(
            {
                "task": task,
                "direction": direction,
                "title": title,
                "description": description,
                "scale": scale,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return RubricRecord(
        rubric_id=rubric_id,
        title=title,
        direction=direction,
        description=description,
        scale=scale,
        weight=1 if direction == "positive" else -1,
        source_round=round_index,
    )


def _build_initial_rubric_bank(task: str) -> list[RubricRecord]:
    bank = [
        _convert_generated_rubric(
            task,
            {
                "rubric": {
                    "polarity": "positive",
                    "title": "Evidence-to-Decision Traceability",
                    "description": "Ties major next steps or fix proposals to concrete evidence already surfaced in the trajectory—such as an observed failure, command/check output, file or function inspection, a named code location, or a test result—and makes clear how that evidence changes the plan.",
                    "scale": {
                        "1": "Major decisions are not tied to any concrete evidence already surfaced in the trajectory",
                        "2": "Mentions concrete evidence, but the link from evidence to the chosen next step is mostly implicit or weak",
                        "3": "At least one important next step is explicitly justified by concrete evidence, but other key decisions remain weakly supported",
                        "4": "Most important next steps or fix proposals are explicitly tied to concrete evidence and the effect on the plan is clear",
                        "5": "Nearly every important pivot, hypothesis update, or fix proposal is explicitly anchored to concrete evidence, with a clear explanation of how that evidence drives the next move",
                    },
                }
            },
            0,
        ),
        _convert_generated_rubric(
            task,
            {
                "rubric": {
                    "polarity": "negative",
                    "title": "Closure Without a Decisive Check",
                    "description": "Claims or strongly implies that the issue is fixed, understood, or ready to close without naming a concrete check or test that would decide the claim, or despite already available evidence that leaves the claim unresolved.",
                    "scale": {
                        "1": "No closure claim is made without a concrete deciding check",
                        "2": "Slight overconfidence, but the continuation stays tentative or names a plausible confirming check",
                        "3": "Makes a noticeable completion or success claim while the decisive confirming check is missing or underspecified",
                        "4": "Treats the issue as effectively resolved without a concrete deciding check or despite unresolved contrary evidence",
                        "5": "Strongly declares success or completion and proceeds as if resolved, with no concrete deciding check and no serious engagement with unresolved evidence",
                    },
                }
            },
            0,
        ),
    ]
    return [rubric for rubric in bank if rubric is not None]


def _parse_judge_score(response: str) -> int | None:
    parsed = extract_json_from_response(response)
    if not isinstance(parsed, dict):
        return None
    try:
        score = int(parsed.get("score"))
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


async def _generate_round_rubrics(
    *,
    question: dict[str, str],
    previous_state: dict[str, Any],
    latest_shared_segment: dict[str, Any] | None,
    continuations: list[dict[str, Any]],
    active_bank: list[RubricRecord],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    round_index: int,
    sample_index: int = 0,
    model_kwargs: dict[str, Any] | None = None,
) -> RubricGenerationSample:
    latest_shared_segment_text = json.dumps(latest_shared_segment, indent=2, ensure_ascii=False) if latest_shared_segment else "None"
    prompt_parts = [
        SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT.strip(),
        "\n\n## Question:",
        f"System Prompt:\n{question.get('system_prompt', '')}",
        "",
        f"User Prompt:\n{question.get('user_prompt', '')}",
        "",
        "## Previous Persistent State:",
        json.dumps(previous_state, ensure_ascii=False, indent=2),
        "",
        "## Latest Agent Trajectory:",
        latest_shared_segment_text,
        "",
        "## Agent Trajectory Continuations:"
    ]
    for index, continuation in enumerate(continuations, start=1):
        prompt_parts.extend(
            [
                f"## Continuation {index}:",
                json.dumps(
                    {
                        "summary": continuation.get("summary", {}),
                        "trajectory_continuation": continuation.get("trajectory_continuation"),
                        **({"note": continuation["note"]} if continuation.get("note") else {}),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "",
            ]
        )
    if active_bank:
        prompt_parts.extend(
            [
                "## Existing Rubrics:",
                json.dumps(
                    [{"polarity": rubric.direction, "title": rubric.title, "description": rubric.description, "scale": rubric.scale} for rubric in active_bank],
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )
    prompt = "\n".join(prompt_parts)
    conversation_messages = [{"role": "user", "content": prompt}]
    task_text = "\n\n".join(part for part in [question.get("system_prompt", ""), question.get("user_prompt", "")] if part)
    generated: list[RubricRecord] = []
    format_errors: list[dict[str, Any]] = []
    remaining_budget = 6
    for idx in range(remaining_budget):
        parsed_candidate: dict[str, Any] | None = None
        parsed: dict[str, Any] | None = None
        assistant_content = ""
        assistant_content_no_thinking = ""
        last_error = ""
        for _ in range(EVALUATOR_MAX_RETRIES):
            try:
                completion = await run_chat_with_route_completion_async(
                    "rubric_generation",
                    model_name=model_name,
                    messages=conversation_messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    response_format=copy.deepcopy(RUBRIC_GENERATION_RESPONSE_FORMAT),
                    enable_json_schema_validation=True,
                    **(model_kwargs or {}),
                )
                assistant_content = completion.content or ""
                assistant_content_no_thinking = completion.metadata.get("content_no_thinking", assistant_content)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            parsed_candidate = extract_json_from_response(assistant_content or "")
            if parsed_candidate == {}:
                break
            if isinstance(parsed_candidate, dict):
                parsed = _convert_generated_rubric(task_text, parsed_candidate, round_index)
                break
        conversation_messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "content_no_thinking": assistant_content_no_thinking,
            }
        )
        if parsed_candidate == {}:
            break
        if parsed is None:
            format_errors.append({"turn_index": len(generated) + 1, "error": last_error})
            break
        generated.append(parsed)
        if idx < remaining_budget - 1:
            conversation_messages.append({"role": "user", "content": RUBRIC_GENERATION_CONTINUE_PROMPT})
    return RubricGenerationSample(
        sample_index=sample_index,
        rubric_list_id=f"rubric-r{round_index:03d}-s{sample_index:02d}",
        generated=generated,
        messages=copy.deepcopy(conversation_messages),
        format_errors=format_errors,
    )


async def _score_round(
    *,
    question: dict[str, str],
    shared_context: dict[str, Any],
    continuations: list[dict[str, Any]],
    rubrics: list[RubricRecord],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, str]]]:
    if not continuations or not rubrics:
        return [[] for _ in continuations], []
    calls = []
    mapping: list[tuple[int, str, RubricRecord]] = []
    question_text = f"System Prompt:\n{question.get('system_prompt', '')}\n\nUser Prompt:\n{question.get('user_prompt', '')}"
    for view_index, continuation in enumerate(continuations):
        node_id = continuation.get("node_id")
        response_text = json.dumps(
            {
                "summary": continuation.get("summary", {}),
                "trajectory_continuation": continuation.get("trajectory_continuation"),
                **({"note": continuation["note"]} if continuation.get("note") else {}),
            },
            indent=2,
            ensure_ascii=False,
        )
        for rubric in rubrics:
            criterion = "\n".join(
                [
                    f"Title: {rubric.title}",
                    f"Type: {rubric.direction}",
                    f"Description: {rubric.description}",
                    "Scale:",
                    *[f"{score}: {rubric.scale[str(score)]}" for score in range(1, 6)],
                ]
            )

            async def _judge_single(
                *,
                response_text: str = response_text,
                criterion: str = criterion,
            ) -> tuple[str, int, str | None]:
                last_error = None
                for _ in range(EVALUATOR_MAX_RETRIES):
                    try:
                        response = await run_chat_with_route_async(
                            "rubric_judge",
                            model_name=model_name,
                            user_prompt=
                                SWE_TRAJECTORY_RUBRIC_JUDGE_PROMPT.strip() + (
                                f"\n\n## Question:\n{question_text}\n"
                                f"## Previous Persistent State:\n{shared_context.get('previous_persistent_state')}\n"
                                f"## Previous Trajectory:\n{shared_context.get('latest_agent_trajectory')}\n"
                                f"## Continuation Trajectory:\n{response_text}\n"
                                f"## Criterion:\n{criterion}"
                            ),
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            response_format=copy.deepcopy(JUDGE_RESPONSE_FORMAT),
                            enable_json_schema_validation=True,
                            **(model_kwargs or {}),
                        )
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        continue
                    score_raw = _parse_judge_score(response)
                    if score_raw is not None:
                        return response, score_raw, None
                    last_error = "InvalidJudgeResponse"
                return json.dumps({"score": 1}, ensure_ascii=False), 1, last_error
            calls.append(_judge_single())
            mapping.append((view_index, node_id, rubric))
    responses = await asyncio.gather(*calls)
    per_view_scores: list[list[dict[str, Any]]] = [[] for _ in continuations]
    errors: list[dict[str, str]] = []
    for (view_index, node_id, rubric), response in zip(mapping, responses):
        judge_response, score_raw, error = response
        normalized = max(0.0, min(1.0, (score_raw - 1.0) / 4.0))
        record = {
            "rubric_id": rubric.rubric_id,
            "rubric": {
                "rubric_id": rubric.rubric_id,
                "title": rubric.title,
                "direction": rubric.direction,
                "description": rubric.description,
                "scale": copy.deepcopy(rubric.scale),
                "weight": rubric.weight,
                "source_round": rubric.source_round,
            },
            "score_raw": score_raw,
            "score_normalized": normalized,
            "weighted_score": float(rubric.weight) * normalized,
            "judge_response": judge_response,
        }
        per_view_scores[view_index].append(record)
        if error is not None:
            errors.append(
                {
                    "rubric_id": rubric.rubric_id,
                    "node_id": node_id,
                    "error": error,
                }
            )
    return per_view_scores, errors


async def _score_parent_round(
    *,
    question: dict[str, str],
    shared_context: dict[str, Any],
    rubrics: list[RubricRecord],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    node_id: str,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not rubrics:
        return [], []
    calls = []
    mapping: list[RubricRecord] = []
    question_text = f"System Prompt:\n{question.get('system_prompt', '')}\n\nUser Prompt:\n{question.get('user_prompt', '')}"
    for rubric in rubrics:
        criterion = "\n".join(
            [
                f"Title: {rubric.title}",
                f"Type: {rubric.direction}",
                f"Description: {rubric.description}",
                "Scale:",
                *[f"{score}: {rubric.scale[str(score)]}" for score in range(1, 6)],
            ]
        )

        async def _judge_single(
            *,
            criterion: str = criterion,
        ) -> tuple[str, int, str | None]:
            last_error = None
            for _ in range(EVALUATOR_MAX_RETRIES):
                try:
                    response = await run_chat_with_route_async(
                        "rubric_judge",
                        model_name=model_name,
                        user_prompt=
                            SWE_TRAJECTORY_RUBRIC_JUDGE_PARENT_PROMPT.strip() + (
                            f"\n\n## Question:\n{question_text}\n"
                            f"## Previous Persistent State:\n{shared_context.get('previous_persistent_state')}\n"
                            f"## Agent Trajectory:\n{shared_context.get('latest_agent_trajectory')}\n"
                            f"## Criterion:\n{criterion}"
                        ),
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        response_format=copy.deepcopy(JUDGE_RESPONSE_FORMAT),
                        enable_json_schema_validation=True,
                        **(model_kwargs or {}),
                    )
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    continue
                score_raw = _parse_judge_score(response)
                if score_raw is not None:
                    return response, score_raw, None
                last_error = "InvalidJudgeResponse"
            return json.dumps({"score": 1}, ensure_ascii=False), 1, last_error
        calls.append(_judge_single())
        mapping.append(rubric)
    responses = await asyncio.gather(*calls)
    score_records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for rubric, response in zip(mapping, responses):
        judge_response, score_raw, error = response
        normalized = max(0.0, min(1.0, (score_raw - 1.0) / 4.0))
        record = {
            "rubric_id": rubric.rubric_id,
            "rubric": {
                "rubric_id": rubric.rubric_id,
                "title": rubric.title,
                "direction": rubric.direction,
                "description": rubric.description,
                "scale": copy.deepcopy(rubric.scale),
                "weight": rubric.weight,
                "source_round": rubric.source_round,
            },
            "score_raw": score_raw,
            "score_normalized": normalized,
            "weighted_score": float(rubric.weight) * normalized,
            "judge_response": judge_response,
        }
        score_records.append(record)
        if error is not None:
            errors.append(
                {
                    "node_id": node_id,
                    "rubric_id": rubric.rubric_id,
                    "error": error,
                }
            )
    return score_records, errors


# NOTE: the correlation score will be unstable when there are very few data points
def _redundancy_reward(
    candidate_scores: list[float],
    existing_score_vectors: list[list[float]],
) -> float:
    if not existing_score_vectors:
        return 1.0
    candidate = np.asarray(candidate_scores, dtype=float)
    max_redundency = max(gap_redundancy(candidate, other_scores) for other_scores in existing_score_vectors)
    return float(1.0 - max_redundency)


def _sample_by_strategy(
    items: list[Any],
    scores: list[float],
    *,
    count: int,
    strategy: str,
) -> list[Any]:
    if not items or count <= 0:
        return []
    capped = min(count, len(items))
    if strategy == "random":
        return random.sample(items, capped)
    if strategy == "best":
        return [item for item, _ in sorted(zip(items, scores), key=lambda x: x[1], reverse=True)[:capped]]
    weights = np.asarray(scores, dtype=float)
    min_score = weights.min()
    max_score = weights.max()
    if max_score > min_score:
        weights = (weights - min_score) / (max_score - min_score)
    else:
        return random.sample(items, capped)
    probs = weights / weights.sum()
    indices = np.random.choice(len(items), size=capped, replace=False, p=probs)
    return [items[i] for i in indices]


def _update_rubric_bank(
    *,
    active_bank: list[RubricRecord],
    inactive_bank: list[RubricRecord],
    generated: list[RubricRecord],
    rewards: dict[str, float],
    max_active_rubrics: int,
) -> tuple[list[RubricRecord], list[RubricRecord], list[RubricRecord]]:
    deduped_by_title: dict[str, RubricRecord] = {}
    for rubric in active_bank + generated:
        candidate = RubricRecord(**{**asdict(rubric), "reward": rewards.get(rubric.rubric_id, 0.0)})
        title_key = candidate.title.strip().casefold()
        existing = deduped_by_title.get(title_key)
        if existing is None or (candidate.reward or 0.0) > (existing.reward or 0.0) or (
            (candidate.reward or 0.0) == (existing.reward or 0.0) and candidate.source_round > existing.source_round
        ):
            deduped_by_title[title_key] = candidate
    ranked = list(deduped_by_title.values())
    ranked.sort(key=lambda rubric: (rubric.reward or 0.0), reverse=True)
    active = [rubric for rubric in ranked][:max_active_rubrics]
    active_ids = {rubric.rubric_id for rubric in active}
    inactive = [rubric for rubric in ranked + inactive_bank if rubric.rubric_id not in active_ids]
    return active, inactive, ranked


def _docker_commit(executable: str, container_id: str, image_tag: str) -> tuple[str, str]:
    subprocess.run([executable, "commit", container_id, image_tag], check=True, capture_output=True, text=True)
    image_id = subprocess.run(
        [executable, "image", "inspect", image_tag, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return image_tag, image_id


class TrajectorySearchRunner:
    def __init__(
        self,
        *,
        instance: dict[str, Any],
        backend: SWEAgentRolloutBackend,
        run_dir: Path,
        policy_model_name: str,
        student_policy_model_name: str | None = None,
        search_config: SearchConfig | None = None,
        rubric_model_name: str | None = None,
        judge_model_name: str | None = None,
        rubric_model_kwargs: dict[str, Any] | None = None,
        judge_model_kwargs: dict[str, Any] | None = None,
        harness_namespace: str | None = None,
        resume: bool = False,
    ) -> None:
        self.instance = copy.deepcopy(instance)
        self.backend = backend
        self.run_dir = Path(run_dir)
        self.nodes_dir = self.run_dir / "nodes"
        self.rubrics_dir = self.run_dir / "rubrics"
        self.policy_model_name = policy_model_name
        self.student_policy_model_name = student_policy_model_name
        self.rubric_model_name = rubric_model_name or policy_model_name
        self.judge_model_name = judge_model_name or policy_model_name
        self.search_config = search_config or SearchConfig()
        self.rubric_model_kwargs = copy.deepcopy(rubric_model_kwargs or {})
        self.judge_model_kwargs = copy.deepcopy(judge_model_kwargs or {})
        self.harness_namespace = harness_namespace
        self.resume = resume
        self.task = instance["problem_statement"]
        self.task_id = instance["instance_id"]
        self.run_id = f"{self.task_id}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.base_image = str(self.backend.environment_config.get("image", ""))
        self.docker_executable = str(self.backend.environment_config.get("executable", "docker"))
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.node_index_path = self.run_dir / "node_index.jsonl"
        base_image_lookup = subprocess.run(
            [self.docker_executable, "image", "inspect", self.base_image, "--format", "{{.Id}}"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.base_image_id = base_image_lookup.stdout.strip() or None if base_image_lookup.returncode == 0 else None
        self.nodes: dict[str, SearchNode] = {}
        self.frontier_ids: list[str] = []
        self.active_bank: list[RubricRecord] = []
        self.inactive_bank: list[RubricRecord] = []
        self.current_round = 0
        self.system_prompt = ""
        self.user_prompt = ""
        self.best_node_id: str | None = None
        self.finished_node_ids: list[str] = []
        self.image_repository = f"rler-search/{self.task_id.replace('__', '-').lower()}"
        self.artifact_writer = ArtifactWriter(write_artifacts=self.search_config.write_artifacts)
        self._manifest_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="search-manifest")
        self._manifest_futures: list[Future] = []
        self._node_judge_cache: dict[str, dict[str, Any]] = {}
        self._node_snapshot_cache: dict[str, dict[str, Any]] = {}
        self.grpo_collector = (
            GRPOCollector(instance_id=self.task_id, run_dir=self.run_dir)
            if self.search_config.export_grpo_bundles or not self.search_config.write_artifacts
            else None
        )
        self.patch_eval_manager = (
            PatchEvalManager(
                instance=self.instance,
                task_id=self.task_id,
                model_name=self.policy_model_name,
                namespace=self.harness_namespace,
                work_dir=self.run_dir,
                evaluate_patches_fn=evaluate_swebench_instance_patches,
                collector=self.grpo_collector,
                write_artifacts=self.search_config.write_artifacts,
                max_workers=max(1, self.search_config.gt_reward_workers),
            )
            if self.search_config.calculate_gt_reward
            else None
        )

    def run(self) -> None:
        if self.resume and not self.search_config.write_artifacts:
            raise RuntimeError("resume requires write_artifacts=True")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.resume and self.manifest_path.exists():
                self._load_manifest()
            else:
                self._initialize_root()            
            while self.current_round < self.search_config.max_rounds:
                if self.frontier_ids and self.nodes[self.frontier_ids[0]].status == "finished":
                    break
                if len(self.frontier_ids) == 0:
                    break
                self.current_round += 1
                self._run_round(self.frontier_ids[0], self.current_round)
                self._save_manifest()
            self.artifact_writer.wait()
            if self.patch_eval_manager is not None:
                self.patch_eval_manager.wait()
            self._finalize_outputs()
        finally:
            if self.patch_eval_manager is not None:
                self.patch_eval_manager.close()
            self.artifact_writer.close()
            self._sweep_checkpoint_images()
            if self.nodes:
                self._save_manifest()
            while self._manifest_futures:
                self._manifest_futures.pop(0).result()
            self._manifest_executor.shutdown(wait=True, cancel_futures=False)

    def _save_manifest(self) -> None:
        if not self.search_config.write_artifacts:
            return
        manifest_payload = {
            "run_id": self.run_id,
            "instance_id": self.task_id,
            "search_config": asdict(self.search_config),
            "base_image": self.base_image,
            "base_image_id": self.base_image_id,
            "policy_model_name": self.policy_model_name,
            "student_policy_model_name": self.student_policy_model_name,
            "rubric_model_name": self.rubric_model_name,
            "judge_model_name": self.judge_model_name,
            "frontier_ids": list(self.frontier_ids),
            "best_node_id": self.best_node_id,
            "finished_node_ids": list(self.finished_node_ids),
            "current_round": self.current_round,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "active_bank": [asdict(rubric) for rubric in self.active_bank],
            "inactive_bank": [asdict(rubric) for rubric in self.inactive_bank],
            "node_ids": sorted(self.nodes),
        }
        node_index_text = "\n".join(
            json.dumps(asdict(self.nodes[node_id]), ensure_ascii=False, default=str) for node_id in sorted(self.nodes)
        )
        if node_index_text:
            node_index_text += "\n"
        self._manifest_futures.append(
            self._manifest_executor.submit(
                self._write_manifest_files,
                manifest_payload,
                node_index_text,
            )
        )

    def _write_manifest_files(self, manifest_payload: dict[str, Any], node_index_text: str) -> None:
        _atomic_write_json(self.manifest_path, manifest_payload)
        self.node_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.node_index_path.write_text(node_index_text, encoding="utf-8")

    def _load_manifest(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.run_id = manifest.get("run_id", self.run_id)
        self.frontier_ids = list(manifest.get("frontier_ids", []))
        self.best_node_id = manifest.get("best_node_id")
        self.finished_node_ids = list(manifest.get("finished_node_ids", []))
        self.current_round = int(manifest.get("current_round", 0))
        self.system_prompt = manifest.get("system_prompt", "")
        self.user_prompt = manifest.get("user_prompt", "")
        self.active_bank = [RubricRecord(**rubric) for rubric in manifest.get("active_bank", [])]
        self.inactive_bank = [RubricRecord(**rubric) for rubric in manifest.get("inactive_bank", [])]
        self.nodes = {}
        for node_id in manifest.get("node_ids", []):
            node_path = self.nodes_dir / node_id / "node.json"
            if node_path.exists():
                self.nodes[node_id] = SearchNode(**json.loads(node_path.read_text(encoding="utf-8")))
                judge_path = node_path.parent / "judge.json"
                snapshot_path = node_path.parent / "snapshot.json"
                if judge_path.exists():
                    self._node_judge_cache[node_id] = json.loads(judge_path.read_text(encoding="utf-8"))
                if snapshot_path.exists():
                    self._node_snapshot_cache[node_id] = json.loads(snapshot_path.read_text(encoding="utf-8"))

    def _dispose_session(self, session: Any) -> None:
        env = getattr(session.agent, "env", None)
        container_id = getattr(env, "container_id", None)
        if container_id:
            subprocess.run([self.docker_executable, "rm", "-f", container_id], check=False, capture_output=True, text=True)
        if env is not None:
            setattr(env, "container_id", None)
            setattr(env, "_owns_container", False)

    def _sweep_checkpoint_images(self) -> None:
        keep_tags = {
            image_tag
            for node_id in self.frontier_ids
            if node_id in self.nodes
            for image_tag in [((self._node_snapshot_cache.get(node_id) or {}).get("metadata", {}) or {}).get("checkpoint_image_tag")]
            if image_tag and image_tag != self.base_image
        }
        listed = subprocess.run(
            [self.docker_executable, "image", "ls", self.image_repository, "--format", "{{.Repository}}:{{.Tag}}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if listed.returncode != 0:
            return
        removable_tags = sorted({line.strip() for line in listed.stdout.splitlines() if line.strip()} - keep_tags)
        if not removable_tags:
            return
        with ThreadPoolExecutor(
            max_workers=min(4, len(removable_tags)),
            thread_name_prefix="checkpoint-image-sweep",
        ) as executor:
            futures = [
                executor.submit(
                    subprocess.run,
                    [self.docker_executable, "image", "rm", "-f", image_tag],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                for image_tag in removable_tags
            ]
            for future in futures:
                future.result()
    
    def _initialize_root(self) -> None:
        spec = RolloutSessionSpec(
            session_id=f"root-{uuid.uuid4().hex}",
            task=self.task,
            task_id=self.task_id,
            sample_index=0,
            policy_ref=self.policy_model_name,
            policy_version=self.policy_model_name,
            dataset_name="swebench",
            ground_truth=self.instance.get("patch"),
            raw_user_query=self.task,
            limits={"step_limit": self.backend.agent_config.get("step_limit", 0)},
            metadata={"template_vars": copy.deepcopy(self.instance)},
        )
        session = self.backend.create_session(spec)
        snapshot = session.snapshot().model_dump(mode="json")
        workspace_meta = _collect_workspace_meta(session.agent.env)
        messages = snapshot["agent"]["state"].get("messages", [])
        self.system_prompt = messages[0].get("content", "") if messages else ""
        self.user_prompt = messages[1].get("content", "") if len(messages) > 1 else self.task
        initial_task_text = "\n\n".join(part for part in [self.system_prompt, self.user_prompt] if part)
        self.active_bank = _build_initial_rubric_bank(initial_task_text or self.task)
        self.inactive_bank = []
        root_node = SearchNode(
            node_id="root",
            parent_id=None,
            round_index=0,
            depth=0,
            session_id=spec.session_id,
            status="frontier",
            step_start=0,
            step_end=0,
            policy_source="teacher",
            policy_model_name=self.policy_model_name,
        )
        root_judge = {
            "persistent_state": copy.deepcopy(EMPTY_PERSISTENT_STATE),
            "recent_segments": [],
            "workspace_meta": workspace_meta,
            "rubric_round": {
                "generated": [],
                "active_bank_before": [],
                "active_bank_after": [asdict(rubric) for rubric in self.active_bank],
                "inactive_bank_after": [],
                "variance_by_rubric": {},
            },
            "overall_reward": 0.0,
            "ground_truth_reward": 0.0,
            "baseline_parent_reward": None,
            "regressed_vs_parent": False,
        }
        root_snapshot = {
            **copy.deepcopy(snapshot),
            "environment": {
                **copy.deepcopy(snapshot["environment"]),
                "config": {
                    **copy.deepcopy(snapshot["environment"]["config"]),
                    "image": self.base_image,
                },
                "state": {"owns_container": True},
            },
            "metadata": {
                **copy.deepcopy(snapshot.get("metadata", {})),
                "checkpoint_image_id": self.base_image_id,
                "checkpoint_image_tag": self.base_image,
            },
        }
        self.nodes[root_node.node_id] = root_node
        self._node_judge_cache[root_node.node_id] = copy.deepcopy(root_judge)
        self._node_snapshot_cache[root_node.node_id] = copy.deepcopy(root_snapshot)
        root_bundle = NodeArtifactBundle(
            node_id=root_node.node_id,
            node_dir=self.nodes_dir / root_node.node_id,
            node_payload=asdict(root_node),
            messages_payload=build_messages([], model_name=self.policy_model_name),
            prompt_payload={"messages": copy.deepcopy(messages)},
            judge_payload=root_judge,
            snapshot_payload=root_snapshot,
        )
        self.artifact_writer.submit_round([root_bundle])
        self.frontier_ids = ["root"]
        self.best_node_id = None
        self.finished_node_ids = []
        self._save_manifest()
        self._dispose_session(session)

    async def _prepare_round_judging(
        self,
        *,
        parent_node: SearchNode,
        parent_judge: dict[str, Any],
        branch_records: list[dict[str, Any]],
        round_index: int,
        compare_parent: bool,
    ) -> dict[str, Any]:
        parent_state = parent_judge["persistent_state"]
        parent_recent_segments = copy.deepcopy(parent_judge.get("recent_segments", []))
        latest_shared_segment = copy.deepcopy(parent_recent_segments[-1]) if parent_recent_segments else None
        updated_parent_state = copy.deepcopy(parent_state)
        evicted_step_cards = None
        if len(parent_recent_segments) >= 2:
            evicted_step_cards = copy.deepcopy(parent_recent_segments[0].get("step_cards", []))
        if evicted_step_cards:
            evicted_workspace_meta = copy.deepcopy(EMPTY_WORKSPACE_META)
            if parent_node.parent_id:
                grandparent_judge = self._node_judge_cache.get(parent_node.parent_id)
                if grandparent_judge is not None:
                    evicted_workspace_meta = copy.deepcopy(grandparent_judge.get("workspace_meta", evicted_workspace_meta))
            try:
                updated_parent_state = await _update_persistent_state(
                    system_prompt=self.system_prompt,
                    user_prompt=self.task,
                    previous_state=parent_state,
                    evicted_step_cards=evicted_step_cards,
                    workspace_meta=evicted_workspace_meta,
                    model_name=self.rubric_model_name,
                    temperature=self.search_config.rubric_temperature,
                    top_p=self.search_config.rubric_top_p,
                    max_tokens=self.search_config.rubric_max_tokens,
                    model_kwargs=self.rubric_model_kwargs,
                )
            except Exception:
                updated_parent_state = copy.deepcopy(parent_state)
        question = {"system_prompt": self.system_prompt, "user_prompt": self.task}
        shared_context = {
            "previous_persistent_state": updated_parent_state,
            "latest_agent_trajectory": latest_shared_segment,
        }
        continuations = []
        for branch in branch_records:
            branch["persistent_state"] = copy.deepcopy(updated_parent_state)
            step_cards = branch["recent_segments"][-1].get("step_cards", []) if branch["recent_segments"] else []
            workspace_meta = branch.get("workspace_meta", {})
            result = branch.get("result", {})
            branch["continuation_view"] = {
                "node_id": branch["node_id"],
                "summary": {
                    "step_count": len(step_cards),
                    "changed_files": list(workspace_meta.get("changed_files", []))[:8],
                    "untracked_files": list(workspace_meta.get("untracked_files", []))[:8],
                    "diff_stat": workspace_meta.get("diff_stat", ""),
                    "current_patch_chars": int(workspace_meta.get("current_patch_chars", 0) or 0),
                    "result_status": result.get("status", ""),
                    "exit_status": result.get("exit_status", ""),
                },
                "trajectory_continuation": copy.deepcopy(branch["recent_segments"][-1]) if branch["recent_segments"] else None,
            }
            continuations.append(branch["continuation_view"])
        rubric_generation_kwargs = {
            "question": {**question, "instance_id": self.task_id},
            "previous_state": updated_parent_state,
            "latest_shared_segment": latest_shared_segment,
            "continuations": continuations,
            "active_bank": self.active_bank,
            "model_name": self.rubric_model_name,
            "temperature": self.search_config.rubric_temperature,
            "top_p": self.search_config.rubric_top_p,
            "max_tokens": self.search_config.rubric_max_tokens,
            "round_index": round_index,
            "model_kwargs": self.rubric_model_kwargs,
        }
        if self.search_config.n == 1:
            generated_sample = await _generate_round_rubrics(
                **rubric_generation_kwargs,
                sample_index=0,
            )
            for _ in range(RUBRIC_LIST_FORMAT_MAX_RETRIES):
                if not generated_sample.format_errors:
                    break
                generated_sample = await _generate_round_rubrics(
                    **rubric_generation_kwargs,
                    sample_index=0,
                )
            generated_samples = [generated_sample]
        else:
            generated_samples = await asyncio.gather(
                *[
                    _generate_round_rubrics(
                        **rubric_generation_kwargs,
                        sample_index=sample_index,
                    )
                    for sample_index in range(self.search_config.n)
                ],
            )
        rubric_samples: list[dict[str, Any]] = []
        valid_rubric_samples: list[dict[str, Any]] = []
        rubric_generation_errors: list[dict[str, Any]] = []
        active_continuation_scores, active_continuation_errors = await _score_round(
            question=question,
            shared_context=shared_context,
            continuations=continuations,
            rubrics=self.active_bank,
            model_name=self.judge_model_name,
            temperature=self.search_config.judge_temperature,
            top_p=self.search_config.judge_top_p,
            max_tokens=self.search_config.judge_max_tokens,
            model_kwargs=self.judge_model_kwargs,
        )
        active_parent_scores: list[dict[str, Any]] = []
        active_parent_errors: list[dict[str, str]] = []
        if compare_parent:
            active_parent_scores, active_parent_errors = await _score_parent_round(
                question=question,
                shared_context=shared_context,
                rubrics=self.active_bank,
                model_name=self.judge_model_name,
                temperature=self.search_config.judge_temperature,
                top_p=self.search_config.judge_top_p,
                max_tokens=self.search_config.judge_max_tokens,
                node_id=parent_node.node_id,
                model_kwargs=self.judge_model_kwargs,
            )
        active_ids = {rubric.rubric_id for rubric in self.active_bank}
        for generated_sample in generated_samples:
            format_errors = copy.deepcopy(generated_sample.format_errors or [])
            if format_errors:
                error_payload = {
                    "rubric_list_id": generated_sample.rubric_list_id,
                    "error": "Invalid rubric generation response",
                    "format_errors": format_errors,
                }
                rubric_generation_errors.append(error_payload)
                rubric_samples.append(
                    {
                        "sample_index": generated_sample.sample_index,
                        "rubric_list_id": generated_sample.rubric_list_id,
                        "generated": generated_sample.generated,
                        "messages": generated_sample.messages,
                        "format_errors": format_errors,
                        "is_valid": False,
                        "selected": False,
                    }
                )
                continue
            generated_rubrics: list[RubricRecord] = []
            seen_rubric_ids: set[str] = set()
            for rubric in generated_sample.generated:
                if rubric.rubric_id in active_ids or rubric.rubric_id in seen_rubric_ids:
                    continue
                seen_rubric_ids.add(rubric.rubric_id)
                generated_rubrics.append(rubric)
            scoring_rubrics = self.active_bank + generated_rubrics
            if generated_rubrics:
                generated_continuation_scores, continuation_errors = await _score_round(
                    question=question,
                    shared_context=shared_context,
                    continuations=continuations,
                    rubrics=generated_rubrics,
                    model_name=self.judge_model_name,
                    temperature=self.search_config.judge_temperature,
                    top_p=self.search_config.judge_top_p,
                    max_tokens=self.search_config.judge_max_tokens,
                    model_kwargs=self.judge_model_kwargs,
                )
            else:
                generated_continuation_scores = [[] for _ in continuations]
                continuation_errors = []
            scored_continuations = [
                copy.deepcopy(active_records) + generated_records
                for active_records, generated_records in zip(active_continuation_scores, generated_continuation_scores)
            ]
            parent_scores = copy.deepcopy(active_parent_scores)
            judge_errors_payload = copy.deepcopy(active_continuation_errors + active_parent_errors + continuation_errors)
            if compare_parent and generated_rubrics:
                parent_scores, parent_errors = await _score_parent_round(
                    question=question,
                    shared_context=shared_context,
                    rubrics=generated_rubrics,
                    model_name=self.judge_model_name,
                    temperature=self.search_config.judge_temperature,
                    top_p=self.search_config.judge_top_p,
                    max_tokens=self.search_config.judge_max_tokens,
                    node_id=parent_node.node_id,
                    model_kwargs=self.judge_model_kwargs,
                )
                parent_scores = copy.deepcopy(active_parent_scores) + parent_scores
                judge_errors_payload.extend(parent_errors)

            child_score_lookup_by_node = {branch["node_id"]: {} for branch in branch_records}
            for branch, score_records in zip(branch_records, scored_continuations):
                for record in score_records:
                    child_score_lookup_by_node[branch["node_id"]][record["rubric_id"]] = float(record["score_normalized"])
            parent_score_lookup = {record["rubric_id"]: float(record["score_normalized"]) for record in parent_scores}

            parent_score_by_rubric: dict[str, float] = {}
            child_score_by_rubric: dict[str, dict[str, float]] = {}
            variance_by_rubric: dict[str, float] = {}
            redundency_by_rubric: dict[str, float] = {}
            judge_error_by_rubric: dict[str, float] = {}
            reward_by_rubric: dict[str, float] = {}
            previous_score_vectors: list[list[float]] = []
            for error in judge_errors_payload:
                rubric_id = error.get("rubric_id")
                if rubric_id:
                    judge_error_by_rubric[rubric_id] = judge_error_by_rubric.get(rubric_id, 0.0) + JUDGE_ERROR_REWARD
            for rubric in scoring_rubrics:
                vector: list[float] = []
                parent_score = parent_score_lookup.get(rubric.rubric_id, 0.0)
                parent_score_by_rubric[rubric.rubric_id] = parent_score
                if compare_parent:
                    vector.append(parent_score)
                child_scores_for_rubric: dict[str, float] = {}
                for branch in branch_records:
                    score = child_score_lookup_by_node[branch["node_id"]].get(rubric.rubric_id, 0.0)
                    child_scores_for_rubric[branch["node_id"]] = score
                    vector.append(score)
                child_score_by_rubric[rubric.rubric_id] = child_scores_for_rubric
                variance_by_rubric[rubric.rubric_id] = 0.0 if len(vector) <= 1 else float(pvariance(vector))
                redundancy_reward = _redundancy_reward(vector, previous_score_vectors)
                redundency_by_rubric[rubric.rubric_id] = redundancy_reward
                # NOTE: when computing the rubric bank, we use equal weight
                reward_by_rubric[rubric.rubric_id] = (
                    variance_by_rubric[rubric.rubric_id]
                    + redundancy_reward
                    + judge_error_by_rubric.get(rubric.rubric_id, 0.0)
                )
                previous_score_vectors.append(vector)

            active_after, inactive_after, _ = _update_rubric_bank(
                active_bank=self.active_bank,
                inactive_bank=self.inactive_bank,
                generated=generated_sample.generated,
                rewards=reward_by_rubric,
                max_active_rubrics=self.search_config.max_active_rubrics,
            )
            parent_reward = 0.0
            if compare_parent and active_after:
                parent_reward = sum(parent_score_lookup.get(rubric.rubric_id, 0.0) for rubric in active_after) / len(active_after)
            child_rewards: dict[str, float] = {}
            for branch in branch_records:
                score_lookup = child_score_lookup_by_node[branch["node_id"]]
                child_rewards[branch["node_id"]] = (
                    sum(score_lookup.get(rubric.rubric_id, 0.0) * rubric.weight for rubric in active_after) / len(active_after)
                    if active_after
                    else 0.0
                )

            generated_ids = {rubric.rubric_id for rubric in generated_sample.generated}
            sample_payload = {
                "sample_index": generated_sample.sample_index,
                "rubric_list_id": generated_sample.rubric_list_id,
                "generated": generated_sample.generated,
                "messages": generated_sample.messages,
                "format_errors": copy.deepcopy(generated_sample.format_errors or []),
                "generated_titles": [rubric.title for rubric in generated_sample.generated],
                "active_before": copy.deepcopy(self.active_bank),
                "active_after": active_after,
                "inactive_after": inactive_after,
                "child_score_by_rubric": {
                    rubric_id: {
                        node_id: score
                        for node_id, score in node_scores.items()
                    }
                    for rubric_id, node_scores in child_score_by_rubric.items()
                    if rubric_id in generated_ids
                },
                "parent_score_by_rubric": {
                    rubric_id: score
                    for rubric_id, score in parent_score_by_rubric.items()
                    if rubric_id in generated_ids
                },
                "child_rewards": child_rewards,
                "parent_reward": parent_reward,
                "variance_by_rubric": {
                    rubric_id: variance
                    for rubric_id, variance in variance_by_rubric.items()
                    if rubric_id in generated_ids
                },
                "redundency_by_rubric": {
                    rubric_id: reward
                    for rubric_id, reward in redundency_by_rubric.items()
                    if rubric_id in generated_ids
                },
                "judge_error_by_rubric": {
                    rubric_id: judge_error_by_rubric.get(rubric_id, 0.0)
                    for rubric_id in generated_ids
                },
                "reward_by_rubric": {
                    rubric_id: reward
                    for rubric_id, reward in reward_by_rubric.items()
                    if rubric_id in generated_ids
                },
                "judge_errors": judge_errors_payload,
                "is_valid": True,
                "selected": False,
            }
            rubric_samples.append(sample_payload)
            valid_rubric_samples.append(sample_payload)

        if len(valid_rubric_samples) < min(2, self.search_config.n):
            return {
                "rubric_samples": rubric_samples,
                "rubric_generation_errors": rubric_generation_errors,
                "stop_search": True,
            }

        selected_sample = random.choice(valid_rubric_samples)
        selected_sample["selected"] = True
        averaged_child_rewards = [
            sum(sample["child_rewards"][branch["node_id"]] for sample in valid_rubric_samples) / len(valid_rubric_samples)
            for branch in branch_records
        ]
        parent_reward = sum(sample["parent_reward"] for sample in valid_rubric_samples) / len(valid_rubric_samples)
        return {
            "rubric_samples": rubric_samples,
            "child_rewards": averaged_child_rewards,
            "parent_reward": parent_reward,
            "selected_sample_index": selected_sample["sample_index"],
            "selected_sample": selected_sample,
            "rubric_generation_errors": rubric_generation_errors,
            "stop_search": False,
        }

    def _run_round(self, parent_id: str, round_index: int) -> None:
        parent_node = self.nodes[parent_id]
        if parent_id not in self._node_snapshot_cache:
            raise RuntimeError(f"Node {parent_id} does not have a cached restorable snapshot")
        if parent_id not in self._node_judge_cache:
            raise RuntimeError(f"Node {parent_id} does not have a cached judge payload")
        parent_snapshot = copy.deepcopy(self._node_snapshot_cache[parent_id])
        parent_judge = copy.deepcopy(self._node_judge_cache[parent_id])
        parent_rubric_round = parent_judge.get("rubric_round", {})
        self.active_bank = [RubricRecord(**rubric) for rubric in parent_rubric_round.get("active_bank_after", [])]
        if parent_id == "root" and not self.active_bank:
            raise RuntimeError("Root rubric bank was not initialized in _initialize_root")
        self.inactive_bank = [RubricRecord(**rubric) for rubric in parent_rubric_round.get("inactive_bank_after", [])]
        previous_frontier = list(self.frontier_ids)
        branch_records: list[dict[str, Any]] = []
        valid_branch_records: list[dict[str, Any]] = []
        policy_generation_errors: list[dict[str, Any]] = []
        sample_plan: list[tuple[str, str]] = []
        sample_count = self.search_config.m if parent_id == "root" else self.search_config.m
        if self.student_policy_model_name:
            teacher_count = (sample_count + 1) // 2
            student_count = sample_count // 2
            sample_plan = [("teacher", self.policy_model_name)] * teacher_count + [("student", self.student_policy_model_name)] * student_count
        else:
            sample_plan = [("teacher", self.policy_model_name)] * sample_count

        for sample_index, (policy_source, policy_model_name) in enumerate(sample_plan):
            node_id = f"node-r{round_index:03d}-s{sample_index:02d}-{uuid.uuid4().hex[:6]}"
            resumed_snapshot = copy.deepcopy(parent_snapshot)
            resumed_snapshot["session_id"] = f"{node_id}-session"
            resumed_snapshot["spec"]["session_id"] = resumed_snapshot["session_id"]
            resumed_snapshot["spec"]["policy_ref"] = policy_model_name
            
            resumed_snapshot["spec"]["policy_version"] = policy_model_name
            resumed_snapshot["model"]["config"]["model_name"] = policy_model_name
            resumed_snapshot["model"]["config"]["route_name"] = "policy_student" if policy_source == "student" else "policy"
            for index, event in enumerate(resumed_snapshot.get("metadata", {}).get("events", [])):
                event["session_id"] = resumed_snapshot["session_id"]
                event["event_id"] = f"{resumed_snapshot['session_id']}:{index}"
            for turn in resumed_snapshot.get("metadata", {}).get("model_turns", []):
                turn["session_id"] = resumed_snapshot["session_id"]
            session = self.backend.resume_session(RolloutSnapshot(**resumed_snapshot))
            try:
                result = session.run_until_pause(max_steps=self.search_config.k).model_dump(mode="json")
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                policy_generation_errors.append(
                    {
                        "node_id": node_id,
                        "error": error_text,
                    }
                )
                snapshot_after = session.snapshot().model_dump(mode="json")
                before_message_count = len(parent_snapshot.get("agent", {}).get("state", {}).get("messages", []))
                message_payload = copy.deepcopy(
                    snapshot_after.get("agent", {}).get("state", {}).get("messages", [])[before_message_count:]
                )
                result = {
                    "status": "error",
                    "exit_status": type(exc).__name__,
                    "submission": "",
                    "metadata": {},
                }
                branch_records.append({
                    "node_id": node_id,
                    "result": result,
                    "policy_source": policy_source,
                    "policy_model_name": policy_model_name,
                    "error": error_text,
                    "prompt_payload": {"messages": copy.deepcopy(resumed_snapshot.get("agent").get("state").get("messages"))},
                    "message_payload": build_messages(message_payload, model_name=policy_model_name),
                    "is_valid": False,
                })
                self._dispose_session(session)
                continue
            snapshot_after = session.snapshot().model_dump(mode="json")
            workspace_meta = _collect_workspace_meta(session.agent.env)
            before_event_count = len(parent_snapshot.get("metadata", {}).get("events", []))
            before_message_count = len(parent_snapshot.get("agent", {}).get("state", {}).get("messages", []))
            segment_events = copy.deepcopy(snapshot_after.get("metadata", {}).get("events", [])[before_event_count:])
            message_payload = copy.deepcopy(
                snapshot_after.get("agent", {}).get("state", {}).get("messages", [])[before_message_count:]
            )
            step_start = parent_node.step_end
            step_cards = _build_step_cards(segment_events, step_start)
            step_end = step_start + len(step_cards)
            recent_segments = copy.deepcopy(parent_judge["recent_segments"])
            recent_segments.append({"step_cards": copy.deepcopy(step_cards), "segment_step_range": [step_start, step_end]})
            if len(recent_segments) > 2:
                recent_segments = recent_segments[-2:]
            branch_record = {
                "node_id": node_id,
                "session": session,
                "result": result,
                "snapshot_after": snapshot_after,
                "policy_source": policy_source,
                "policy_model_name": policy_model_name,
                "workspace_meta": workspace_meta,
                "prompt_payload": {"messages": copy.deepcopy(resumed_snapshot.get("agent").get("state").get("messages"))},
                "message_payload": build_messages(message_payload, model_name=policy_model_name),
                "step_start": step_start,
                "step_end": step_end,
                "recent_segments": recent_segments,
                "is_valid": True,
            }
            branch_records.append(branch_record)
            valid_branch_records.append(branch_record)

        if len(valid_branch_records) < 2:
            self.peaceful_exit(branch_records, parent_id)
            return

        judged = asyncio.run(
            self._prepare_round_judging(
                parent_node=parent_node,
                parent_judge=parent_judge,
                branch_records=valid_branch_records,
                round_index=round_index,
                compare_parent=parent_id != "root",
            )
        )
        if bool(judged.get("stop_search")):
            self.peaceful_exit(branch_records, parent_id)
            return
        selected_sample = judged["selected_sample"]
        parent_baseline_reward = float(judged["parent_reward"])
        node_round_records = []
        valid_branches: list[dict[str, Any]] = []
        for index, (branch, averaged_reward) in enumerate(zip(valid_branch_records, judged["child_rewards"])):
            branch["reward"] = float(averaged_reward)
            valid_branches.append(branch)
            node_round_records.append({
                "node_id": branch["node_id"],
                "score": branch["reward"],
            })
        valid_branches.sort(key=lambda branch: (branch["reward"], branch["result"]["status"] == "finished"), reverse=True)
        frontier_branches = (
            valid_branches
            if parent_id == "root"
            else [branch for branch in valid_branches if branch["reward"] >= parent_baseline_reward + self.search_config.regression_margin]
        )
        chosen_branches = _sample_by_strategy(
            frontier_branches,
            [branch["reward"] for branch in frontier_branches],
            count=self.search_config.p,
            strategy=self.search_config.strategy,
        )
        top_child_ids = [branch["node_id"] for branch in chosen_branches]
        regressed = parent_id != "root" and not top_child_ids
        candidate_frontier_ids = list(top_child_ids)
        candidate_frontier_ids.extend(
            node_id for node_id in previous_frontier if node_id != parent_id and node_id not in candidate_frontier_ids
        )
        kept_child_ids = set(top_child_ids)
        if regressed and parent_id != "root":
            self.best_node_id = parent_id

        round_rubric_path = self.rubrics_dir / f"round_{round_index:03d}.json"
        rubric_sample_payloads = []
        rubric_artifact_bundles: list[RubricArtifactBundle] = []
        for sample in judged["rubric_samples"]:
            rubric_dir = self.rubrics_dir / sample["rubric_list_id"]
            if sample.get("is_valid") is False:
                rubric_payload = {
                    "rubric_list_id": sample["rubric_list_id"],
                    "parent_node_id": parent_id,
                    "generated": [asdict(rubric) for rubric in sample["generated"]],
                    "format_errors": copy.deepcopy(sample["format_errors"]),
                    "is_valid": False,
                    "selected": bool(sample["selected"]),
                }
            else:
                rubric_payload = {
                    "rubric_list_id": sample["rubric_list_id"],
                    "parent_node_id": parent_id,
                    "generated": [asdict(rubric) for rubric in sample["generated"]],
                    "active_bank_before": [asdict(rubric) for rubric in sample["active_before"]],
                    "active_bank_after": [asdict(rubric) for rubric in sample["active_after"]],
                    "inactive_bank_after": [asdict(rubric) for rubric in sample["inactive_after"]],
                    "variance_by_rubric": copy.deepcopy(sample["variance_by_rubric"]),
                    "redundency_by_rubric": copy.deepcopy(sample["redundency_by_rubric"]),
                    "judge_error_by_rubric": copy.deepcopy(sample["judge_error_by_rubric"]),
                    "reward_by_rubric": copy.deepcopy(sample["reward_by_rubric"]),
                    "child_score_by_rubric": copy.deepcopy(sample["child_score_by_rubric"]),
                    "parent_score_by_rubric": copy.deepcopy(sample["parent_score_by_rubric"]),
                    "child_rewards": copy.deepcopy(sample["child_rewards"]),
                    "parent_reward": sample["parent_reward"],
                    "generated_titles": list(sample["generated_titles"]),
                    "is_valid": True,
                    "selected": bool(sample["selected"]),
                    "gt_by_rubric": {},
                    "gt_reward_siblings": 0.0,
                    "gt_reward_parent": 0.0,
                }
            rubric_sample_payloads.append(rubric_payload)
            rubric_artifact_bundles.append(
                RubricArtifactBundle(
                    rubric_dir=rubric_dir,
                    rubric_payload=rubric_payload,
                    messages_payload=sample["messages"],
                    round_summary_path=round_rubric_path,
                    selected_for_round_summary=bool(sample["selected"]),
                )
            )
        round_payload = {
            "round_index": round_index,
            "parent_id": parent_id,
            "selected_sample_index": judged["selected_sample_index"],
            "selected_rubric_list_id": selected_sample["rubric_list_id"],
            "regressed": regressed,
            "generated": [asdict(rubric) for rubric in selected_sample["generated"]],
            "active_bank_before": [asdict(rubric) for rubric in selected_sample["active_before"]],
            "active_bank_after": [asdict(rubric) for rubric in selected_sample["active_after"]],
            "inactive_bank_after": [asdict(rubric) for rubric in selected_sample["inactive_after"]],
            "parent_reward": selected_sample["parent_reward"],
            "child_rewards": copy.deepcopy(selected_sample["child_rewards"]),
            "parent_score_by_rubric": copy.deepcopy(selected_sample["parent_score_by_rubric"]),
            "child_score_by_rubric": copy.deepcopy(selected_sample["child_score_by_rubric"]),
            "variance_by_rubric": copy.deepcopy(selected_sample["variance_by_rubric"]),
            "redundency_by_rubric": copy.deepcopy(selected_sample["redundency_by_rubric"]),
            "judge_error_by_rubric": copy.deepcopy(selected_sample["judge_error_by_rubric"]),
            "reward_by_rubric": copy.deepcopy(selected_sample["reward_by_rubric"]),
            "gt_by_rubric": {},
            "gt_reward_siblings": 0.0,
            "gt_reward_parent": 0.0,
            "baseline_parent_reward": parent_baseline_reward,
            "node_scores": node_round_records,
            "judge_errors": copy.deepcopy(selected_sample["judge_errors"]),
            "policy_generation_errors": policy_generation_errors,
            "rubric_generation_errors": judged["rubric_generation_errors"],
            "rubric_samples": rubric_sample_payloads,
        }
        self.active_bank = copy.deepcopy(selected_sample["active_after"])
        self.inactive_bank = copy.deepcopy(selected_sample["inactive_after"])

        for branch in valid_branch_records:
            branch["keep_snapshot"] = branch["node_id"] in kept_child_ids
            branch["image_tag"] = None
            branch["image_id"] = None
            if not branch["keep_snapshot"]:
                continue
            image_tag = f"{self.image_repository}:round-{round_index:03d}-{branch['node_id'][-6:]}"
            image_tag, image_id = _docker_commit(
                self.docker_executable,
                getattr(branch["session"].agent.env, "container_id", None),
                image_tag,
            )
            branch["image_tag"] = image_tag
            branch["image_id"] = image_id

        artifact_bundles: list[NodeArtifactBundle] = []
        for branch in branch_records:
            if not branch["is_valid"]:
                artifact_bundles.append(
                    NodeArtifactBundle(
                        node_id=branch["node_id"],
                        node_dir=self.nodes_dir / branch["node_id"],
                        node_payload={
                            "node_id": branch["node_id"],
                            "parent_id": parent_id,
                            "round_index": round_index,
                            "depth": parent_node.depth + 1,
                            "status": "error",
                            "error": branch.get("error"),
                            "policy_source": branch["policy_source"],
                            "policy_model_name": branch["policy_model_name"],
                        },
                        messages_payload=branch["message_payload"],
                        judge_payload={"is_valid": False},
                        prompt_payload=branch["prompt_payload"],
                    )
                )
                continue
            node = SearchNode(
                node_id=branch["node_id"],
                parent_id=parent_id,
                round_index=round_index,
                depth=parent_node.depth + 1,
                session_id=branch["snapshot_after"]["session_id"],
                status="finished" if branch["result"]["status"] == "finished" and branch["keep_snapshot"] else ("frontier" if branch["keep_snapshot"] else "dropped"),
                step_start=branch["step_start"],
                step_end=branch["step_end"],
                submission=branch["result"].get("submission", ""),
                exit_status=branch["result"].get("exit_status", ""),
                policy_source=branch["policy_source"],
                policy_model_name=branch["policy_model_name"],
            )
            judge_payload = {
                "persistent_state": branch.get("persistent_state", copy.deepcopy(parent_judge.get("persistent_state", EMPTY_PERSISTENT_STATE))),
                "recent_segments": branch["recent_segments"],
                "workspace_meta": branch["workspace_meta"],
                "rubric_round": {
                    "selected_sample_index": judged["selected_sample_index"],
                    "rubric_list_id": selected_sample["rubric_list_id"],
                    "generated": [asdict(rubric) for rubric in selected_sample["generated"]],
                    "active_bank_before": [asdict(rubric) for rubric in selected_sample["active_before"]],
                    "active_bank_after": [asdict(rubric) for rubric in selected_sample["active_after"]],
                    "inactive_bank_after": [asdict(rubric) for rubric in selected_sample["inactive_after"]],
                    "variance_by_rubric": copy.deepcopy(selected_sample["variance_by_rubric"]),
                    "redundency_by_rubric": copy.deepcopy(selected_sample["redundency_by_rubric"]),
                    "judge_error_by_rubric": copy.deepcopy(selected_sample["judge_error_by_rubric"]),
                    "reward_by_rubric": copy.deepcopy(selected_sample["reward_by_rubric"]),
                    "judge_errors": copy.deepcopy(selected_sample["judge_errors"]),
                },
                "overall_reward": branch["reward"],
                "baseline_parent_reward": parent_baseline_reward if parent_id != "root" else None,
                "regressed_vs_parent": regressed,
            }

            terminal_messages = None
            terminal_patch = None
            if self.search_config.calculate_gt_reward:
                judge_payload["ground_truth_reward"] = None
                terminal_result = branch["result"]
                terminal_snapshot = branch["snapshot_after"]
                terminal_error = None
                if branch["result"]["status"] != "finished":
                    completed_steps = max(branch["step_end"] + 1, 0)
                    remaining_steps = max(self.search_config.step_limit - completed_steps, 0)
                    branch["session"].agent.model.config.model_kwargs["temperature"] = self.search_config.judge_temperature
                    branch["session"].agent.model.config.model_kwargs["top_p"] = self.search_config.judge_top_p
                    if remaining_steps > 0:
                        try:
                            terminal_result = branch["session"].run_until_pause(max_steps=remaining_steps).model_dump(mode="json")
                        except Exception as exc:
                            terminal_result = {
                                "status": "error",
                                "exit_status": type(exc).__name__,
                            }
                            terminal_error = f"{type(exc).__name__}: {exc}"
                        terminal_snapshot = branch["session"].snapshot().model_dump(mode="json")
                terminal_messages = build_messages(
                    terminal_snapshot["agent"]["state"].get("messages", []),
                    model_name=branch["policy_model_name"],
                )
                terminal_patch = {
                    self.task_id: {
                        "model_name_or_path": branch["policy_model_name"],
                        "instance_id": self.task_id,
                        "model_patch": terminal_result.get("submission", "") or "",
                        "terminal_error": terminal_error,
                    }
                }
            snapshot_payload = (
                {
                    **copy.deepcopy(branch["snapshot_after"]),
                    "environment": {
                        **copy.deepcopy(branch["snapshot_after"]["environment"]),
                        "config": {
                            **copy.deepcopy(branch["snapshot_after"]["environment"]["config"]),
                            "image": branch["image_tag"],
                        },
                        "state": {"owns_container": True},
                    },
                    "metadata": {
                        **copy.deepcopy(branch["snapshot_after"].get("metadata", {})),
                        "checkpoint_image_id": branch["image_id"],
                        "checkpoint_image_tag": branch["image_tag"],
                    },
                }
                if branch["keep_snapshot"]
                else None
            )
            self.nodes[node.node_id] = node
            self._node_judge_cache[node.node_id] = copy.deepcopy(judge_payload)
            if snapshot_payload is not None:
                self._node_snapshot_cache[node.node_id] = copy.deepcopy(snapshot_payload)
            else:
                self._node_snapshot_cache.pop(node.node_id, None)
            artifact_bundles.append(
                NodeArtifactBundle(
                    node_id=node.node_id,
                    node_dir=self.nodes_dir / node.node_id,
                    node_payload=asdict(node),
                    messages_payload=branch["message_payload"],
                    judge_payload=judge_payload,
                    prompt_payload=branch["prompt_payload"],
                    snapshot_payload=snapshot_payload,
                    terminal_messages_payload=terminal_messages,
                    terminal_patch_payload=terminal_patch,
                )
            )
            if node.status == "finished":
                self.finished_node_ids.append(node.node_id)

        self.finished_node_ids = sorted(set(self.finished_node_ids))
        self.frontier_ids = [
            node_id
            for node_id in candidate_frontier_ids
            if node_id in self.nodes and self.nodes[node_id].status in {"frontier", "finished"}
        ]
        non_gt_extra_json_writes: list[tuple[Path, Any]] = []
        gt_extra_json_writes: list[tuple[Path, Any]] = [(round_rubric_path, round_payload)]
        for node_id in previous_frontier:
            if node_id in self.nodes and self.nodes[node_id].status == "frontier" and node_id not in self.frontier_ids:
                self.nodes[node_id].status = "archived"
                non_gt_extra_json_writes.append((self.nodes_dir / node_id / "node.json", asdict(self.nodes[node_id])))
        for branch in valid_branch_records:
            self._dispose_session(branch["session"])

        if self.patch_eval_manager is not None:
            self.artifact_writer.submit_round(
                artifact_bundles,
                rubric_artifact_bundles,
                extra_json_writes=non_gt_extra_json_writes,
                write_gt_files=False,
            )
            self.patch_eval_manager.submit_round(
                artifact_bundles,
                rubric_artifact_bundles,
                extra_json_writes=gt_extra_json_writes,
            )
        else:
            self.artifact_writer.submit_round(
                artifact_bundles,
                rubric_artifact_bundles,
                extra_json_writes=gt_extra_json_writes + non_gt_extra_json_writes,
            )
        self._sweep_checkpoint_images()

    def _finalize_outputs(self) -> None:
        def _cached_overall_reward(node_id: str) -> float:
            reward = (self._node_judge_cache.get(node_id) or {}).get("overall_reward")
            return float(reward) if reward is not None else float("-inf")

        if self.finished_node_ids:
            final_node_id = max(
                (node_id for node_id in self.finished_node_ids if node_id in self.nodes),
                key=_cached_overall_reward,
            )
        elif self.frontier_ids:
            final_node_id = max(
                (node_id for node_id in self.frontier_ids if node_id in self.nodes),
                key=_cached_overall_reward,
            )
        elif self.best_node_id is not None:
            final_node_id = self.best_node_id
        else:
            return

        final_node = self.nodes[final_node_id]
        if final_node_id not in self._node_snapshot_cache:
            raise RuntimeError(f"Node {final_node_id} does not have a cached restorable snapshot")
        snapshot = copy.deepcopy(self._node_snapshot_cache[final_node_id])
        final_policy_model_name = final_node.policy_model_name or self.policy_model_name
        if self.search_config.write_artifacts:
            _atomic_write_json(
                self.run_dir / "messages.json",
                build_messages(
                    snapshot["agent"]["state"].get("messages", []),
                    model_name=final_policy_model_name,
                ),
            )
            _atomic_write_json(
                self.run_dir / "model_patch.json",
                {
                    self.task_id: {
                        "model_name_or_path": final_policy_model_name,
                        "instance_id": self.task_id,
                        "model_patch": final_node.submission,
                    }
                },
            )
        if self.search_config.evaluate_final_patch:
            patch_text = final_node.submission or ""
            if not patch_text.strip():
                if self.search_config.write_artifacts:
                    _atomic_write_json(
                        self.run_dir / "evaluation.json",
                        _build_evaluation_payload(
                            instance_id=self.task_id,
                            completed=False,
                            resolved=False,
                            empty_patch=True,
                            error=False,
                        ),
                    )
            else:
                error = False
                try:
                    reward = evaluate_swebench_instance_patches(
                        instance=self.instance,
                        patches_by_key={final_node_id: patch_text},
                        model_name=final_policy_model_name,
                        max_workers=1,
                        namespace=self.harness_namespace,
                        work_dir=self.run_dir,
                    ).get(final_node_id)
                except Exception:
                    reward = 0.0
                    error = True
                reward = float(reward)
                if self.search_config.write_artifacts:
                    _atomic_write_json(
                        self.run_dir / "evaluation.json",
                        _build_evaluation_payload(
                            instance_id=self.task_id,
                            completed=not error,
                            resolved=reward >= 1.0 and not error,
                            empty_patch=False,
                            error=error,
                        ),
                    )
        self.best_node_id = final_node_id
        if final_node.status == "finished":
            self.frontier_ids = [final_node_id]
        self._save_manifest()

    def peaceful_exit(self, branch_records, parent_id: int) -> None:
        for branch in branch_records:
            if branch.get("session") is not None:
                self._dispose_session(branch["session"])
                branch["session"] = None
        self.best_node_id = parent_id
        self.frontier_ids = []
