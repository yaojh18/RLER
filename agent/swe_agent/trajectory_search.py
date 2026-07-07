from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import os
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
    extract_last_json_object,
    extract_json_from_response,
    freeform_thought_model_kwargs,
    route_completion_message,
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
    gap_redundancy,
    rubric_score_record,
)
from swe_agent.run.run_swe_agent import build_messages, evaluate_swebench_instance_patches, make_evaluation_payload
from swe_agent.models.litellm_model import LitellmModel
from swe_agent.models.utils.retry import retry
from swe_agent.prompt import *
from swe_agent.rubric_bank import (
    ExperienceRubricBank,
    RubricGenerationSample,
    RubricRecord,
    ScoreRubricBank,
    _convert_rubric_item,
    _truncate_middle,
)


logger = logging.getLogger("swe_agent.trajectory_search")
RETURN_CODE_RE = re.compile(r"<returncode>(.*?)</returncode>", re.DOTALL)
EXCEPTION_RE = re.compile(r"<exception>(.*?)</exception>", re.DOTALL)
OUTPUT_RE = re.compile(r"<output>\s*(.*?)</output>", re.DOTALL)
PR_DESCRIPTION_RE = re.compile(r"<pr_description>\s*(.*?)\s*</pr_description>", re.DOTALL)
MAX_OBSERVATION_CHARS = 512
MIN_OBSERVATION_SECTION_CHARS = 128
MAX_RUBRICS = 6
MAX_RUBRIC_GENERATION_ROUNDS = 10
MAX_FORMAT_CORRECTION_ROUNDS = 4
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
    rubric_bank_strategy: Literal["score", "experience", "both"] = "score"


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
            commands = [
                _truncate_middle(action.get("command", ""), MAX_OBSERVATION_CHARS)
                for action in (event.get("payload") or {}).get("actions", [])
            ]
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


def _render_step_cards(step_cards: list[dict[str, Any]]) -> str:
    if not step_cards:
        return "None"
    return "\n\n".join(
        "\n".join(
            [
                f"### Step {card.get('step_index', index)}",
                "Assistant:",
                str(card.get("assistant_message") or ""),
                "Observation:",
                str(card.get("observation") or ""),
            ]
        )
        for index, card in enumerate(step_cards)
    )


def _render_trajectory_segment(segment: dict[str, Any] | None) -> str:
    if not segment:
        return "None"
    parts: list[str] = []
    step_range = segment.get("segment_step_range")
    if isinstance(step_range, list) and len(step_range) == 2:
        parts.append(f"Segment step range: {step_range[0]}-{step_range[1]}")
    parts.append(_render_step_cards(segment.get("step_cards") or []))
    return "\n\n".join(parts)


def _render_continuation_view(continuation: dict[str, Any]) -> str:
    parts = [
        "Summary:",
        json.dumps(continuation.get("summary", {}), indent=2, ensure_ascii=False),
        "Trajectory:",
        _render_trajectory_segment(continuation.get("trajectory_continuation")),
    ]
    if continuation.get("note"):
        parts.extend(["Note:", str(continuation["note"])])
    return "\n".join(parts)


def _collect_workspace_meta(environment: Any) -> dict[str, Any]:
    cwd = getattr(getattr(environment, "config", None), "cwd", "") or "/testbed"
    command = f"""python - <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess

requested_cwd = {json.dumps(cwd)}

def run(cmd):
    completed = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()

cwd = requested_cwd
for candidate in (requested_cwd, "/testbed", os.getcwd()):
    if candidate and os.path.isdir(candidate):
        cwd = candidate
        break
os.chdir(cwd)
git_root_ok, git_root, _ = run("git rev-parse --show-toplevel")
if git_root_ok == 0 and git_root and os.path.isdir(git_root):
    cwd = git_root
    os.chdir(cwd)

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


def _normalize_terminal_patch_text(patch: Any) -> str:
    text = "" if patch is None else str(patch)
    if not text.strip():
        return ""
    return text if text.endswith("\n") else text + "\n"


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
        return {"state": copy.deepcopy(previous_state), "messages": []}
    prompt = "\n\n".join(
        [
            PERSISTENT_STATE_UPDATE_PROMPT.strip(),
            f"\n\n## Question:\n System Prompt:\n{system_prompt}\n User Prompt:\n{user_prompt}",
            f"## Previous Persistent State:\n{json.dumps(previous_state, indent=2, ensure_ascii=False)}",
            f"## Evicted Older Trajectory:\n{_render_step_cards(evicted_step_cards)}",
            f"## Workspace Metadata:\n{json.dumps(workspace_meta, indent=2, ensure_ascii=False)}",
        ]
    )
    messages = [{"role": "user", "content": prompt}]
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
        response = assistant_message.get("content_no_thinking") or assistant_message.get("content") or ""
        messages.append(assistant_message)
        parsed = _parse_persistent_state_response(response)
        if parsed is not None:
            return {"state": parsed, "messages": copy.deepcopy(messages)}
        messages.append(
            {
                "role": "user",
                "content": STRUCTURED_SUMMARY_FORMAT_CORRECTION_PROMPT.strip(),
            }
        )
    raise ValueError("InvalidPersistentStateResponse")


def _parse_judge_score(response: str) -> int | None:
    parsed = extract_last_json_object(response)
    if not isinstance(parsed, dict) or "score" not in parsed:
        return None
    score = parsed.get("score")
    if isinstance(score, bool) or not isinstance(score, int):
        return None
    return score if 1 <= score <= 5 else None


def _parse_persistent_state_response(response: str) -> dict[str, Any] | None:
    parsed = extract_last_json_object(response)
    return parsed if isinstance(parsed, dict) else None


async def _generate_round_rubrics(
    *,
    question: dict[str, str],
    previous_state: dict[str, Any],
    latest_shared_segment: dict[str, Any] | None,
    continuations: list[dict[str, Any]],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    round_index: int,
    sample_index: int = 0,
    model_kwargs: dict[str, Any] | None = None,
    extra_prompt_sections: list[str] | None = None,
    generation_prompt: str = SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT,
    continue_prompt: str = RUBRIC_GENERATION_CONTINUE_PROMPT,
    rubric_list_prefix: str = "rubric",
    require_first_rubric: bool = False,
) -> RubricGenerationSample:
    latest_shared_segment_text = _render_trajectory_segment(latest_shared_segment)
    prompt_parts = [
        generation_prompt.strip(),
        f"\n\n## Question:\nSystem Prompt:\n{question.get('system_prompt', '')}",
        f"\nUser Prompt:\n{question.get('user_prompt', '')}",
        f"\n## Previous Persistent State:\n{json.dumps(previous_state, ensure_ascii=False, indent=2)}",
        f"\n## Parent Trajectory:\n{latest_shared_segment_text}\n",
    ]
    prompt_parts.append("## Agent Trajectory Continuations:")
    for index, continuation in enumerate(continuations, start=1):
        prompt_parts.extend(
            [
                f"## Continuation {index}:",
                _render_continuation_view(continuation),
                "",
            ]
        )
    for section in extra_prompt_sections or []:
        if section:
            prompt_parts.extend(["", section])
    prompt = "\n".join(prompt_parts)
    conversation_messages = [{"role": "user", "content": prompt}]
    task_text = "\n\n".join(part for part in [question.get("system_prompt", ""), question.get("user_prompt", "")] if part)
    generated: list[RubricRecord] = []
    format_errors: list[dict[str, Any]] = []
    length_error = None
    turn_index = 1
    while True:
        if turn_index > MAX_RUBRIC_GENERATION_ROUNDS:
            length_error = f"Reached rubric generation max rounds={MAX_RUBRIC_GENERATION_ROUNDS}."
            break
        parsed_candidate: dict[str, Any] | None = None
        parsed: dict[str, Any] | None = None
        assistant_content = ""
        last_error = ""
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
                    messages=conversation_messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    model_kwargs=freeform_thought_model_kwargs(model_kwargs),
                )
        assistant_content = assistant_message.get("content_no_thinking") or assistant_message.get("content") or ""
        conversation_messages.append(assistant_message)
        parsed_candidate = extract_last_json_object(assistant_content)
        if parsed_candidate == {}:
            if require_first_rubric and not generated:
                last_error = "Expected one complete rubric before returning {}."
            else:
                break
        elif parsed_candidate is None:
            last_error = "Expected a JSON object or {}, but no JSON object could be parsed."
        elif not {"polarity", "weight", "title", "description", "metadata", "scale"}.issubset(parsed_candidate):
            last_error = "Expected polarity, weight, title, description, metadata, and scale fields."
        else:
            parsed = _convert_rubric_item(task_text, parsed_candidate, round_index)
            if parsed is None:
                last_error = (
                    "Expected a rubric object with polarity, positive weight, title, description, metadata, "
                    "and a 1-5 scale."
                )
        if parsed is None:
            format_errors.append({"turn_index": turn_index, "error": last_error})
            conversation_messages.append(
                {
                    "role": "user",
                    "content": (
                        last_error
                        + " Follow the output format example above. The final JSON must contain polarity, positive weight, title, "
                        + "description, metadata, and scale fields, or contain {} only after at least "
                        + f"one complete rubric has been generated. {continue_prompt}"
                    ),
                }
            )
            turn_index += 1
            continue
        generated.append(parsed)
        if require_first_rubric:
            break
        if len(generated) >= MAX_RUBRICS:
            length_error = f"Reached max rubrics={MAX_RUBRICS}."
            break
        conversation_messages.append({"role": "user", "content": continue_prompt})
        turn_index += 1
    return RubricGenerationSample(
        sample_index=sample_index,
        rubric_list_id=f"{rubric_list_prefix}-r{round_index:03d}-s{sample_index:02d}",
        generated=generated,
        messages=copy.deepcopy(conversation_messages),
        format_errors=format_errors,
        terminal_error=length_error,
    )


def _rubric_judge_view(rubric: RubricRecord) -> dict[str, Any]:
    return {
        "title": rubric.title,
        "direction": rubric.direction,
        "description": rubric.description,
        "scale": rubric.scale,
        "metadata": rubric.metadata,
    }


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
    judge_prompt: str = SWE_TRAJECTORY_RUBRIC_JUDGE_PROMPT,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, str]]]:
    if not continuations or not rubrics:
        return [[] for _ in continuations], []
    calls = []
    mapping: list[tuple[int, str, RubricRecord]] = []
    question_text = f"System Prompt:\n{question.get('system_prompt', '')}\n\nUser Prompt:\n{question.get('user_prompt', '')}"
    parent_trajectory_text = _render_trajectory_segment(shared_context.get("latest_agent_trajectory"))
    for view_index, continuation in enumerate(continuations):
        node_id = continuation.get("node_id")
        response_text = _render_continuation_view(continuation)
        for rubric in rubrics:
            judge_rubric = _rubric_judge_view(rubric)
            criterion = "\n".join(
                [
                    f"Title: {judge_rubric['title']}",
                    f"Type: {judge_rubric['direction']}",
                    f"Description: {judge_rubric['description']}",
                    "Scale:",
                    *[f"{score}: {judge_rubric['scale'][str(score)]}" for score in range(1, 6)],
                    "Metadata:",
                    json.dumps(judge_rubric["metadata"], indent=2, ensure_ascii=False),
                ]
            )

            async def _judge_single(
                *,
                response_text: str = response_text,
                criterion: str = criterion,
            ) -> tuple[int, str | None, dict[str, Any]]:
                prompt_parts = [
                    judge_prompt.strip(),
                    f"\n\n## Question:\n{question_text}\n",
                    f"## Previous Persistent State:\n{shared_context.get('previous_persistent_state')}\n",
                    f"## Parent Trajectory:\n{parent_trajectory_text}\n",
                    f"## Continuation Trajectory:\n{response_text}\n",
                    f"## Criterion:\n{criterion}",
                ]
                messages = [{"role": "user", "content": "".join(prompt_parts)}]
                last_message: dict[str, Any] | None = None
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
                        last_message = assistant_message
                        messages.append(assistant_message)
                        response = assistant_message.get("content_no_thinking") or assistant_message.get("content") or ""
                        score_raw = _parse_judge_score(response)
                        if score_raw is not None:
                            return score_raw, None, assistant_message
                        messages.append(
                            {
                                "role": "user",
                                "content": RUBRIC_JUDGE_FORMAT_CORRECTION_PROMPT.strip(),
                            }
                        )
                except Exception:
                    return 1, "InvalidJudgeResponse", last_message or {
                        "role": "assistant",
                        "content": "",
                    }
                return 1, "InvalidJudgeResponse", last_message or {
                    "role": "assistant",
                    "content": "",
                }
            calls.append(_judge_single())
            mapping.append((view_index, node_id, rubric))
    responses = await asyncio.gather(*calls)
    per_view_scores: list[list[dict[str, Any]]] = [[] for _ in continuations]
    errors: list[dict[str, str]] = []
    for (view_index, node_id, rubric), response in zip(mapping, responses):
        score_raw, error, judge_message = response
        record = rubric_score_record(rubric, score_raw, judge_message)
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


async def _generate_and_score_rubric_batch(
    *,
    sample_count: int,
    round_index: int,
    generation_kwargs: dict[str, Any],
    generation_prompt: str,
    rubric_list_prefix: str,
    question: dict[str, Any],
    shared_context: dict[str, Any],
    continuations: list[dict[str, Any]],
    extra_prompt_sections: list[str],
    judge_kwargs: dict[str, Any],
    judge_prompt: str = SWE_TRAJECTORY_RUBRIC_JUDGE_PROMPT,
    require_first_rubric: bool = False,
) -> dict[str, Any]:
    generated_samples = await asyncio.gather(
        *[
            _generate_round_rubrics(
                **generation_kwargs,
                question=question,
                previous_state=shared_context["previous_persistent_state"],
                latest_shared_segment=shared_context["latest_agent_trajectory"],
                continuations=continuations,
                round_index=round_index,
                extra_prompt_sections=extra_prompt_sections,
                sample_index=sample_index,
                generation_prompt=generation_prompt,
                rubric_list_prefix=rubric_list_prefix,
                require_first_rubric=require_first_rubric,
            )
            for sample_index in range(sample_count)
        ],
    )

    sample_generated_rubrics: list[list[RubricRecord]] = []
    for generated_sample in generated_samples:
        seen_rubric_ids: set[str] = set()
        generated_rubrics: list[RubricRecord] = []
        for rubric in generated_sample.generated:
            if rubric.rubric_id in seen_rubric_ids:
                continue
            seen_rubric_ids.add(rubric.rubric_id)
            generated_rubrics.append(rubric)
        sample_generated_rubrics.append(generated_rubrics)

    generated_score_tasks: dict[int, asyncio.Task] = {
        sample_index: asyncio.create_task(
            _score_round(
                question=question,
                shared_context=shared_context,
                continuations=continuations,
                rubrics=generated_rubrics,
                **judge_kwargs,
                judge_prompt=judge_prompt,
            )
        )
        for sample_index, generated_rubrics in enumerate(sample_generated_rubrics)
        if generated_rubrics
    }
    await asyncio.gather(*generated_score_tasks.values())
    generated_score_results = {
        sample_index: ([[] for _ in continuations], [])
        for sample_index in range(len(generated_samples))
    }
    for sample_index, task in generated_score_tasks.items():
        generated_score_results[sample_index] = task.result()
    return {
        "generated_samples": list(generated_samples),
        "sample_generated_rubrics": sample_generated_rubrics,
        "generated_score_results": generated_score_results,
    }


def _combine_score_results(
    left: tuple[list[list[dict[str, Any]]], list[dict[str, str]]] | None,
    right: tuple[list[list[dict[str, Any]]], list[dict[str, str]]] | None,
    *,
    continuation_count: int,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, str]]]:
    combined_scores: list[list[dict[str, Any]]] = [[] for _ in range(continuation_count)]
    combined_errors: list[dict[str, str]] = []
    for score_result in (left, right):
        if score_result is None:
            continue
        per_view_scores, errors = score_result
        for index in range(min(continuation_count, len(per_view_scores))):
            combined_scores[index].extend(copy.deepcopy(per_view_scores[index]))
        combined_errors.extend(copy.deepcopy(errors))
    return combined_scores, combined_errors


def _rubric_sample_evaluation(
    *,
    score_batch: dict[str, Any],
    sample_index: int,
    node_ids: list[str],
    scoring_rubrics: list[RubricRecord],
    include_variance_reward: bool,
) -> dict[str, Any]:
    generated_scores, generated_errors = score_batch["generated_score_results"][sample_index]
    scored_continuations = copy.deepcopy(generated_scores)
    judge_errors = copy.deepcopy(generated_errors)
    score_lookup_by_node = {node_id: {} for node_id in node_ids}
    for node_id, score_records in zip(node_ids, scored_continuations):
        for record in score_records:
            score_lookup_by_node[node_id][record["rubric_id"]] = float(record["score_normalized"])
    score_by_rubric: dict[str, dict[str, float]] = {}
    variance_by_rubric: dict[str, float] = {}
    redundency_by_rubric: dict[str, float] = {}
    judge_error_by_rubric: dict[str, float] = {}
    reward_by_rubric: dict[str, float] = {}
    previous_score_vectors: list[list[float]] = []
    for error in judge_errors:
        rubric_id = error.get("rubric_id")
        if rubric_id:
            judge_error_by_rubric[rubric_id] = judge_error_by_rubric.get(rubric_id, 0.0) + JUDGE_ERROR_REWARD
    for rubric in scoring_rubrics:
        vector = [
            score_lookup_by_node[node_id].get(rubric.rubric_id, 0.0)
            for node_id in node_ids
        ]
        score_by_rubric[rubric.rubric_id] = {
            node_id: score_lookup_by_node[node_id].get(rubric.rubric_id, 0.0)
            for node_id in node_ids
        }
        variance_by_rubric[rubric.rubric_id] = 0.0 if len(vector) <= 1 else float(pvariance(vector))
        redundancy_reward = _redundancy_reward(vector, previous_score_vectors)
        redundency_by_rubric[rubric.rubric_id] = redundancy_reward
        reward_by_rubric[rubric.rubric_id] = (
            (variance_by_rubric[rubric.rubric_id] if include_variance_reward else 0.0)
            + redundancy_reward
            + judge_error_by_rubric.get(rubric.rubric_id, 0.0)
        )
        previous_score_vectors.append(vector)
    metrics = {
        "score_lookup_by_node": score_lookup_by_node,
        "score_by_rubric": score_by_rubric,
        "variance_by_rubric": variance_by_rubric,
        "redundency_by_rubric": redundency_by_rubric,
        "judge_error_by_rubric": judge_error_by_rubric,
        "reward_by_rubric": reward_by_rubric,
    }
    return {
        "scored_continuations": scored_continuations,
        "judge_errors": judge_errors,
        "score_lookup_by_node": score_lookup_by_node,
        "metrics": metrics,
    }


def _rubric_metrics_payload(metrics: dict[str, Any], include_ids: set[str]) -> dict[str, Any]:
    return {
        "score_by_rubric": {
            rubric_id: copy.deepcopy(scores)
            for rubric_id, scores in metrics["score_by_rubric"].items()
            if rubric_id in include_ids
        },
        "variance_by_rubric": {
            rubric_id: value
            for rubric_id, value in metrics["variance_by_rubric"].items()
            if rubric_id in include_ids
        },
        "redundency_by_rubric": {
            rubric_id: value
            for rubric_id, value in metrics["redundency_by_rubric"].items()
            if rubric_id in include_ids
        },
        "judge_error_by_rubric": {
            rubric_id: metrics["judge_error_by_rubric"].get(rubric_id, 0.0)
            for rubric_id in include_ids
        },
        "reward_by_rubric": {
            rubric_id: value
            for rubric_id, value in metrics["reward_by_rubric"].items()
            if rubric_id in include_ids
        },
    }


def _avg_scores_from_rubrics(
    *,
    node_ids: list[str],
    score_lookup_by_node: dict[str, dict[str, float]],
    rubrics: list[RubricRecord],
) -> dict[str, float]:
    if not rubrics:
        return {node_id: 0.0 for node_id in node_ids}
    normalized_weights = _normalized_weight_by_rubric(rubrics)
    rewards: dict[str, float] = {}
    for node_id in node_ids:
        score_lookup = score_lookup_by_node[node_id]
        rewards[node_id] = sum(
            score_lookup.get(rubric.rubric_id, 0.0)
            * normalized_weights[rubric.rubric_id]
            * (-1.0 if rubric.direction == "negative" else 1.0)
            for rubric in rubrics
        )
    return rewards


def _pc_avg_scores_from_rubrics(
    *,
    node_ids: list[str],
    score_lookup_by_node: dict[str, dict[str, float]],
    rubrics: list[RubricRecord],
) -> dict[str, float]:
    if not rubrics:
        return {node_id: 0.5 for node_id in node_ids}
    normalized_weights = _normalized_weight_by_rubric(rubrics)
    rewards: dict[str, float] = {}
    for node_id in node_ids:
        total = 0.0
        score_lookup = score_lookup_by_node[node_id]
        for rubric in rubrics:
            raw = score_lookup.get(rubric.rubric_id, 0.0)
            if rubric.direction == "negative":
                contribution = 0.5 * (1.0 - raw)
            else:
                contribution = raw
            total += normalized_weights[rubric.rubric_id] * contribution
        rewards[node_id] = total
    return rewards


def _normalized_weight_by_rubric(rubrics: list[RubricRecord]) -> dict[str, float]:
    importance_by_rubric: dict[str, float] = {}
    for rubric in rubrics:
        importance = abs(float(rubric.weight))
        if not math.isfinite(importance) or importance <= 0.0:
            raise ValueError(f"Rubric {rubric.rubric_id} has invalid weight {rubric.weight!r}")
        importance_by_rubric[rubric.rubric_id] = importance
    total = sum(importance_by_rubric.values())
    return {rubric_id: importance / total for rubric_id, importance in importance_by_rubric.items()}


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


def _docker_commit(
    executable: str,
    container_id: str,
    image_tag: str,
    *,
    inspect_image: bool = True,
) -> tuple[str, str]:
    # --pause=false: containers can leak in 'paused' state under high commit
    # load. Safe here because we only commit after run_until_pause() returns,
    # i.e. when the agent step has finished and no in-container command is in
    # progress.
    subprocess.run(
        [executable, "commit", "--pause=false", container_id, image_tag],
        check=True, capture_output=True, text=True,
    )
    if not inspect_image:
        return image_tag, ""
    image_id = subprocess.run(
        [executable, "image", "inspect", image_tag, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return image_tag, image_id


def _resume_snapshot_payload(
    snapshot: dict[str, Any],
    *,
    image_tag: str,
    image_id: str,
) -> dict[str, Any]:
    payload = {
        "session_id": snapshot["session_id"],
        "status": snapshot["status"],
        "spec": copy.deepcopy(snapshot["spec"]),
        "agent": copy.deepcopy(snapshot["agent"]),
        "model": copy.deepcopy(snapshot["model"]),
        "environment": {
            **copy.deepcopy(snapshot["environment"]),
            "config": {
                **copy.deepcopy(snapshot["environment"]["config"]),
                "image": image_tag,
            },
            "state": {"owns_container": True},
        },
        "last_step_index": snapshot.get("last_step_index", -1),
        "last_event_id": None,
        "metadata": {
            "checkpoint_image_id": image_id,
            "checkpoint_image_tag": image_tag,
        },
    }
    if snapshot.get("memory") is not None:
        payload["memory"] = copy.deepcopy(snapshot["memory"])
    return payload


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
        experience_banks: dict[str, ExperienceRubricBank] | None = None,
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
        uses_experience_bank = self.search_config.rubric_bank_strategy in {"experience", "both"}
        uses_score_bank = self.search_config.rubric_bank_strategy in {"score", "both"}
        self.rubric_scopes = ("siblings", "pc")
        if uses_score_bank:
            self.rubric_bank = ScoreRubricBank(max_active_rubrics=self.search_config.max_active_rubrics, scope="siblings")
            self.pc_rubric_bank = ScoreRubricBank(max_active_rubrics=self.search_config.max_active_rubrics, scope="pc")
        else:
            self.rubric_bank = None
            self.pc_rubric_bank = None
        provided_experience_banks = experience_banks if experience_banks is not None else {}
        if uses_experience_bank:
            siblings_experience_bank = provided_experience_banks.get("siblings")
            self.experience_bank = (siblings_experience_bank if isinstance(siblings_experience_bank, ExperienceRubricBank)
                else ExperienceRubricBank(
                    retrieval_prompt=RUBRIC_EXPERIENCE_RETRIEVAL_PROMPT,
                    update_prompt=RUBRIC_EXPERIENCE_UPDATE_PROMPT,
                    scope="siblings",
                )
            )
            pc_experience_bank = provided_experience_banks.get("pc")
            self.pc_experience_bank = (pc_experience_bank if isinstance(pc_experience_bank, ExperienceRubricBank)
                else ExperienceRubricBank(
                    retrieval_prompt=PC_RUBRIC_EXPERIENCE_RETRIEVAL_PROMPT,
                    update_prompt=PC_RUBRIC_EXPERIENCE_UPDATE_PROMPT,
                    scope="pc",
                )
            )
        else:
            self.experience_bank = None
            self.pc_experience_bank = None
        self.score_banks = {
            "siblings": self.rubric_bank,
            "pc": self.pc_rubric_bank,
        }
        self.experience_banks = {
            "siblings": self.experience_bank,
            "pc": self.pc_experience_bank,
        }
        self.current_round = 0
        self.system_prompt = ""
        self.user_prompt = ""
        self.best_node_id: str | None = None
        self.finished_node_ids: list[str] = []
        self.peaceful_exit_triggered = False
        self.image_repository = f"rler-search/{self.task_id.replace('__', '-').lower()}"
        self.artifact_writer = ArtifactWriter(write_artifacts=self.search_config.write_artifacts)
        self._manifest_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="search-manifest")
        self._manifest_futures: list[Future] = []
        self._node_judge_cache: dict[str, dict[str, Any]] = {}
        self._node_snapshot_cache: dict[str, dict[str, Any]] = {}
        self._node_rubric_round_cache: dict[str, dict[str, Any]] = {}
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
        rubric_update_records: list[dict[str, Any]] = []
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
                round_rubric_records = self._run_round(self.frontier_ids[0], self.current_round)
                if any(bank is not None for bank in self.experience_banks.values()):
                    rubric_update_records.extend(round_rubric_records)
                self._save_manifest()
            self.artifact_writer.wait()
            patch_eval_payloads = []
            if self.patch_eval_manager is not None:
                patch_eval_payloads = self.patch_eval_manager.wait()
            if not self.peaceful_exit_triggered:
                self._update_experience_banks(rubric_update_records, patch_eval_payloads)
            elif any(bank is not None for bank in self.experience_banks.values()):
                logger.warning(
                    "Skipping experience bank update for %s because peaceful_exit was triggered.",
                    self.task_id,
                )
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


    def _update_experience_banks(
        self,
        rubric_update_records: list[dict[str, Any]],
        patch_eval_payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        terminal_evidence_by_rubric = {}
        for payload in patch_eval_payloads or []:
            for item in payload.get("rubric_update_payloads") or []:
                key = (item.get("scope"), item.get("rubric_list_id"))
                terminal_evidence_by_rubric[key] = {
                    "terminal_patch": str(item.get("terminal_patch") or ""),
                    "passed_tests": str(item.get("passed_tests") or ""),
                }
        experience_update_specs = [
            {"scope": scope, "bank": self.experience_banks[scope]}
            for scope in self.rubric_scopes
        ]
        experience_update_jobs = []
        for spec in experience_update_specs:
            if spec["bank"] is None:
                continue
            rubric_payloads = [
                {
                    **copy.deepcopy(record["rubric_payload"]),
                    **terminal_evidence_by_rubric.get(
                        (
                            spec["scope"],
                            record["rubric_payload"].get("rubric_list_id"),
                        ),
                        {},
                    ),
                    "round_index": int(record["round_index"]),
                    "messages": copy.deepcopy(record["messages"]),
                }
                for record in rubric_update_records
                if record.get("scope") == spec["scope"] and isinstance(record.get("rubric_payload"), dict)
            ]
            experience_update_jobs.append(
                (
                    spec["scope"],
                    spec["bank"].update_after_instance(
                        instance=self.instance,
                        rubric_payloads=rubric_payloads,
                        model_name=self.rubric_model_name,
                        temperature=self.search_config.rubric_temperature,
                        top_p=self.search_config.rubric_top_p,
                        max_tokens=self.search_config.rubric_max_tokens,
                        model_kwargs=self.rubric_model_kwargs,
                    ),
                )
            )
        if experience_update_jobs:
            async def _run_experience_updates() -> list[dict[str, Any]]:
                return list(await asyncio.gather(*(job for _, job in experience_update_jobs)))
            update_payloads = asyncio.run(_run_experience_updates())
            if self.search_config.write_artifacts:
                scoped_updates = {
                    scope: payload
                    for (scope, _), payload in zip(experience_update_jobs, update_payloads)
                }
                records_by_scope_round: dict[tuple[str, int], list[dict[str, Any]]] = {}
                for record in rubric_update_records:
                    if not isinstance(record.get("rubric_payload"), dict):
                        continue
                    key = (str(record.get("scope")), int(record["round_index"]))
                    records_by_scope_round.setdefault(key, []).append(record)
                for scope, payload in scoped_updates.items():
                    _atomic_write_json(
                        self.run_dir / f"{scope}_rubric_bank.json",
                        {
                            "before": copy.deepcopy(payload.get("before", [])),
                            "after": copy.deepcopy(payload.get("after", [])),
                        },
                    )
                    for update in payload.get("groups", []):
                        round_index = int(update["round_index"])
                        for record in records_by_scope_round.get((scope, round_index), []):
                            rubric_payload = record["rubric_payload"]
                            rubric_list_id = rubric_payload.get("rubric_list_id")
                            if not rubric_list_id:
                                continue
                            update_dir = self.rubrics_dir / scope / str(rubric_list_id)
                            _atomic_write_json(
                                update_dir / "rubric_bank.json",
                                {
                                    "before": copy.deepcopy(update.get("before", [])),
                                    "after": copy.deepcopy(update.get("after", [])),
                                    "actions": copy.deepcopy(update.get("actions", [])),
                                },
                            )
                            _atomic_write_json(
                                update_dir / "rubric_bank_message.json",
                                copy.deepcopy(update.get("messages", [])),
                            )


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
            "rubric_bank_strategy": self.search_config.rubric_bank_strategy,
            "frontier_ids": list(self.frontier_ids),
            "best_node_id": self.best_node_id,
            "finished_node_ids": list(self.finished_node_ids),
            "current_round": self.current_round,
            "peaceful_exit_triggered": self.peaceful_exit_triggered,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "node_ids": sorted(self.nodes),
        }
        if self.experience_bank is not None:
            manifest_payload["rubric_bank"] = self.experience_bank.to_list()
        if self.pc_experience_bank is not None:
            manifest_payload["pc_rubric_bank"] = self.pc_experience_bank.to_list()
        if self.rubric_bank is not None:
            manifest_payload["active_bank"] = [asdict(rubric) for rubric in self.rubric_bank.active_bank]
            manifest_payload["inactive_bank"] = [asdict(rubric) for rubric in self.rubric_bank.inactive_bank]
        if self.pc_rubric_bank is not None:
            manifest_payload["pc_active_bank"] = [asdict(rubric) for rubric in self.pc_rubric_bank.active_bank]
            manifest_payload["pc_inactive_bank"] = [asdict(rubric) for rubric in self.pc_rubric_bank.inactive_bank]
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
        self.peaceful_exit_triggered = bool(manifest.get("peaceful_exit_triggered", False))
        self.system_prompt = manifest.get("system_prompt", "")
        self.user_prompt = manifest.get("user_prompt", "")
        if self.rubric_bank is not None:
            active_bank = [RubricRecord(**rubric) for rubric in manifest.get("active_bank", [])]
            inactive_bank = [RubricRecord(**rubric) for rubric in manifest.get("inactive_bank", [])]
            self.rubric_bank.set_state(active_bank=active_bank, inactive_bank=inactive_bank)
        if self.pc_rubric_bank is not None:
            pc_active_bank = [RubricRecord(**rubric) for rubric in manifest.get("pc_active_bank", [])]
            pc_inactive_bank = [RubricRecord(**rubric) for rubric in manifest.get("pc_inactive_bank", [])]
            self.pc_rubric_bank.set_state(active_bank=pc_active_bank, inactive_bank=pc_inactive_bank)
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
        root_rubric_round = {"generated": []}
        if self.experience_bank is not None:
            root_rubric_round["retrieved"] = []
        if self.rubric_bank is not None:
            self.rubric_bank.initialize(initial_task_text or self.task)
            root_rubric_round.update(
                {
                    "active_bank_before": [],
                    "active_bank_after": [asdict(rubric) for rubric in self.rubric_bank.active_bank],
                    "inactive_bank_after": [],
                }
            )
        if self.pc_rubric_bank is not None:
            self.pc_rubric_bank.initialize(initial_task_text or self.task)
            root_rubric_round.update(
                {
                    "pc_active_bank_before": [],
                    "pc_active_bank_after": [asdict(rubric) for rubric in self.pc_rubric_bank.active_bank],
                    "pc_inactive_bank_after": [],
                }
            )
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
            "rubric_round": root_rubric_round,
            "overall_score": 0.0,
            "ground_truth_reward": 0.0,
        }
        root_snapshot = _resume_snapshot_payload(
            snapshot,
            image_tag=self.base_image,
            image_id=self.base_image_id,
        )
        self.nodes[root_node.node_id] = root_node
        self._node_judge_cache[root_node.node_id] = copy.deepcopy(root_judge)
        self._node_snapshot_cache[root_node.node_id] = copy.deepcopy(root_snapshot)
        self._node_rubric_round_cache[root_node.node_id] = copy.deepcopy(root_rubric_round)
        root_bundle = NodeArtifactBundle(
            node_id=root_node.node_id,
            node_dir=self.nodes_dir / root_node.node_id,
            node_payload=asdict(root_node),
            messages_payload=build_messages([], model_name=self.policy_model_name, preserve_token_fields=True),
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

    def _rubric_scope_specs(self, *, compare_parent: bool) -> list[dict[str, Any]]:
        specs = [
            {
                "scope": "siblings",
                "score_bank": self.score_banks["siblings"],
                "experience_bank": self.experience_banks["siblings"],
                "generation_prompt": SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT,
                "judge_prompt": SWE_TRAJECTORY_RUBRIC_JUDGE_PROMPT,
                "rubric_list_prefix": "rubric",
                "include_variance_reward": True,
            }
        ]
        if compare_parent:
            specs.append(
                {
                    "scope": "pc",
                    "score_bank": self.score_banks["pc"],
                    "experience_bank": self.experience_banks["pc"],
                    "generation_prompt": PC_TRAJECTORY_RUBRIC_GENERATION_PROMPT,
                    "judge_prompt": PC_TRAJECTORY_RUBRIC_JUDGE_PROMPT,
                    "rubric_list_prefix": "pc-rubric",
                    "include_variance_reward": False,
                }
            )
        return specs

    async def _run_rubric_scope_judging(
        self,
        *,
        scope_spec: dict[str, Any],
        question: dict[str, Any],
        shared_context: dict[str, Any],
        previous_state: dict[str, Any],
        latest_shared_segment: dict[str, Any] | None,
        continuations: list[dict[str, Any]],
        node_ids: list[str],
        round_index: int,
        generation_kwargs: dict[str, Any],
        judge_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        score_bank = scope_spec["score_bank"]
        experience_bank = scope_spec["experience_bank"]
        extra_prompt_sections: list[str] = []
        retrieved_experiences = []
        retrieve_messages: list[dict[str, Any]] = []
        active_bank_rubrics: list[RubricRecord] = []
        if score_bank is not None:
            score_context = score_bank.build_generation_context()
            active_bank_rubrics = copy.deepcopy(score_context.existing_rubrics)
            extra_prompt_sections.extend(score_context.extra_prompt_sections)
        if experience_bank is not None:
            experience_context = await experience_bank.build_generation_context(
                question=question,
                previous_state=previous_state,
                latest_shared_segment=latest_shared_segment,
                continuations=continuations,
                model_name=self.rubric_model_name,
                temperature=self.search_config.rubric_temperature,
                top_p=self.search_config.rubric_top_p,
                max_tokens=self.search_config.rubric_max_tokens,
                model_kwargs=self.rubric_model_kwargs,
            )
            extra_prompt_sections.extend(experience_context.extra_prompt_sections)
            retrieved_experiences = copy.deepcopy(experience_context.retrieved)
            retrieve_messages = copy.deepcopy(experience_context.retrieve_messages)
        extra_prompt_sections.extend(section for section in scope_spec.get("extra_prompt_sections", []) if section)

        active_score_result = None
        if active_bank_rubrics:
            active_score_result = await _score_round(
                question=question,
                shared_context=shared_context,
                continuations=continuations,
                rubrics=active_bank_rubrics,
                **judge_kwargs,
                judge_prompt=scope_spec["judge_prompt"],
            )

        score_batch = await _generate_and_score_rubric_batch(
            sample_count=self.search_config.n,
            round_index=round_index,
            generation_kwargs=generation_kwargs,
            generation_prompt=scope_spec["generation_prompt"],
            rubric_list_prefix=scope_spec["rubric_list_prefix"],
            question=question,
            shared_context=shared_context,
            continuations=continuations,
            extra_prompt_sections=extra_prompt_sections,
            judge_kwargs=judge_kwargs,
            judge_prompt=scope_spec["judge_prompt"],
            require_first_rubric=bool(scope_spec.get("require_first_rubric")),
        )

        rubric_samples: list[dict[str, Any]] = []
        valid_rubric_samples: list[dict[str, Any]] = []
        for sample_index, generated_sample in enumerate(score_batch["generated_samples"]):
            generated_rubrics = score_batch["sample_generated_rubrics"][sample_index]
            scoring_rubrics = active_bank_rubrics + generated_rubrics
            combined_score_batch = {
                "generated_score_results": {
                    sample_index: _combine_score_results(
                        active_score_result,
                        score_batch["generated_score_results"][sample_index],
                        continuation_count=len(continuations),
                    )
                },
            }
            evaluation = _rubric_sample_evaluation(
                score_batch=combined_score_batch,
                sample_index=sample_index,
                node_ids=node_ids,
                scoring_rubrics=scoring_rubrics,
                include_variance_reward=bool(scope_spec["include_variance_reward"]),
            )
            metrics = evaluation["metrics"]
            active_before = []
            active_after = []
            inactive_after = []
            if score_bank is not None:
                bank_update = score_bank.update_after_round(
                    generated=generated_rubrics,
                    rewards=metrics["reward_by_rubric"],
                )
                active_before = bank_update.active_before
                active_after = bank_update.active_after
                inactive_after = bank_update.inactive_after
            if scope_spec["scope"] == "pc":
                avg_scores = _pc_avg_scores_from_rubrics(
                    node_ids=node_ids,
                    score_lookup_by_node=evaluation["score_lookup_by_node"],
                    rubrics=scoring_rubrics,
                )
            else:
                avg_scores = _avg_scores_from_rubrics(
                    node_ids=node_ids,
                    score_lookup_by_node=evaluation["score_lookup_by_node"],
                    rubrics=scoring_rubrics,
                )
            scored_rubric_ids = {rubric.rubric_id for rubric in scoring_rubrics}
            sample_payload = {
                "scope": scope_spec["scope"],
                "sample_index": generated_sample.sample_index,
                "round_index": round_index,
                "rubric_list_id": generated_sample.rubric_list_id,
                "generated": generated_sample.generated,
                "messages": generated_sample.messages,
                "format_errors": copy.deepcopy(generated_sample.format_errors or []),
                "terminal_error": generated_sample.terminal_error,
                "generated_titles": [rubric.title for rubric in generated_sample.generated],
                **_rubric_metrics_payload(metrics, scored_rubric_ids),
                "average_rubric_judged_scores": avg_scores,
                "judge_errors": evaluation["judge_errors"],
                "selected": False,
            }
            if experience_bank is not None:
                sample_payload.update(
                    {
                        "retrieved": [asdict(experience) for experience in retrieved_experiences],
                        "retrieve_messages": copy.deepcopy(retrieve_messages),
                    }
                )
            if score_bank is not None:
                sample_payload.update(
                    {
                        "active_before": active_before,
                        "active_after": active_after,
                        "inactive_after": inactive_after,
                    }
                )
            rubric_samples.append(sample_payload)
            if scoring_rubrics:
                valid_rubric_samples.append(sample_payload)
        return {
            "scope": scope_spec["scope"],
            "samples": rubric_samples,
            "valid_samples": valid_rubric_samples,
        }

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
            update_payload = await _update_persistent_state(
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
            updated_parent_state = copy.deepcopy(update_payload["state"])
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
            "model_name": self.rubric_model_name,
            "temperature": self.search_config.rubric_temperature,
            "top_p": self.search_config.rubric_top_p,
            "max_tokens": self.search_config.rubric_max_tokens,
            "model_kwargs": self.rubric_model_kwargs,
        }
        rubric_judge_kwargs = {
            "model_name": self.judge_model_name,
            "temperature": self.search_config.judge_temperature,
            "top_p": self.search_config.judge_top_p,
            "max_tokens": self.search_config.judge_max_tokens,
            "model_kwargs": self.judge_model_kwargs,
        }
        node_ids = [branch["node_id"] for branch in branch_records]
        scope_results = await asyncio.gather(
            *[
                self._run_rubric_scope_judging(
                    scope_spec=scope_spec,
                    question=question,
                    shared_context=shared_context,
                    previous_state=updated_parent_state,
                    latest_shared_segment=latest_shared_segment,
                    continuations=continuations,
                    node_ids=node_ids,
                    round_index=round_index,
                    generation_kwargs=rubric_generation_kwargs,
                    judge_kwargs=rubric_judge_kwargs,
                )
                for scope_spec in self._rubric_scope_specs(compare_parent=compare_parent)
            ]
        )
        results_by_scope = {result["scope"]: result for result in scope_results}
        rubric_samples = results_by_scope["siblings"]["samples"]
        valid_rubric_samples = results_by_scope["siblings"]["valid_samples"]
        if compare_parent:
            pc_rubric_samples = results_by_scope["pc"]["samples"]
            valid_pc_rubric_samples = results_by_scope["pc"]["valid_samples"]
        else:
            pc_rubric_samples = None
            valid_pc_rubric_samples = None

        valid_rubric_samples = valid_rubric_samples or rubric_samples
        if compare_parent:
            valid_pc_rubric_samples = valid_pc_rubric_samples or pc_rubric_samples

        selected_sample = random.choice(valid_rubric_samples)
        selected_sample["selected"] = True
        selected_pc_sample = None
        if compare_parent:
            selected_pc_sample = random.choice(valid_pc_rubric_samples)
            selected_pc_sample["selected"] = True

        sibling_scores = [
            sum(sample["average_rubric_judged_scores"][branch["node_id"]] for sample in valid_rubric_samples) / len(valid_rubric_samples)
            for branch in branch_records
        ]
        if compare_parent:
            pc_scores = [
                sum(sample["average_rubric_judged_scores"][branch["node_id"]] for sample in valid_pc_rubric_samples) / len(valid_pc_rubric_samples)
                for branch in branch_records
            ]
        else:
            pc_scores = None
        return {
            "rubric_samples": rubric_samples,
            "pc_rubric_samples": pc_rubric_samples,
            "siblings_scores": sibling_scores,
            "pc_scores": pc_scores,
            "selected_sample_index": selected_sample["sample_index"],
            "selected_sample": selected_sample,
            "selected_pc_sample": selected_pc_sample,
        }

    def _rubric_bank_payload_fields(self, sample: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if "retrieved" in sample or "retrieve_messages" in sample:
            payload.update(
                {
                    "retrieved": copy.deepcopy(sample.get("retrieved", [])),
                    "retrieve_messages": copy.deepcopy(sample.get("retrieve_messages")),
                }
            )
        if "active_before" in sample and "active_after" in sample and "inactive_after" in sample:
            payload.update(
                {
                    "active_bank_before": [asdict(rubric) for rubric in sample["active_before"]],
                    "active_bank_after": [asdict(rubric) for rubric in sample["active_after"]],
                    "inactive_bank_after": [asdict(rubric) for rubric in sample["inactive_after"]],
                }
            )
        return payload

    def _rubric_round_payload(self, sample: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "rubric_list_id": sample["rubric_list_id"],
            "round_index": sample["round_index"],
            "generated": [asdict(rubric) for rubric in sample["generated"]],
            "format_errors": copy.deepcopy(sample.get("format_errors", [])),
            "terminal_error": sample.get("terminal_error"),
            "variance_by_rubric": copy.deepcopy(sample["variance_by_rubric"]),
            "redundency_by_rubric": copy.deepcopy(sample["redundency_by_rubric"]),
            "judge_error_by_rubric": copy.deepcopy(sample["judge_error_by_rubric"]),
            "reward_by_rubric": copy.deepcopy(sample["reward_by_rubric"]),
            "judge_errors": copy.deepcopy(sample["judge_errors"]),
        }
        payload.update(self._rubric_bank_payload_fields(sample))
        return payload

    def _rubric_artifact_payload(self, sample: dict[str, Any], parent_id: str) -> dict[str, Any]:
        payload = {
            **self._rubric_round_payload(sample),
            "parent_node_id": parent_id,
            "score_by_rubric": copy.deepcopy(sample["score_by_rubric"]),
            "average_rubric_judged_scores": copy.deepcopy(sample["average_rubric_judged_scores"]),
            "generated_titles": list(sample["generated_titles"]),
            "selected": bool(sample["selected"]),
            "gt_by_rubric": {},
            "reward": 0.0,
        }
        return payload

    def _run_round(self, parent_id: str, round_index: int) -> list[dict[str, Any]]:
        parent_node = self.nodes[parent_id]
        if parent_id not in self._node_snapshot_cache:
            raise RuntimeError(f"Node {parent_id} does not have a cached restorable snapshot")
        if parent_id not in self._node_judge_cache:
            raise RuntimeError(f"Node {parent_id} does not have a cached judge payload")
        parent_snapshot = copy.deepcopy(self._node_snapshot_cache[parent_id])
        parent_judge = copy.deepcopy(self._node_judge_cache[parent_id])
        parent_rubric_round = copy.deepcopy(self._node_rubric_round_cache.get(parent_id))
        if self.rubric_bank is not None:
            active_bank = [RubricRecord(**rubric) for rubric in parent_rubric_round.get("active_bank_after", [])]
            inactive_bank = [RubricRecord(**rubric) for rubric in parent_rubric_round.get("inactive_bank_after", [])]
            self.rubric_bank.set_state(active_bank=active_bank, inactive_bank=inactive_bank)
        if self.pc_rubric_bank is not None:
            pc_active_bank = [RubricRecord(**rubric) for rubric in parent_rubric_round.get("pc_active_bank_after", [])]
            pc_inactive_bank = [RubricRecord(**rubric) for rubric in parent_rubric_round.get("pc_inactive_bank_after", [])]
            self.pc_rubric_bank.set_state(active_bank=pc_active_bank, inactive_bank=pc_inactive_bank)
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

        # Run the m sibling branches in parallel using a thread pool.
        # Each thread owns its own docker session; the underlying sglang server
        # batches their token generations naturally for higher GPU utilization.
        def _run_one_branch(sample_index_and_plan):
            sample_index, (policy_source, policy_model_name) = sample_index_and_plan
            node_id = f"node-r{round_index:03d}-s{sample_index:02d}-{uuid.uuid4().hex[:6]}"
            _t_br_start = time.perf_counter()
            try:
                logging.getLogger("swe_agent.trajectory_search").info(
                    "[TIMING] branch.start round=%d sample=%d node=%s k=%d"
                    % (round_index, sample_index, node_id, self.search_config.k)
                )
            except Exception:
                pass
            resumed_snapshot = copy.deepcopy(parent_snapshot)
            resumed_snapshot["session_id"] = f"{node_id}-session"
            resumed_snapshot["spec"]["session_id"] = resumed_snapshot["session_id"]
            # When the model class is litellm_textbased (openai backend in
            # search_swe_agent.py), litellm.completion() needs a provider
            # prefix on the model name: bare "Qwen/Qwen3.5-9B" raises
            # BadRequestError ("LLM Provider NOT provided"). For local
            # OpenAI-compatible sglang servers (which is what we're hitting
            # via the api_base in model_kwargs), "openai/<name>" is the
            # right prefix and is a no-op when already present.
            _eff_model_name = policy_model_name
            for _prefix in ("openai/", "azure/", "anthropic/", "huggingface/", "hosted_vllm/"):
                if _eff_model_name.startswith(_prefix):
                    break
            else:
                _eff_model_name = "openai/" + _eff_model_name
            resumed_snapshot["spec"]["policy_ref"] = _eff_model_name
            resumed_snapshot["spec"]["policy_version"] = _eff_model_name
            resumed_snapshot["model"]["config"]["model_name"] = _eff_model_name
            resumed_snapshot["model"]["config"]["route_name"] = "policy_student" if policy_source == "student" else "policy"
            # Per-branch api_base swap: RouteTextbasedModel.query (token-in/
            # token-out path against sglang /generate) reads api_base from
            # model_kwargs, NOT from the route config. If teacher and student
            # are on different sglang servers (e.g. DSv4 vs Qwen3.5-9B on
            # separate GCP nodes), each branch must point at the right one.
            # Env vars set by the orchestrator:
            #   SEARCH_SWE_TEACHER_API_BASE/_API_KEY   (teacher endpoint)
            #   SEARCH_SWE_STUDENT_API_BASE/_API_KEY   (student endpoint)
            # If unset, leaves the inherited model_kwargs alone (current
            # behavior).
            _mk = resumed_snapshot["model"]["config"].setdefault("model_kwargs", {})
            if policy_source == "student":
                _ab = os.environ.get("SEARCH_SWE_STUDENT_API_BASE")
                _ak = os.environ.get("SEARCH_SWE_STUDENT_API_KEY", "EMPTY")
            else:
                _ab = os.environ.get("SEARCH_SWE_TEACHER_API_BASE")
                _ak = os.environ.get("SEARCH_SWE_TEACHER_API_KEY", "EMPTY")
            if _ab:
                _mk["api_base"] = _ab if _ab.endswith("/v1") else _ab.rstrip("/") + "/v1"
                _mk["api_key"] = _ak
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
                rec = {
                    "node_id": node_id,
                    "result": result,
                    "policy_source": policy_source,
                    "policy_model_name": policy_model_name,
                    "error": error_text,
                    "prompt_payload": {"messages": copy.deepcopy(resumed_snapshot.get("agent").get("state").get("messages"))},
                    "message_payload": build_messages(message_payload, model_name=policy_model_name, preserve_token_fields=True),
                    "is_valid": False,
                }
                self._dispose_session(session)
                try:
                    _step_n = (rec.get("step_end", 0) or 0) - (rec.get("step_start", 0) or 0)
                    logging.getLogger("swe_agent.trajectory_search").info(
                        "[TIMING] branch.end   round=%d sample=%d node=%s took=%.1fs steps=%d (ERR)"
                        % (round_index, sample_index, node_id, time.perf_counter()-_t_br_start, _step_n)
                    )
                except Exception:
                    pass
                return rec, error_text
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
            rec = {
                "node_id": node_id,
                "session": session,
                "result": result,
                "snapshot_after": snapshot_after,
                "policy_source": policy_source,
                "policy_model_name": policy_model_name,
                "workspace_meta": workspace_meta,
                "prompt_payload": {"messages": copy.deepcopy(resumed_snapshot.get("agent").get("state").get("messages"))},
                "message_payload": build_messages(message_payload, model_name=policy_model_name, preserve_token_fields=True),
                "step_start": step_start,
                "step_end": step_end,
                "recent_segments": recent_segments,
                "is_valid": True,
            }
            try:
                _step_n = (rec.get("step_end", 0) or 0) - (rec.get("step_start", 0) or 0)
                logging.getLogger("swe_agent.trajectory_search").info(
                    "[TIMING] branch.end   round=%d sample=%d node=%s took=%.1fs steps=%d"
                    % (round_index, sample_index, node_id, time.perf_counter()-_t_br_start, _step_n)
                )
            except Exception:
                pass
            return rec, None

        with ThreadPoolExecutor(max_workers=max(1, len(sample_plan)), thread_name_prefix="search-branch") as _branch_pool:
            ordered_results = list(_branch_pool.map(_run_one_branch, enumerate(sample_plan)))
        for rec, err_text in ordered_results:
            branch_records.append(rec)
            if err_text is not None:
                policy_generation_errors.append({"node_id": rec["node_id"], "error": err_text})
            elif rec["is_valid"]:
                valid_branch_records.append(rec)

        if len(valid_branch_records) < 2:
            self.peaceful_exit(branch_records, parent_id)
            return []

        judged = asyncio.run(
            self._prepare_round_judging(
                parent_node=parent_node,
                parent_judge=parent_judge,
                branch_records=valid_branch_records,
                round_index=round_index,
                compare_parent=parent_id != "root",
            )
        )
        node_round_records = []
        valid_branches: list[dict[str, Any]] = []
        pc_scores = judged.get("pc_scores")
        pc_regression_threshold = 0.5 + self.search_config.regression_margin
        for branch, sibling_score, pc_score in zip(valid_branch_records, judged["siblings_scores"], pc_scores if pc_scores is not None else [None] * len(valid_branch_records)):
            branch["score"] = float(sibling_score)
            branch["regressed_vs_parent"] = parent_id != "root" and pc_score is not None and pc_score < pc_regression_threshold
            valid_branches.append(branch)
            node_round_records.append({
                "node_id": branch["node_id"],
                "score": branch["score"],
                "siblings_score": branch["score"],
                "pc_score": float(pc_score) if pc_score is not None else None,
                "regressed_vs_parent": branch["regressed_vs_parent"],
            })
        valid_branches.sort(key=lambda branch: (branch["score"], branch["result"]["status"] == "finished"), reverse=True)
        frontier_branches = (
            valid_branches
            if parent_id == "root" or pc_scores is None
            else [branch for branch in valid_branches if not branch["regressed_vs_parent"]]
        )
        chosen_branches = _sample_by_strategy(
            frontier_branches,
            [branch["score"] for branch in frontier_branches],
            count=self.search_config.p,
            strategy=self.search_config.strategy,
        )
        top_child_ids = [branch["node_id"] for branch in chosen_branches]
        regressed = parent_id != "root" and pc_scores is not None and not top_child_ids
        candidate_frontier_ids = list(top_child_ids)
        candidate_frontier_ids.extend(
            node_id for node_id in previous_frontier if node_id != parent_id and node_id not in candidate_frontier_ids
        )
        kept_child_ids = set(top_child_ids)
        if regressed:
            self.best_node_id = parent_id

        round_rubric_path = self.rubrics_dir / f"round_{round_index:03d}.json"
        rubric_sample_payloads = []
        rubric_update_records = []
        rubric_artifact_bundles: list[RubricArtifactBundle] = []
        for sample in judged["rubric_samples"]:
            rubric_dir = self.rubrics_dir / "siblings" / sample["rubric_list_id"]
            rubric_payload = self._rubric_artifact_payload(sample, parent_id)
            rubric_sample_payloads.append(rubric_payload)
            rubric_update_records.append(
                {
                    "scope": "siblings",
                    "round_index": round_index,
                    "rubric_payload": rubric_payload,
                    "messages": sample["messages"],
                }
            )
            rubric_artifact_bundles.append(
                RubricArtifactBundle(
                    rubric_dir=rubric_dir,
                    rubric_payload=rubric_payload,
                    messages_payload=sample["messages"],
                    retrieve_messages_payload=rubric_payload.get("retrieve_messages"),
                    round_summary_path=round_rubric_path,
                    selected_for_round_summary=bool(sample["selected"]),
                )
            )
        pc_rubric_sample_payloads = []
        for sample in judged.get("pc_rubric_samples") or []:
            rubric_dir = self.rubrics_dir / "pc" / sample["rubric_list_id"]
            rubric_payload = self._rubric_artifact_payload(sample, parent_id)
            pc_rubric_sample_payloads.append(rubric_payload)
            rubric_update_records.append(
                {
                    "scope": "pc",
                    "round_index": round_index,
                    "rubric_payload": rubric_payload,
                    "messages": sample["messages"],
                }
            )
            rubric_artifact_bundles.append(
                RubricArtifactBundle(
                    rubric_dir=rubric_dir,
                    rubric_payload=rubric_payload,
                    messages_payload=sample["messages"],
                    retrieve_messages_payload=rubric_payload.get("retrieve_messages"),
                    scope="pc",
                )
            )

        selected_sample = judged["selected_sample"]
        round_payload = {
            **self._rubric_artifact_payload(selected_sample, parent_id),
            "round_index": round_index,
            "parent_id": parent_id,
            "selected_sample_index": judged["selected_sample_index"],
            "selected_rubric_list_id": selected_sample["rubric_list_id"],
            "regressed": regressed,
            "node_scores": node_round_records,
            "policy_generation_errors": policy_generation_errors,
            "rubric_samples": rubric_sample_payloads,
        }
        if self.rubric_bank is not None:
            self.rubric_bank.set_state(
                active_bank=copy.deepcopy(selected_sample["active_after"]),
                inactive_bank=copy.deepcopy(selected_sample["inactive_after"]),
            )
        selected_pc_sample = judged.get("selected_pc_sample", None)
        if selected_pc_sample is not None:
            self.pc_rubric_bank.set_state(
                active_bank=copy.deepcopy(selected_pc_sample["active_after"]),
                inactive_bank=copy.deepcopy(selected_pc_sample["inactive_after"]),
            )
            round_payload.update(
                {
                    "pc_rubric_samples": pc_rubric_sample_payloads,
                    "pc_active_bank_after": [asdict(rubric) for rubric in self.pc_rubric_bank.active_bank],
                    "pc_inactive_bank_after": [asdict(rubric) for rubric in self.pc_rubric_bank.inactive_bank],
                }
            )
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
                "overall_score": branch["score"],
                "regressed_vs_parent": branch["regressed_vs_parent"],
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
                    preserve_token_fields=True,
                )
                _patch = _normalize_terminal_patch_text(terminal_result.get("submission", ""))
                if not _patch:
                    # Agent didn't explicitly submit (ran out of step budget mid-edit).
                    # Fall back to the actual git diff inside the docker container so we
                    # don't lose real intermediate work. Run from the agent's current
                    # working directory rather than hardcoding /testbed — ReBench docker
                    # images don't mount the repo at /testbed.
                    try:
                        _diff = branch["session"].agent.env.execute(
                            {"command": "git add -N . >/dev/null 2>&1; git diff"},
                            timeout=30,
                        )
                        _patch = _normalize_terminal_patch_text(_diff.get("output") or "")
                    except Exception:
                        _patch = ""
                terminal_patch = {
                    self.task_id: {
                        "model_name_or_path": branch["policy_model_name"],
                        "instance_id": self.task_id,
                        "model_patch": _patch,
                        "terminal_error": terminal_error,
                    }
                }
            snapshot_payload = (
                _resume_snapshot_payload(
                    branch["snapshot_after"],
                    image_tag=branch["image_tag"],
                    image_id=branch["image_id"],
                )
                if branch["keep_snapshot"]
                else None
            )
            self.nodes[node.node_id] = node
            self._node_judge_cache[node.node_id] = copy.deepcopy(judge_payload)
            self._node_rubric_round_cache[node.node_id] = copy.deepcopy(round_payload)
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
        gt_extra_json_writes: list[tuple[Path, Any]] = [(round_rubric_path, round_payload)]
        self.nodes[parent_id].status = "archived"
        non_gt_extra_json_writes = [(self.nodes_dir / parent_id/ "node.json", asdict(self.nodes[parent_id]))]
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
        return rubric_update_records

        try:
            logging.getLogger("swe_agent.trajectory_search").info(
                "[TIMING] run_round.end   round=%d took=%.1fs frontier=%d best=%s"
                % (round_index, time.perf_counter()-_t_round_start, len(self.frontier_ids), str(self.best_node_id))
            )
        except Exception:
            pass

    def _finalize_outputs(self) -> None:
        def _cached_overall_score(node_id: str) -> float:
            reward = (self._node_judge_cache.get(node_id) or {}).get("overall_score")
            return float(reward) if reward is not None else float("-inf")

        if self.finished_node_ids:
            final_node_id = max(
                (node_id for node_id in self.finished_node_ids if node_id in self.nodes),
                key=_cached_overall_score,
            )
        elif self.frontier_ids:
            final_node_id = max(
                (node_id for node_id in self.frontier_ids if node_id in self.nodes),
                key=_cached_overall_score,
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
                    preserve_token_fields=True,
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
                        make_evaluation_payload("empty"),
                    )
            else:
                try:
                    evaluation = evaluate_swebench_instance_patches(
                        instance=self.instance,
                        patches_by_key={final_node_id: patch_text},
                        model_name=final_policy_model_name,
                        max_workers=1,
                        namespace=self.harness_namespace,
                        work_dir=self.run_dir,
                    )[final_node_id]
                except Exception as exc:
                    evaluation = make_evaluation_payload("error", error=exc)
                if self.search_config.write_artifacts:
                    _atomic_write_json(
                        self.run_dir / "evaluation.json",
                        evaluation,
                    )
        self.best_node_id = final_node_id
        if final_node.status == "finished":
            self.frontier_ids = [final_node_id]
        self._save_manifest()

    def peaceful_exit(self, branch_records, parent_id: str) -> None:
        self.peaceful_exit_triggered = True
        error_records = [
            {"node_id": branch.get("node_id"), "error": branch.get("error")}
            for branch in branch_records
            if branch.get("error")
        ]
        logger.warning(
            "Peaceful exit for parent %s with %d branches (%d valid). Errors: %s",
            parent_id,
            len(branch_records),
            sum(1 for branch in branch_records if branch.get("is_valid")),
            error_records,
        )
        for branch in branch_records:
            if branch.get("session") is not None:
                self._dispose_session(branch["session"])
                branch["session"] = None
        self.nodes[parent_id].status = "archived"
        non_gt_extra_json_writes = [(self.nodes_dir / parent_id / "node.json", asdict(self.nodes[parent_id]))]
        self.artifact_writer.submit_round(
            extra_json_writes=non_gt_extra_json_writes,
            write_gt_files=False,
        )
        self.frontier_ids = [node_id for node_id in self.frontier_ids if node_id != parent_id]
        self.best_node_id = self.frontier_ids[0] if self.frontier_ids else parent_id
