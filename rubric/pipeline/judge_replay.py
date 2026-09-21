"""Frozen Luna rubric-judge adapter used by automatic generation/refinement."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

from RLER.rubric.pipeline.prompts import HANDBOOK_RUBRIC_JUDGE_PROMPT

JUDGE_TEMPERATURE = 0.02
JUDGE_TOP_P = 1.0


def parse_generation_view(payload: dict[str, Any]) -> dict[str, Any]:
    messages = (payload.get("generation") or {}).get("messages") or []
    first_user = next(
        (
            message.get("content")
            for message in messages
            if message.get("role") == "user"
            and "## Agent Trajectory Continuations:" in str(message.get("content") or "")
        ),
        None,
    )
    if not isinstance(first_user, str):
        raise ValueError("adaptive generation prompt is unavailable")
    markers = {
        "question": "## Question:\n",
        "previous": "\n## Previous Persistent State:\n",
        "parent": "\n## Parent Trajectory:\n",
        "continuations": "\n## Agent Trajectory Continuations:\n",
        "handbook": "\n\n## Instance-level privileged handbook\n",
    }
    question_start = first_user.index(markers["question"]) + len(markers["question"])
    previous_start = first_user.index(markers["previous"], question_start)
    parent_start = first_user.index(markers["parent"], previous_start + 1)
    continuations_start = first_user.index(markers["continuations"], parent_start + 1)
    handbook_start = first_user.index(markers["handbook"], continuations_start + 1)
    continuation_section = first_user[
        continuations_start + len(markers["continuations"]) : handbook_start
    ]
    rendered = []
    for index in range(1, 9):
        marker = f"## Continuation {index}:\n"
        start = continuation_section.index(marker) + len(marker)
        end = (
            continuation_section.index(f"\n\n## Continuation {index + 1}:\n", start)
            if index < 8
            else len(continuation_section)
        )
        rendered.append(continuation_section[start:end].rstrip("\n"))
    return {
        "question_text": first_user[question_start:previous_start],
        "previous_state": first_user[previous_start + len(markers["previous"]) : parent_start],
        "parent": first_user[parent_start + len(markers["parent"]) : continuations_start],
        "continuations": rendered,
    }


def judge_request_content(
    judge_prompt: str,
    view: dict[str, Any],
    node_index: int,
    criterion: str,
) -> str:
    return "".join(
        [
            judge_prompt.strip(),
            f"\n\n## Question:\n{view['question_text']}\n",
            f"## Previous Persistent State:\n{view['previous_state']}\n",
            f"## Parent Trajectory:\n{view['parent']}\n",
            f"## Continuation Trajectory:\n{view['continuations'][node_index]}\n",
            f"## Criterion:\n{criterion}",
        ]
    )


async def judge_call(
    *,
    request: list[dict[str, str]],
    semaphore: asyncio.Semaphore,
    model_name: str,
    max_tokens: int,
) -> dict[str, Any]:
    from agent_rl.run_utils import freeform_thought_model_kwargs, route_completion_message
    from RLER.rubric.pipeline.model_runtime import model_kwargs

    conversation = copy.deepcopy(request)
    format_errors = []
    async with semaphore:
        for _ in range(8):
            response = await route_completion_message(
                route_name="rubric_judge",
                model_name=model_name,
                messages=conversation,
                temperature=JUDGE_TEMPERATURE,
                top_p=JUDGE_TOP_P,
                max_tokens=max_tokens,
                model_kwargs=freeform_thought_model_kwargs(model_kwargs(model_name)),
            )
            conversation.append(response)
            text = response.get("content_no_thinking") or response.get("content") or ""
            try:
                parsed = json.loads(text[text.index("{") : text.rindex("}") + 1])
            except (ValueError, json.JSONDecodeError):
                parsed = None
            if (
                isinstance(parsed, dict)
                and isinstance(parsed.get("evidence"), str)
                and isinstance(parsed.get("score"), int)
                and 1 <= parsed["score"] <= 5
            ):
                return {
                    "score_raw": parsed["score"],
                    "evidence": parsed["evidence"],
                    "messages": conversation[1:],
                    "format_errors": format_errors,
                }
            format_errors.append("invalid judge JSON")
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        "Return exactly one JSON object with string evidence "
                        "and integer score from 1 through 5."
                    ),
                }
            )
    raise ValueError("judge format correction exhausted")
