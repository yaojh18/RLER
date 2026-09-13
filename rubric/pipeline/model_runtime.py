#!/usr/bin/env python3
"""Low-temperature hosted-model calls with complete request/response artifacts."""

from __future__ import annotations

import copy
import json
import os
from typing import Any

from agent_rl.run_utils import extract_last_json_object, route_completion_message
from swe_agent.run.run_swe_agent import _litellm_model_kwargs


API_BASE = "https://inference-api.nvidia.com/v1"


def model_kwargs(model: str) -> dict[str, Any]:
    api_key = os.environ.get("RLER_MODEL_API_KEY", "")
    if not api_key:
        raise RuntimeError("RLER_MODEL_API_KEY is not set")
    kwargs = copy.deepcopy(_litellm_model_kwargs(model))
    kwargs.update(
        {
            "api_base": API_BASE,
            "api_key": api_key,
            "completion_backend": "litellm",
        }
    )
    return kwargs


async def call_json(
    *,
    route_name: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.02,
    max_tokens: int = 20_480,
    correction_rounds: int = 3,
) -> dict[str, Any]:
    conversation = copy.deepcopy(messages)
    errors: list[str] = []
    for _ in range(correction_rounds):
        response = await route_completion_message(
            route_name=route_name,
            model_name=model,
            messages=conversation,
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs(model),
        )
        conversation.append(response)
        text = response.get("content_no_thinking") or response.get("content") or ""
        parsed = extract_last_json_object(text)
        if parsed is None and response.get("content") != text:
            parsed = extract_last_json_object(response.get("content") or "")
        if isinstance(parsed, dict):
            return {"parsed": parsed, "messages": conversation, "format_errors": errors}
        errors.append("no JSON object found in model response")
        conversation.append(
            {
                "role": "user",
                "content": (
                    "Return the requested single valid JSON object now. "
                    "Do not add prose after it."
                ),
            }
        )
    raise ValueError(json.dumps({"errors": errors, "messages": conversation}, ensure_ascii=False))
