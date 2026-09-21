from __future__ import annotations

import copy
from typing import Any, Callable

from agent_rl.run_utils import (
    extract_last_json_object,
    freeform_thought_model_kwargs,
    run_litellm_completion_async,
)

from .config import MODEL_COMPLETION_TOKENS


async def litellm_message(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    route_name: str,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every experience-generation inference through the shared LiteLLM path."""

    kwargs = copy.deepcopy(model_kwargs or {})
    if model.startswith("openai/"):
        # The configured model is already a complete LiteLLM route such as
        # openai/azure/openai/gpt-5.6-sol. Preserve it exactly so experience
        # generation uses the same effective endpoint as the rubric pipeline.
        kwargs.pop("extra_body", None)
        kwargs["reasoning_effort"] = "high"
    else:
        kwargs = freeform_thought_model_kwargs(kwargs)
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    completion = await run_litellm_completion_async(
        model_name=model,
        messages=messages,
        usage_model_role=route_name,
        max_tokens=max_tokens,
        **kwargs,
    )
    error = completion.metadata.get("error")
    if error is not None:
        raise RuntimeError(f"{route_name} LiteLLM completion failed: {error}")
    return {
        "role": "assistant",
        "content": completion.content,
        "content_no_thinking": completion.metadata.get(
            "content_no_thinking", completion.content
        ),
        "usage": completion.usage,
        "finish_reason": completion.finish_reason,
    }


class InvalidJsonResponse(ValueError):
    """Raised after every configured token budget failed schema validation."""

    def __init__(self, message: str, messages: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.messages = copy.deepcopy(messages)


class JsonModelClient:
    """OpenAI-compatible model client using the same routing as trajectory search."""

    def __init__(self, *, top_p: float = 0.95, temperature: float = 0.1) -> None:
        self.top_p = top_p
        self.temperature = temperature

    async def call(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        retry_max_tokens: int | None = None,
        validator: Callable[[dict[str, Any]], bool] | None = None,
        route_name: str = "experience_generation",
        model_kwargs: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        for name, value in (
            ("max_tokens", max_tokens),
            ("retry_max_tokens", retry_max_tokens),
        ):
            if value is not None and not 1 <= value <= MODEL_COMPLETION_TOKENS:
                raise ValueError(
                    f"{name} must be between 1 and {MODEL_COMPLETION_TOKENS}"
                )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        budgets = [max_tokens]
        if retry_max_tokens:
            budgets.append(retry_max_tokens)
        last_error = "No valid JSON object was returned."
        for index, budget in enumerate(budgets):
            assistant = await litellm_message(
                model=model,
                messages=messages,
                max_tokens=budget,
                temperature=self.temperature,
                top_p=self.top_p,
                route_name=route_name,
                model_kwargs=model_kwargs,
            )
            messages.append(copy.deepcopy(assistant))
            content = assistant.get("content_no_thinking") or assistant.get("content") or ""
            parsed = extract_last_json_object(content)
            if parsed is None and content != (assistant.get("content") or ""):
                parsed = extract_last_json_object(assistant.get("content") or "")
            if isinstance(parsed, dict) and (validator is None or validator(parsed)):
                return parsed, messages
            finish_reason = str(assistant.get("finish_reason") or "unknown")
            last_error = (
                "Model response did not contain an object matching the requested "
                f"schema (finish_reason={finish_reason}, token_budget={budget})."
            )
            if index + 1 < len(budgets):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            last_error
                            + " Do not repeat the analysis. Append one concise, complete "
                            "JSON object matching the schema, with non-empty title, "
                            "description, context, experience, and metadata fields."
                        ),
                    }
                )
        raise InvalidJsonResponse(last_error, messages)
