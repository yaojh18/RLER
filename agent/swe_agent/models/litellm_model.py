import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import litellm
from pydantic import BaseModel

from swe_agent.exceptions import FormatError
from swe_agent.models import GLOBAL_MODEL_STATS
from swe_agent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from swe_agent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks
from swe_agent.models.utils.cache_control import set_cache_control
from swe_agent.models.utils.openai_multimodal import expand_multimodal_content
from swe_agent.models.utils.retry import retry

logger = logging.getLogger("litellm_model")
DEFAULT_LITELLM_TIMEOUT_SECONDS = float(os.getenv("MSWEA_LITELLM_TIMEOUT", "300"))


def _message_contents(message: Any) -> tuple[str, str]:
    if isinstance(message, dict):
        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", None)
        if reasoning_content is None:
            reasoning_content = message.get("reasoning", None)
        provider_specific_fields = message.get("provider_specific_fields", None)
    else:
        content = getattr(message, "content", "")
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content is None:
            reasoning_content = getattr(message, "reasoning", None)
        provider_specific_fields = getattr(message, "provider_specific_fields", None)
    if reasoning_content is None and isinstance(provider_specific_fields, dict):
        reasoning_content = provider_specific_fields.get("reasoning_content")
    if reasoning_content:
        full_content = f"<think>{reasoning_content}</think>\n{content}" if content else f"<think>{reasoning_content}</think>"
        return full_content, content
    if isinstance(content, str) and content.startswith("<think>"):
        closing_tag = "</think>"
        closing_index = content.find(closing_tag)
        if closing_index >= 0:
            return content, content[closing_index + len(closing_tag) :].lstrip("\r\n")
    return content, content


def _build_assistant_message(
    *,
    content: str | None,
    content_no_thinking: str | None,
    usage: dict[str, Any] | None,
    prompt_token_ids: list[int] | None,
    token_ids: list[int] | None,
    logprobs: list[float] | None,
    cost: float,
    timestamp: float,
    finish_reason: str | None,
) -> dict[str, Any]:
    """Build the canonical assistant-message schema shared by all text backends."""
    return {
        "role": "assistant",
        "content": content or "",
        "content_no_thinking": content_no_thinking or "",
        "usage": dict(usage or {}),
        "prompt_token_ids": list(prompt_token_ids or []),
        "token_ids": list(token_ids or []),
        "logprobs": list(logprobs or []),
        "extra": {
            "actions": [],
            "cost": cost,
            "timestamp": timestamp,
            "finish_reason": finish_reason or "stop",
        },
    }


class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name. Highly recommended to include the provider in the model name, e.g., `anthropic/claude-sonnet-4-5-20250929`."""
    model_kwargs: dict[str, Any] = {}
    """Additional arguments passed to the API."""
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    """Model registry for cost tracking and model metadata. See the local model guide (https://mini-swe-agent.com/latest/models/local_models/) for more details."""
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class LitellmModel:
    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.BadRequestError,
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def _completion_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        merged = self.config.model_kwargs | kwargs
        if DEFAULT_LITELLM_TIMEOUT_SECONDS > 0 and "timeout" not in merged and "request_timeout" not in merged:
            merged["timeout"] = DEFAULT_LITELLM_TIMEOUT_SECONDS
        merged.setdefault("stream", False)
        if (
            DEFAULT_LITELLM_TIMEOUT_SECONDS > 0
            and "client" not in merged
            and any(name in self.config.model_name.lower() for name in ("gemini", "vertex"))
        ):
            from litellm.llms.custom_httpx.http_handler import HTTPHandler

            merged["client"] = HTTPHandler(timeout=DEFAULT_LITELLM_TIMEOUT_SECONDS)
        return merged

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[BASH_TOOL],
                **self._completion_kwargs(kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _prepare_messages_for_api(
        self, messages: list[dict], *, preserve_token_fields: bool = False
    ) -> list[dict]:
        internal_fields = {"extra", "content_no_thinking", "usage", "tokens"}
        if not preserve_token_fields:
            internal_fields.update({"prompt_token_ids", "token_ids", "logprobs"})
        prepared = []
        for message in messages:
            item = {key: value for key, value in message.items() if key not in internal_fields}
            actions = (message.get("extra") or {}).get("actions") or []
            tool_calls = [
                {
                    "id": action["tool_call_id"],
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": action["command"]})},
                }
                for action in actions
                if action.get("tool_call_id") and "command" in action
            ]
            if tool_calls and "tool_calls" not in item:
                item["tool_calls"] = tool_calls
            prepared.append(item)
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions, model_name=self.config.model_name):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        choice = response.choices[0]
        choice_message = choice.message
        full_content, content_no_thinking = _message_contents(choice_message)
        usage_obj = getattr(response, "usage", None)
        usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else dict(usage_obj or {})
        logprob_entries = getattr(getattr(choice, "logprobs", None), "content", None) or []
        logprobs = []
        for entry in logprob_entries:
            value = entry.get("logprob") if isinstance(entry, dict) else getattr(entry, "logprob", None)
            if value is not None:
                logprobs.append(float(value))
        message = _build_assistant_message(
            content=full_content,
            content_no_thinking=content_no_thinking,
            usage=usage,
            prompt_token_ids=getattr(response, "prompt_token_ids", None),
            token_ids=getattr(response, "output_token_ids", None),
            logprobs=logprobs,
            cost=cost_output["cost"],
            timestamp=time.time(),
            finish_reason=choice.finish_reason,
        )
        try:
            message["extra"]["actions"] = self._parse_actions(response)
        except FormatError as exc:
            message["extra"]["format_error"] = True
            setattr(exc, "assistant_message", message)
            raise
        return message

    def _calculate_cost(self, response) -> dict[str, float]:
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        return {"cost": cost}

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(tool_calls, format_error_template=self.config.format_error_template)

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        del state
