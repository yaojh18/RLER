from __future__ import annotations

from typing import Any

from agent_rl import ChatCompletion, ChatSamplingParams
from openai import AsyncOpenAI, OpenAI


class SGLangChatService:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "EMPTY",
        default_model_name: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        normalized_base_url = base_url.rstrip("/")
        if not normalized_base_url.endswith("/v1"):
            normalized_base_url = f"{normalized_base_url}/v1"
        self.default_model_name = default_model_name
        self._base_url = normalized_base_url
        self._api_key = api_key
        self._timeout = timeout
        self._client = OpenAI(base_url=normalized_base_url, api_key=api_key, timeout=timeout)

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                key: value
                for key, value in message.items()
                if key not in {"metadata", "source", "trainable", "extra", "content_no_thinking"}
            }
            for message in messages
        ]

    def _completion_kwargs(self, sampling: ChatSamplingParams) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
        }
        if sampling.stop:
            kwargs["stop"] = sampling.stop
        if sampling.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        for key, value in sampling.extra.items():
            if key in {
                "frequency_penalty",
                "presence_penalty",
                "n",
                "seed",
                "top_k",
                "min_p",
                "logprobs",
                "extra_body",
                "response_format",
            }:
                kwargs[key] = value
        return kwargs

    def _to_completion(self, response, *, model_name: str | None, policy_version: str | None) -> ChatCompletion:
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content is None:
            reasoning_content = getattr(message, "reasoning", None)
        if reasoning_content is None:
            provider_specific_fields = getattr(message, "provider_specific_fields", None)
            if isinstance(provider_specific_fields, dict):
                reasoning_content = provider_specific_fields.get("reasoning_content")
        if reasoning_content:
            content = f"<think>{reasoning_content}</think>\n{content}" if content else f"<think>{reasoning_content}</think>"
        return ChatCompletion(
            content=content,
            finish_reason=choice.finish_reason or "stop",
            model_name=model_name or self.default_model_name,
            usage=response.usage.model_dump() if getattr(response, "usage", None) is not None else {},
            raw_response=response.model_dump(),
            metadata={
                "requested_policy_version": policy_version,
                "content_no_thinking": message.content or "",
            },
        )

    def generate_text(
        self,
        *,
        messages: list[dict[str, Any]],
        model_name: str | None,
        sampling: ChatSamplingParams,
        policy_version: str | None = None,
    ) -> ChatCompletion:
        response = self._client.chat.completions.create(
            model=model_name or self.default_model_name,
            messages=self._normalize_messages(messages),
            **self._completion_kwargs(sampling),
        )
        return self._to_completion(response, model_name=model_name, policy_version=policy_version)

    async def agenerate_text(
        self,
        *,
        messages: list[dict[str, Any]],
        model_name: str | None,
        sampling: ChatSamplingParams,
        policy_version: str | None = None,
    ) -> ChatCompletion:
        async_client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
        )
        try:
            response = await async_client.chat.completions.create(
                model=model_name or self.default_model_name,
                messages=self._normalize_messages(messages),
                **self._completion_kwargs(sampling),
            )
        finally:
            await async_client.close()
        return self._to_completion(response, model_name=model_name, policy_version=policy_version)
