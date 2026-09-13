from __future__ import annotations

import asyncio

import pytest

from agent_rl import ChatCompletion
from experience_gen import models
from experience_gen.config import (
    MODEL_COMPLETION_TOKENS,
    ModelConfig,
)


def test_model_completion_cap() -> None:
    config = ModelConfig()
    assert config.max_tokens == 20_480
    assert MODEL_COMPLETION_TOKENS == 20_480
    assert config.retry_max_tokens == MODEL_COMPLETION_TOKENS


def test_json_client_retries_with_litellm(monkeypatch) -> None:
    calls = []

    async def fake_litellm_message(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "role": "assistant",
                "content": "unfinished",
                "finish_reason": "length",
            }
        return {
            "role": "assistant",
            "content": '{"value": 7}',
            "finish_reason": "stop",
        }

    monkeypatch.setattr(
        models,
        "litellm_message",
        fake_litellm_message,
    )
    parsed, messages = asyncio.run(
        models.JsonModelClient().call(
            model="openai/openai/gpt-5.5",
            system="system",
            user="user",
            max_tokens=256,
            retry_max_tokens=256,
            validator=lambda value: value.get("value") == 7,
        )
    )
    assert parsed == {"value": 7}
    assert [call["max_tokens"] for call in calls] == [256, 256]
    assert messages[-2]["role"] == "user"


def test_gpt_litellm_transport_preserves_gateway_model_id(monkeypatch) -> None:
    calls = []

    async def fake_completion(**kwargs):
        calls.append(kwargs)
        return ChatCompletion(
            content='{"status":"ok"}',
            usage={"completion_tokens": 40},
            metadata={"content_no_thinking": '{"status":"ok"}'},
        )

    monkeypatch.setattr(models, "run_litellm_completion_async", fake_completion)
    message = asyncio.run(
        models.litellm_message(
            model="openai/openai/gpt-5.5",
            messages=[{"role": "user", "content": "return JSON"}],
            max_tokens=512,
            temperature=0.1,
            top_p=0.95,
            route_name="test",
            model_kwargs={"custom_llm_provider": "openai"},
        )
    )
    assert message["content"] == '{"status":"ok"}'
    assert calls[0]["model_name"] == "openai/openai/openai/gpt-5.5"
    assert calls[0]["reasoning_effort"] == "high"
    assert "temperature" not in calls[0]
    assert "top_p" not in calls[0]
    assert "extra_body" not in calls[0]


def test_json_client_preserves_invalid_messages(monkeypatch) -> None:
    async def fake_litellm_message(**kwargs):
        return {
            "role": "assistant",
            "content": '{"wrong": true}',
            "finish_reason": "stop",
        }

    monkeypatch.setattr(
        models,
        "litellm_message",
        fake_litellm_message,
    )
    with pytest.raises(models.InvalidJsonResponse) as caught:
        asyncio.run(
            models.JsonModelClient().call(
                model="nvidia/zai-org/glm-5.2",
                system="system",
                user="user",
                max_tokens=128,
                validator=lambda value: "required" in value,
            )
        )
    assert caught.value.messages[-1]["content"] == '{"wrong": true}'
    assert "finish_reason=stop" in str(caught.value)
