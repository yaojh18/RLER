import asyncio

import pytest

from agent_rl import ChatCompletion, clear_model_services, register_model_service
from open_instruct.search_rewards.utils import run_utils
from open_instruct.search_rewards.utils.run_utils import (
    ModelRouteConfig,
    clear_model_routes,
    configure_model_route,
    run_chat_with_route,
    run_chat_with_route_async,
)
from swe_agent.models.model_service_textbased_model import ModelServiceTextbasedModel


class FakeService:
    def __init__(self, content="```mswea_bash_command\necho hi\n```"):
        self.content = content
        self.calls = []

    def generate_text(self, *, messages, model_name, sampling, policy_version=None):
        self.calls.append(
            {
                "messages": messages,
                "model_name": model_name,
                "sampling": sampling,
                "policy_version": policy_version,
            }
        )
        return ChatCompletion(
            content=self.content,
            finish_reason="stop",
            model_name=model_name,
            metadata={"transport": "fake"},
        )

    async def agenerate_text(self, *, messages, model_name, sampling, policy_version=None):
        return self.generate_text(
            messages=messages,
            model_name=model_name,
            sampling=sampling,
            policy_version=policy_version,
        )


@pytest.fixture(autouse=True)
def _clear_registries():
    clear_model_routes()
    clear_model_services()
    yield
    clear_model_routes()
    clear_model_services()


def test_model_service_textbased_model_uses_registered_service():
    service = FakeService()
    register_model_service("policy-test", service)
    model = ModelServiceTextbasedModel(
        model_name="unused",
        service_name="policy-test",
        service_model_name="shared-policy",
    )
    model.set_policy_version("checkpoint-7")

    response = model.query(
        [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Run echo hi."},
        ]
    )

    assert response["extra"]["actions"] == [{"command": "echo hi"}]
    assert service.calls[0]["policy_version"] == "checkpoint-7"
    assert service.calls[0]["model_name"] == "shared-policy"


def test_route_router_supports_shared_service_sync_and_async():
    service = FakeService(content="service-routed-response")
    register_model_service("policy-test", service)
    configure_model_route(
        "judge",
        ModelRouteConfig(backend="service", service_name="policy-test", model_name="shared-policy"),
    )

    sync_response = run_chat_with_route("judge", "ignored-model", user_prompt="hello", system_prompt="sys")
    async_response = asyncio.run(
        run_chat_with_route_async("judge", "ignored-model", user_prompt="hello", system_prompt="sys")
    )

    assert sync_response == "service-routed-response"
    assert async_response == "service-routed-response"
    assert len(service.calls) == 2
    assert all(call["model_name"] == "shared-policy" for call in service.calls)


def test_route_router_supports_litellm_route(monkeypatch):
    captured = {}

    def fake_run_litellm(model_name, user_prompt, system_prompt=None, messages=None, **chat_kwargs):
        captured["model_name"] = model_name
        captured["user_prompt"] = user_prompt
        captured["messages"] = messages
        captured["chat_kwargs"] = chat_kwargs
        return "litellm-routed-response"

    monkeypatch.setattr(run_utils, "run_litellm", fake_run_litellm)
    configure_model_route("rubric_generation", ModelRouteConfig(backend="litellm"))

    response = run_chat_with_route(
        "rubric_generation",
        "external-api-model",
        user_prompt="generate rubric",
        system_prompt="sys",
    )

    assert response == "litellm-routed-response"
    assert captured["model_name"] == "external-api-model"
    assert captured["user_prompt"] == "generate rubric"
