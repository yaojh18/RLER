import asyncio

from agent_rl import ChatCompletion
import swe_agent.models.route_textbased_model as route_model_module
from swe_agent.models.route_textbased_model import RouteTextbasedModel


def test_policy_route_uses_run_chat_with_route_async(monkeypatch):
    captured = {}

    async def fake_run_chat_with_route_completion_async(
        route_name,
        model_name,
        user_prompt=None,
        system_prompt=None,
        messages=None,
        **chat_kwargs,
    ):
        captured["route_name"] = route_name
        captured["model_name"] = model_name
        captured["user_prompt"] = user_prompt
        captured["system_prompt"] = system_prompt
        captured["messages"] = messages
        captured["chat_kwargs"] = chat_kwargs
        return ChatCompletion(
            content="```mswea_bash_command\necho hi\n```",
            finish_reason="length",
            cost=1.25,
            raw_response={"choices": [{"message": {"content": "```mswea_bash_command\necho hi\n```"}}]},
            metadata={"timestamp": 123.0},
        )

    monkeypatch.setattr(
        route_model_module,
        "run_chat_with_route_completion_async",
        fake_run_chat_with_route_completion_async,
    )
    monkeypatch.setattr(route_model_module, "run_async", lambda coro: asyncio.run(coro))

    model = RouteTextbasedModel(
        model_name="openai/test-model",
        route_name="policy",
        model_kwargs={"temperature": 0.2, "api_base": "http://127.0.0.1:9000/v1"},
    )
    model.set_policy_version("checkpoint-3")
    response = model.query(
        [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Run echo hi."},
        ],
        top_p=0.8,
        max_tokens=96,
    )

    assert response["extra"]["actions"] == [{"command": "echo hi"}]
    assert response["extra"]["finish_reason"] == "length"
    assert response["extra"]["cost"] == 1.25
    assert response["extra"]["response"]["choices"][0]["message"]["content"] == "```mswea_bash_command\necho hi\n```"
    assert captured["route_name"] == "policy"
    assert captured["model_name"] == "openai/test-model"
    assert captured["messages"][0]["content"] == "You are a coding agent."
    assert captured["chat_kwargs"]["temperature"] == 0.2
    assert captured["chat_kwargs"]["top_p"] == 0.8
    assert captured["chat_kwargs"]["max_tokens"] == 96
    assert captured["chat_kwargs"]["api_base"] == "http://127.0.0.1:9000/v1"
    assert captured["chat_kwargs"]["policy_version"] == "checkpoint-3"
