from agent_rl import ChatCompletion
import pytest

from swe_agent.exceptions import LimitsExceeded
from swe_agent.models import route_textbased_model as route_module
from swe_agent.models.route_textbased_model import RouteTextbasedModel


def test_completion_length_stops_before_executing_truncated_action(monkeypatch):
    completion = ChatCompletion(
        content=(
            "partial reasoning\n"
            "```mswea_bash_command\n"
            "python -c 'print(1)'\n"
            "```"
        ),
        finish_reason="length",
        model_name="policy",
        usage={"prompt_tokens": 2, "completion_tokens": 3},
        metadata={"content_no_thinking": "partial reasoning"},
        input_token_ids=[1, 2],
        output_token_ids=[3, 4, 5],
        output_logprobs=[-0.1, -0.2, -0.3],
    )
    monkeypatch.setattr(
        route_module,
        "tokenize_messages_with_template",
        lambda *args, **kwargs: [1, 2],
    )
    monkeypatch.setattr(
        route_module,
        "get_stop_token_ids",
        lambda *args, **kwargs: [99],
    )
    monkeypatch.setattr(
        route_module,
        "run_generate_with_route_async",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(route_module, "run_async", lambda _: completion)

    model = RouteTextbasedModel(
        model_name="Qwen/Qwen3.6-27B",
        model_kwargs={
            "api_base": "http://127.0.0.1:30000/v1",
            "api_key": "EMPTY",
            "max_tokens": 20480,
        },
    )

    with pytest.raises(LimitsExceeded) as exc_info:
        model.query([{"role": "user", "content": "fix it"}])

    interrupt = exc_info.value
    assert (
        interrupt.messages[0]["extra"]["exit_status"]
        == "CompletionLengthExceeded"
    )
    assistant = interrupt.assistant_message
    assert assistant["token_ids"] == [3, 4, 5]
    assert assistant["logprobs"] == [-0.1, -0.2, -0.3]
    # The canonical assistant schema initializes actions to an empty list.
    # Length-truncated output must never be parsed into executable actions.
    assert assistant["extra"]["actions"] == []


def test_route_error_is_raised_instead_of_becoming_an_empty_response(monkeypatch):
    completion = ChatCompletion(
        content="",
        model_name="policy",
        metadata={"error": "Cannot connect to host 127.0.0.1:31003"},
    )
    monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "1")
    monkeypatch.setattr(
        route_module,
        "tokenize_messages_with_template",
        lambda *args, **kwargs: [1, 2],
    )
    monkeypatch.setattr(
        route_module,
        "get_stop_token_ids",
        lambda *args, **kwargs: [99],
    )
    monkeypatch.setattr(
        route_module,
        "run_generate_with_route_async",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(route_module, "run_async", lambda _: completion)

    model = RouteTextbasedModel(
        model_name="Qwen/Qwen3.6-27B",
        model_kwargs={
            "api_base": "http://127.0.0.1:31003/v1",
            "api_key": "EMPTY",
            "max_tokens": 20480,
        },
    )

    with pytest.raises(RuntimeError, match="token-in/out generation failed"):
        model.query([{"role": "user", "content": "fix it"}])
