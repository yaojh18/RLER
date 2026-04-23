import asyncio

from agent_rl import ChatSamplingParams
import slime.swe_agent.serving.sglang_chat_service as service_module
from slime.swe_agent.serving.sglang_chat_service import SGLangChatService


class _FakeUsage:
    def model_dump(self):
        return {"prompt_tokens": 4, "completion_tokens": 2}


class _FakeMessage:
    def __init__(self, content: str, reasoning_content: str | None = None):
        self.content = content
        self.reasoning_content = reasoning_content


class _FakeChoice:
    def __init__(self, content: str, reasoning_content: str | None = None):
        self.message = _FakeMessage(content, reasoning_content=reasoning_content)
        self.finish_reason = "stop"


class _FakeResponse:
    def __init__(self, content: str, reasoning_content: str | None = None):
        self.choices = [_FakeChoice(content, reasoning_content=reasoning_content)]
        self.usage = _FakeUsage()

    def model_dump(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": self.choices[0].message.content,
                        "reasoning_content": self.choices[0].message.reasoning_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": self.usage.model_dump(),
        }


class _FakeSyncOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_agenerate_text_uses_fresh_async_client_per_asyncio_run(monkeypatch):
    created_clients = []

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.loop = asyncio.get_running_loop()
            self.closed = False
            created_clients.append(self)
            self.chat = type(
                "_Chat",
                (),
                {
                    "completions": type(
                        "_Completions",
                        (),
                        {"create": self._create},
                    )()
                },
            )()

        async def _create(self, **kwargs):
            return _FakeResponse("OK")

        async def close(self):
            self.closed = True

    monkeypatch.setattr(service_module, "OpenAI", _FakeSyncOpenAI)
    monkeypatch.setattr(service_module, "AsyncOpenAI", _FakeAsyncOpenAI)

    service = SGLangChatService(base_url="http://127.0.0.1:9000", default_model_name="model")
    sampling = ChatSamplingParams(temperature=0.0, top_p=1.0, max_tokens=16)

    first = asyncio.run(
        service.agenerate_text(
            messages=[{"role": "user", "content": "Reply with OK"}],
            model_name=None,
            sampling=sampling,
        )
    )
    second = asyncio.run(
        service.agenerate_text(
            messages=[{"role": "user", "content": "Reply with OK"}],
            model_name=None,
            sampling=sampling,
        )
    )

    assert first.content == "OK"
    assert second.content == "OK"
    assert len(created_clients) == 2
    assert created_clients[0].loop is not created_clients[1].loop
    assert all(client.closed for client in created_clients)


def test_agenerate_text_forwards_extra_body(monkeypatch):
    captured = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = type(
                "_Chat",
                (),
                {
                    "completions": type(
                        "_Completions",
                        (),
                        {"create": self._create},
                    )()
                },
            )()

        async def _create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse("OK")

        async def close(self):
            return None

    monkeypatch.setattr(service_module, "OpenAI", _FakeSyncOpenAI)
    monkeypatch.setattr(service_module, "AsyncOpenAI", _FakeAsyncOpenAI)

    service = SGLangChatService(base_url="http://127.0.0.1:9000", default_model_name="model")
    sampling = ChatSamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=16,
        extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    )

    asyncio.run(
        service.agenerate_text(
            messages=[{"role": "user", "content": "Reply with OK"}],
            model_name=None,
            sampling=sampling,
        )
    )

    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_agenerate_text_forwards_response_format(monkeypatch):
    captured = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = type(
                "_Chat",
                (),
                {
                    "completions": type(
                        "_Completions",
                        (),
                        {"create": self._create},
                    )()
                },
            )()

        async def _create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse("OK")

        async def close(self):
            return None

    monkeypatch.setattr(service_module, "OpenAI", _FakeSyncOpenAI)
    monkeypatch.setattr(service_module, "AsyncOpenAI", _FakeAsyncOpenAI)

    service = SGLangChatService(base_url="http://127.0.0.1:9000", default_model_name="model")
    response_format = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
    sampling = ChatSamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=16,
        extra={"response_format": response_format},
    )

    asyncio.run(
        service.agenerate_text(
            messages=[{"role": "user", "content": "Reply with OK"}],
            model_name=None,
            sampling=sampling,
        )
    )

    assert captured["response_format"] == response_format


def test_agenerate_text_prepends_reasoning_content(monkeypatch):
    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = type(
                "_Chat",
                (),
                {
                    "completions": type(
                        "_Completions",
                        (),
                        {"create": self._create},
                    )()
                },
            )()

        async def _create(self, **kwargs):
            return _FakeResponse("```mswea_bash_command\necho hi\n```", reasoning_content="diagnose issue")

        async def close(self):
            return None

    monkeypatch.setattr(service_module, "OpenAI", _FakeSyncOpenAI)
    monkeypatch.setattr(service_module, "AsyncOpenAI", _FakeAsyncOpenAI)

    service = SGLangChatService(base_url="http://127.0.0.1:9000", default_model_name="model")
    sampling = ChatSamplingParams(temperature=0.0, top_p=1.0, max_tokens=16)

    result = asyncio.run(
        service.agenerate_text(
            messages=[{"role": "user", "content": "Reply with OK"}],
            model_name=None,
            sampling=sampling,
        )
    )

    assert result.content == "<think>diagnose issue</think>\n```mswea_bash_command\necho hi\n```"
    assert result.metadata["content_no_thinking"] == "```mswea_bash_command\necho hi\n```"


def test_agenerate_text_reads_vllm_reasoning_field(monkeypatch):
    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = type(
                "_Chat",
                (),
                {
                    "completions": type(
                        "_Completions",
                        (),
                        {"create": self._create},
                    )()
                },
            )()

        async def _create(self, **kwargs):
            response = _FakeResponse("OK")
            response.choices[0].message.reasoning_content = None
            response.choices[0].message.reasoning = "diagnose issue"
            return response

        async def close(self):
            return None

    monkeypatch.setattr(service_module, "OpenAI", _FakeSyncOpenAI)
    monkeypatch.setattr(service_module, "AsyncOpenAI", _FakeAsyncOpenAI)

    service = SGLangChatService(base_url="http://127.0.0.1:9000", default_model_name="model")
    sampling = ChatSamplingParams(temperature=0.0, top_p=1.0, max_tokens=16)

    result = asyncio.run(
        service.agenerate_text(
            messages=[{"role": "user", "content": "Reply with OK"}],
            model_name=None,
            sampling=sampling,
        )
    )

    assert result.content == "<think>diagnose issue</think>\nOK"
    assert result.metadata["content_no_thinking"] == "OK"
