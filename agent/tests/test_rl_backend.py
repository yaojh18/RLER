from types import SimpleNamespace

import pytest

from agent_rl import RolloutSessionSpec
from swe_agent.exceptions import InterruptAgentFlow
from swe_agent.rl_backend import SWEAgentSession


class _FakeAgent:
    def __init__(
        self,
        *,
        query_result=None,
        query_exc=None,
        execute_result=None,
        execute_exc=None,
    ):
        self.messages = [
            {"role": "system", "content": "You are SWE-agent."},
            {"role": "user", "content": "Fix the failing test."},
        ]
        self.extra_template_vars = {}
        self.model = SimpleNamespace(config=SimpleNamespace(model_name="openai/fake"))
        self.env = SimpleNamespace()
        self.config = SimpleNamespace(system_template="", instance_template="", step_limit=0, cost_limit=0)
        self.cost = 0.0
        self.n_calls = 0
        self._query_result = query_result
        self._query_exc = query_exc
        self._execute_result = execute_result or []
        self._execute_exc = execute_exc

    def add_messages(self, *messages):
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, error: Exception):
        return self.add_messages(
            {
                "role": "exit",
                "content": str(error),
                "extra": {
                    "exit_status": type(error).__name__,
                    "submission": "",
                    "exception_str": str(error),
                },
            }
        )

    def query(self):
        if self._query_exc is not None:
            raise self._query_exc
        self.n_calls += 1
        self.add_messages(self._query_result)
        return self._query_result

    def execute_actions(self, message):
        if self._execute_exc is not None:
            raise self._execute_exc
        return self.add_messages(*self._execute_result)


def _make_session(agent: _FakeAgent) -> SWEAgentSession:
    return SWEAgentSession(
        agent=agent,
        spec=RolloutSessionSpec(
            session_id="demo-session",
            task="Fix the failing test.",
            task_id="demo__demo",
        ),
    )


def test_step_query_exception_records_agent_error_and_exit_message():
    session = _make_session(_FakeAgent(query_exc=ValueError("boom")))

    with pytest.raises(ValueError, match="boom"):
        session.step()

    assert session.status == "finished"
    assert session.last_step_index == 0
    assert session.agent.messages[-1]["role"] == "exit"
    assert session.export_result().exit_status == "ValueError"
    assert [event.kind for event in session.events] == ["model_request", "agent_error"]
    assert session.events[-1].payload["error_type"] == "ValueError"


def test_step_execute_exception_records_agent_error_after_model_response():
    session = _make_session(
        _FakeAgent(
            query_result={
                "role": "assistant",
                "content": "Run the targeted test.",
                "extra": {"actions": [{"command": "pytest tests/test_alpha.py -q"}]},
            },
            execute_exc=RuntimeError("exec failed"),
        )
    )

    with pytest.raises(RuntimeError, match="exec failed"):
        session.step()

    assert session.status == "finished"
    assert session.last_step_index == 0
    assert session.agent.messages[-1]["role"] == "exit"
    assert len(session.model_turns) == 1
    assert [event.kind for event in session.events] == [
        "model_request",
        "model_response",
        "environment_action",
        "agent_error",
    ]
    assert session.events[-1].payload["error_type"] == "RuntimeError"


def test_non_terminal_interrupt_keeps_session_running():
    session = _make_session(
        _FakeAgent(
            query_exc=InterruptAgentFlow(
                {
                    "role": "user",
                    "content": "No tool calls found.",
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        )
    )

    session.step()

    assert session.status == "running"
    assert session.last_step_index == 0
    assert session.agent.messages[-1]["role"] == "user"
    assert [event.kind for event in session.events] == ["model_request", "agent_interrupt"]
