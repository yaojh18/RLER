from __future__ import annotations

import copy
import time
import uuid
from typing import Any, Optional

from agent_rl import (
    ModelTurn,
    RolloutEvent,
    RolloutResult,
    RolloutSessionSpec,
    RolloutSnapshot,
    SessionComponentState,
)
from swe_agent.agents import get_agent, get_agent_class
from swe_agent.environments import get_environment, get_environment_class
from swe_agent.exceptions import InterruptAgentFlow
from swe_agent.models import get_model, get_model_class


def _load_class(type_path: str) -> type:
    if type_path.startswith("swe_agent.models."):
        return get_model_class("", type_path)
    if type_path.startswith("swe_agent.environments."):
        return get_environment_class(type_path)
    if type_path.startswith("swe_agent.agents."):
        return get_agent_class(type_path)
    raise ValueError(f"Unknown serialized component type: {type_path}")


def _message_to_protocol(message: dict[str, Any], *, source: str, trainable: bool) -> dict[str, Any]:
    metadata = {k: copy.deepcopy(v) for k, v in message.items() if k not in {"role", "content"}}
    return {
        "role": message.get("role", "assistant"),
        "content": copy.deepcopy(message.get("content")),
        "source": source,
        "trainable": trainable,
        "metadata": metadata,
    }


def _messages_to_protocol(messages: list[dict[str, Any]], *, source: str, trainable: bool) -> list[dict[str, Any]]:
    return [_message_to_protocol(message, source=source, trainable=trainable) for message in messages]


def _component_from_serialize(serialized: dict[str, Any], key: str) -> SessionComponentState:
    config_root = serialized.get("info", {}).get("config", {})
    return SessionComponentState(
        type_path=config_root[f"{key}_type"],
        config=copy.deepcopy(config_root[key]),
        state={},
    )


class SWEAgentSession:
    def __init__(
        self,
        *,
        agent: Any,
        spec: RolloutSessionSpec,
        events: Optional[list[RolloutEvent]] = None,
        model_turns: Optional[list[ModelTurn]] = None,
        status: str = "running",
    ) -> None:
        self.agent = agent
        self.spec = spec
        self.events = list(events or [])
        self.model_turns = list(model_turns or [])
        self.status = status
        self.last_step_index = self.model_turns[-1].step_index if self.model_turns else -1
        if not self.agent.messages:
            self._initialize_messages()

    def _record_event(self, *, step_index: int, kind: str, payload: dict[str, Any], provenance: Optional[dict[str, Any]] = None) -> None:
        self.events.append(
            RolloutEvent(
                event_id=f"{self.spec.session_id}:{len(self.events)}",
                session_id=self.spec.session_id,
                step_index=step_index,
                kind=kind,
                ts=time.time(),
                payload=payload,
                provenance=provenance or {},
            )
        )

    def _initialize_messages(self) -> None:
        template_vars = copy.deepcopy(self.spec.metadata.get("template_vars", {}))
        self.agent.extra_template_vars |= {"task": self.spec.task, **template_vars}
        self.agent.messages = []
        self.agent.add_messages(
            self.agent.model.format_message(role="system", content=self.agent._render_template(self.agent.config.system_template)),
            self.agent.model.format_message(role="user", content=self.agent._render_template(self.agent.config.instance_template)),
        )
        self._record_event(
            step_index=-1,
            kind="session_started",
            payload={"messages": copy.deepcopy(self.agent.messages)},
            provenance={
                "policy_ref": self.spec.policy_ref,
                "policy_version": self.spec.policy_version,
            },
        )

    def _record_interrupt(self, *, step_index: int, messages: list[dict[str, Any]]) -> None:
        self._record_event(
            step_index=step_index,
            kind="agent_interrupt",
            payload={"messages": copy.deepcopy(messages)},
        )
        self.last_step_index = step_index

    def _record_uncaught_exception(self, *, step_index: int, error: Exception) -> None:
        added: list[dict[str, Any]] = []
        if hasattr(self.agent, "handle_uncaught_exception"):
            added = self.agent.handle_uncaught_exception(error)
        self._record_event(
            step_index=step_index,
            kind="agent_error",
            payload={
                "messages": copy.deepcopy(added),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        self.last_step_index = step_index
        self.status = "finished" if self.is_finished() else "failed"

    def is_finished(self) -> bool:
        return bool(self.agent.messages and self.agent.messages[-1].get("role") == "exit")

    def _mark_paused_if_needed(self, max_steps: Optional[int], executed_steps: int) -> bool:
        if self.is_finished():
            self.status = "finished"
            return True
        if max_steps is not None and executed_steps >= max_steps:
            self.status = "paused"
            return True
        if "after_step" in self.spec.pause_points:
            self.status = "paused"
            return True
        self.status = "running"
        return False

    def step(self) -> None:
        if self.is_finished():
            self.status = "finished"
            return

        step_index = self.last_step_index + 1
        query_messages = copy.deepcopy(self.agent.messages)
        self._record_event(
            step_index=step_index,
            kind="model_request",
            payload={"messages": query_messages},
            provenance={
                "policy_ref": self.spec.policy_ref,
                "policy_version": self.spec.policy_version,
                "model_name": getattr(getattr(self.agent.model, "config", None), "model_name", None),
            },
        )

        try:
            model_message = self.agent.query()
            self._record_event(
                step_index=step_index,
                kind="model_response",
                payload={"message": copy.deepcopy(model_message)},
            )
            self.model_turns.append(
                ModelTurn(
                    session_id=self.spec.session_id,
                    step_index=step_index,
                    query_messages=_messages_to_protocol(query_messages, source="conversation", trainable=False),
                    response_message=_message_to_protocol(model_message, source="model", trainable=True),
                    metadata={
                        "policy_ref": self.spec.policy_ref,
                        "policy_version": self.spec.policy_version,
                        "dataset_name": self.spec.dataset_name,
                        "ground_truth": self.spec.ground_truth,
                    },
                )
            )
            actions = copy.deepcopy(model_message.get("extra", {}).get("actions") or [])
            if actions:
                self._record_event(
                    step_index=step_index,
                    kind="environment_action",
                    payload={"actions": actions},
                )
            observation_messages = self.agent.execute_actions(model_message)
        except InterruptAgentFlow as exc:
            self._record_interrupt(step_index=step_index, messages=self.agent.add_messages(*exc.messages))
            self.status = "finished" if self.is_finished() else "running"
            return
        except Exception as exc:
            self._record_uncaught_exception(step_index=step_index, error=exc)
            raise

        if observation_messages:
            self._record_event(
                step_index=step_index,
                kind="environment_result",
                payload={"messages": copy.deepcopy(observation_messages)},
            )

        self.last_step_index = step_index
        self.status = "finished" if self.is_finished() else "running"

    def run_until_pause(self, max_steps: Optional[int] = None) -> RolloutResult:
        executed_steps = 0
        while True:
            if self._mark_paused_if_needed(max_steps=max_steps, executed_steps=executed_steps):
                break
            self.step()
            executed_steps += 1
        return self.export_result()

    def _component_state(self, component: Any, *, key: str) -> SessionComponentState:
        if key == "agent":
            return SessionComponentState(
                type_path=f"{component.__class__.__module__}.{component.__class__.__name__}",
                config=component.config.model_dump(mode="json"),
                state={
                    "messages": copy.deepcopy(component.messages),
                    "extra_template_vars": copy.deepcopy(component.extra_template_vars),
                    "cost": component.cost,
                    "n_calls": component.n_calls,
                },
            )

        serialized = component.serialize()
        state = component.get_state() if hasattr(component, "get_state") else {}
        component_state = _component_from_serialize(serialized, key)
        component_state.state = copy.deepcopy(state)
        return component_state

    def snapshot(self) -> RolloutSnapshot:
        snapshot_status = "finished" if self.is_finished() else "paused"
        self.status = snapshot_status
        return RolloutSnapshot(
            session_id=self.spec.session_id,
            status=snapshot_status,
            spec=self.spec,
            agent=self._component_state(self.agent, key="agent"),
            model=self._component_state(self.agent.model, key="model"),
            environment=self._component_state(self.agent.env, key="environment"),
            last_step_index=self.last_step_index,
            last_event_id=self.events[-1].event_id if self.events else None,
            metadata={
                "events": [event.model_dump(mode="json") for event in self.events],
                "model_turns": [turn.model_dump(mode="json") for turn in self.model_turns],
            },
        )

    def export_result(self) -> RolloutResult:
        final_message = self.agent.messages[-1] if self.agent.messages else {}
        final_extra = final_message.get("extra", {})
        result_status = "finished" if self.is_finished() else self.status
        return RolloutResult(
            session_id=self.spec.session_id,
            status=result_status,
            spec=self.spec,
            events=self.events,
            model_turns=self.model_turns,
            final_messages=_messages_to_protocol(self.agent.messages, source="conversation", trainable=False),
            exit_status=final_extra.get("exit_status", ""),
            submission=final_extra.get("submission", ""),
            metadata={
                "n_calls": self.agent.n_calls,
                "cost": self.agent.cost,
            },
        )


class SWEAgentRolloutBackend:
    def __init__(
        self,
        *,
        model: Optional[dict[str, Any]] = None,
        environment: Optional[dict[str, Any]] = None,
        agent: Optional[dict[str, Any]] = None,
        default_agent_type: str = "default",
        default_environment_type: str = "local",
    ) -> None:
        self.model_config = copy.deepcopy(model or {})
        self.environment_config = copy.deepcopy(environment or {})
        self.agent_config = copy.deepcopy(agent or {})
        self.default_agent_type = default_agent_type
        self.default_environment_type = default_environment_type

    def _build_agent(self) -> Any:
        model = get_model(config=copy.deepcopy(self.model_config))
        environment = get_environment(copy.deepcopy(self.environment_config), default_type=self.default_environment_type)
        return get_agent(model, environment, copy.deepcopy(self.agent_config), default_type=self.default_agent_type)

    def _apply_session_spec(self, agent: Any, spec: RolloutSessionSpec) -> None:
        if hasattr(agent, "set_rollout_spec"):
            agent.set_rollout_spec(spec)
        if hasattr(agent.model, "set_rollout_spec"):
            agent.model.set_rollout_spec(spec)
        if hasattr(agent.env, "set_rollout_spec"):
            agent.env.set_rollout_spec(spec)
        if spec.policy_version and hasattr(agent.model, "set_policy_version"):
            agent.model.set_policy_version(spec.policy_version)
        if "step_limit" in spec.limits and hasattr(agent.config, "step_limit"):
            agent.config.step_limit = spec.limits["step_limit"]
        if "cost_limit" in spec.limits and hasattr(agent.config, "cost_limit"):
            agent.config.cost_limit = spec.limits["cost_limit"]

    def create_session(self, spec: RolloutSessionSpec) -> SWEAgentSession:
        if not spec.session_id:
            spec = spec.model_copy(update={"session_id": str(uuid.uuid4())})
        agent = self._build_agent()
        self._apply_session_spec(agent, spec)
        return SWEAgentSession(agent=agent, spec=spec)

    def resume_session(self, snapshot: RolloutSnapshot) -> SWEAgentSession:
        model = self._restore_model(snapshot.model)
        environment = self._restore_environment(snapshot.environment)
        agent = self._restore_agent(snapshot.agent, model=model, environment=environment)
        events = [RolloutEvent(**event) for event in snapshot.metadata.get("events", [])]
        model_turns = [ModelTurn(**turn) for turn in snapshot.metadata.get("model_turns", [])]
        self._apply_session_spec(agent, snapshot.spec)
        return SWEAgentSession(
            agent=agent,
            spec=snapshot.spec,
            events=events,
            model_turns=model_turns,
            status=snapshot.status,
        )

    def _restore_model(self, component: SessionComponentState) -> Any:
        model_class = _load_class(component.type_path)
        model = model_class(**copy.deepcopy(component.config))
        if hasattr(model, "set_state"):
            model.set_state(copy.deepcopy(component.state))
        return model

    def _restore_environment(self, component: SessionComponentState) -> Any:
        config = copy.deepcopy(component.config)
        if "reuse_container_id" in config and component.state.get("container_id"):
            config["reuse_container_id"] = component.state["container_id"]
        if "reuse_sandbox_dir" in config and component.state.get("sandbox_dir"):
            config["reuse_sandbox_dir"] = component.state["sandbox_dir"]
        environment_class = _load_class(component.type_path)
        environment = environment_class(**config)
        if hasattr(environment, "set_state"):
            environment.set_state(copy.deepcopy(component.state))
        return environment

    def _restore_agent(self, component: SessionComponentState, *, model: Any, environment: Any) -> Any:
        agent_class = get_agent_class(component.type_path)
        agent = agent_class(model, environment, **copy.deepcopy(component.config))
        agent.messages = copy.deepcopy(component.state.get("messages", []))
        agent.extra_template_vars = copy.deepcopy(component.state.get("extra_template_vars", {}))
        agent.cost = component.state.get("cost", 0.0)
        agent.n_calls = component.state.get("n_calls", 0)
        return agent
