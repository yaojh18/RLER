"""Agent implementations for mini-SWE-agent."""

import copy

from swe_agent import Agent, Environment, Model

_AGENT_MAPPING = {
    "default": "swe_agent.agents.default.DefaultAgent",
    "interactive": "swe_agent.agents.interactive.InteractiveAgent",
}


def get_agent_class(spec: str) -> type[Agent]:
    full_path = _AGENT_MAPPING.get(spec, spec)
    if full_path == "swe_agent.agents.default.DefaultAgent":
        from swe_agent.agents.default import DefaultAgent

        return DefaultAgent
    if full_path == "swe_agent.agents.interactive.InteractiveAgent":
        from swe_agent.agents.interactive import InteractiveAgent

        return InteractiveAgent
    msg = f"Unknown agent type: {spec} (resolved to {full_path}, available: {_AGENT_MAPPING})"
    raise ValueError(msg)


def get_agent(model: Model, env: Environment, config: dict, *, default_type: str = "") -> Agent:
    config = copy.deepcopy(config)
    agent_class = get_agent_class(config.pop("agent_class", default_type))
    return agent_class(model, env, **config)
