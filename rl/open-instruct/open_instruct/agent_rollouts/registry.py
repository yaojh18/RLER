from __future__ import annotations

import importlib


_ROLLOUT_BACKEND_MAPPING = {
    "swe_agent": "swe_agent.rl_backend.SWEAgentRolloutBackend",
}


def get_rollout_backend_class(spec: str):
    full_path = _ROLLOUT_BACKEND_MAPPING.get(spec, spec)
    module_name, class_name = full_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
