"""Environment implementations for mini-SWE-agent."""

import copy

from swe_agent import Environment

_ENVIRONMENT_MAPPING = {
    "docker": "swe_agent.environments.docker.DockerEnvironment",
    "singularity": "swe_agent.environments.singularity.SingularityEnvironment",
    "local": "swe_agent.environments.local.LocalEnvironment",
    "swerex_docker": "swe_agent.environments.extra.swerex_docker.SwerexDockerEnvironment",
    "swerex_modal": "swe_agent.environments.extra.swerex_modal.SwerexModalEnvironment",
    "bubblewrap": "swe_agent.environments.extra.bubblewrap.BubblewrapEnvironment",
    "contree": "swe_agent.environments.extra.contree.ContreeEnvironment",
}


def _extract_submission(output: dict) -> str | None:
    if output.get("returncode") != 0:
        return None
    lines = output.get("output", "").splitlines(keepends=True)
    marker = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    marker_index = next((index for index, line in enumerate(lines) if line.strip() == marker), None)
    if marker_index is None:
        return None
    return "".join(lines[marker_index + 1 :])


def get_environment_class(spec: str) -> type[Environment]:
    full_path = _ENVIRONMENT_MAPPING.get(spec, spec)
    if full_path == "swe_agent.environments.docker.DockerEnvironment":
        from swe_agent.environments.docker import DockerEnvironment

        return DockerEnvironment
    if full_path == "swe_agent.environments.singularity.SingularityEnvironment":
        from swe_agent.environments.singularity import SingularityEnvironment

        return SingularityEnvironment
    if full_path == "swe_agent.environments.local.LocalEnvironment":
        from swe_agent.environments.local import LocalEnvironment

        return LocalEnvironment
    if full_path == "swe_agent.environments.extra.swerex_docker.SwerexDockerEnvironment":
        from swe_agent.environments.extra.swerex_docker import SwerexDockerEnvironment

        return SwerexDockerEnvironment
    if full_path == "swe_agent.environments.extra.swerex_modal.SwerexModalEnvironment":
        from swe_agent.environments.extra.swerex_modal import SwerexModalEnvironment

        return SwerexModalEnvironment
    if full_path == "swe_agent.environments.extra.bubblewrap.BubblewrapEnvironment":
        from swe_agent.environments.extra.bubblewrap import BubblewrapEnvironment

        return BubblewrapEnvironment
    if full_path == "swe_agent.environments.extra.contree.ContreeEnvironment":
        from swe_agent.environments.extra.contree import ContreeEnvironment

        return ContreeEnvironment
    msg = f"Unknown environment type: {spec} (resolved to {full_path}, available: {_ENVIRONMENT_MAPPING})"
    raise ValueError(msg)


def get_environment(config: dict, *, default_type: str = "") -> Environment:
    config = copy.deepcopy(config)
    environment_class = config.pop("environment_class", default_type)
    return get_environment_class(environment_class)(**config)
