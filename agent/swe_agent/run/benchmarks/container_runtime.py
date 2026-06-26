from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal

from swe_agent.environments.docker import DockerEnvironment, docker_available
from swe_agent.environments.singularity import (
    SingularityEnvironment,
    resolve_singularity_image,
    singularity_available,
)


ContainerBackend = Literal["docker", "singularity"]


def select_container_backend() -> ContainerBackend:
    if docker_available():
        return "docker"
    if singularity_available():
        return "singularity"
    raise RuntimeError("Neither Docker nor Singularity/Apptainer is available.")


def make_bound_environment(
    *,
    image: str,
    instance: dict[str, Any],
    binds: Iterable[tuple[Path, str]],
    cwd: str,
    timeout: int,
):
    backend = select_container_backend()
    bind_specs = [(str(src.resolve()), dst) for src, dst in binds]
    if backend == "docker":
        run_args = ["--rm", "--memory=256g", "--memory-swap=256g"]
        for src, dst in bind_specs:
            run_args.extend(["-v", f"{src}:{dst}"])
        return DockerEnvironment(image=image, cwd=cwd, timeout=timeout, run_args=run_args)

    exec_args = ["--cleanenv", "--no-home"]
    for src, dst in bind_specs:
        exec_args.extend(["--bind", f"{src}:{dst}"])
    env = SingularityEnvironment(
        image=resolve_singularity_image(image, instance),
        cwd=cwd,
        timeout=timeout,
        exec_args=exec_args,
    )
    for _, dst in bind_specs:
        (env.sandbox_dir / dst.lstrip("/")).mkdir(parents=True, exist_ok=True)
    return env
