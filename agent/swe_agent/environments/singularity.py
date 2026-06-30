#!/usr/bin/env python3

import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from swe_agent.environments import _extract_submission
from swe_agent.exceptions import Submitted
from swe_agent.utils.serialize import recursive_merge


def singularity_available(executable: str | None = None) -> bool:
    executable = executable or singularity_executable()
    try:
        result = subprocess.run(
            [executable, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def singularity_executable() -> str:
    return os.getenv("MSWEA_SINGULARITY_EXECUTABLE") or shutil.which("singularity") or shutil.which("apptainer") or "singularity"


def resolve_singularity_image(image_name: str, instance: dict[str, Any] | None = None) -> str:
    instance = instance or {}
    for key in ("sif_path", "singularity_image", "apptainer_image"):
        value = instance.get(key)
        if value:
            return str(value)
    if image_name.endswith(".sif") and Path(image_name).exists():
        return image_name

    sif_dirs = os.getenv("RLER_SIF_DIR") or os.getenv("SWE_AGENT_SIF_DIR") or os.getenv("SIF_DIR")
    if not sif_dirs:
        sif_dirs = str(Path(__file__).resolve().parents[4] / "singularity_images")
    if sif_dirs:
        instance_id = str(instance.get("instance_id") or "")
        image_tag = image_name.rsplit("/", 1)[-1].replace(":", "_")
        image_stub = _image_name_to_sif_stub(image_name)
        candidates = [f"{image_stub}.sif", f"r2egym_{image_tag}.sif"]
        if image_name.startswith("public.ecr.aws/"):
            candidates.append(f"{image_stub}-v1.1.sif")
        if dockerhub_tag := instance.get("dockerhub_tag"):
            candidates.append(f"jefzda_sweap-images_{dockerhub_tag}.sif")
        if instance_id:
            candidates.extend(
                [
                    f"r2egym_{instance_id}.sif",
                    f"swebench_sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}.sif",
                    f"swebench_sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}_latest.sif",
                    f"swegym_sweb.eval.x86_64.{instance_id.replace('__', '_s_')}.sif",
                ]
            )
        for sif_dir in sif_dirs.split(os.pathsep):
            root = Path(sif_dir)
            for candidate in candidates:
                path = root / candidate
                if path.exists():
                    return str(path)
                matches = list(root.glob(f"*/{candidate}")) if root.exists() else []
                if matches:
                    return str(matches[0])

    raise FileNotFoundError(f"Could not resolve local SIF for image {image_name!r} under {sif_dirs}")


def _image_name_to_sif_stub(image_name: str) -> str:
    image_name = image_name.removeprefix("docker://")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", image_name).strip("_")


class SingularityEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    env: dict[str, str] = {}
    """Environment variables to set in the container."""
    forward_env: list[str] = []
    """Environment variables to forward to the container."""
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = singularity_executable()
    """Path to the singularity executable."""
    sandbox_build_retries: int = 3
    """Number of retries for building the sandbox if an error occurs."""
    global_args: list[str] = ["--quiet"]
    """Global arguments passed before the subcommand (e.g., --quiet, --debug)."""
    exec_args: list[str] = ["--cleanenv", "--no-home"]
    """Arguments passed to `singularity exec`."""
    reuse_sandbox_dir: str | None = None
    """Reuse an existing sandbox directory instead of building a new one."""


class SingularityEnvironment:
    def __init__(
        self, *, config_class: type = SingularityEnvironmentConfig, logger: logging.Logger | None = None, **kwargs
    ):
        """Singularity environment. See `SingularityEnvironmentConfig` for kwargs."""
        self.logger = logger or logging.getLogger("swe_agent.environment")
        self.config = config_class(**kwargs)
        self._owns_sandbox = self.config.reuse_sandbox_dir is None
        self.sandbox_dir = Path(self.config.reuse_sandbox_dir) if self.config.reuse_sandbox_dir else self._build_sandbox()

    def _build_sandbox(self) -> Path:
        # Building the sandbox can fail (very rarely), so we retry it
        max_retries = self.config.sandbox_build_retries
        for attempt in range(max_retries):
            sandbox_dir = Path(tempfile.gettempdir()) / f"swe_agent-{uuid.uuid4().hex[:8]}"
            try:
                subprocess.run(
                    [self.config.executable, "build", "--sandbox", sandbox_dir, self.config.image],
                    check=True,
                    capture_output=True,
                )
                break
            except subprocess.CalledProcessError as e:
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                self.logger.error(
                    f"Error building image {self.config.image}, stdout: {e.stdout}, stderr: {e.stderr} (attempt {attempt + 1}/{max_retries})"
                )
                if attempt == max_retries - 1:
                    raise
        return sandbox_dir

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in a Singularity container and return the result as a dict."""
        command = action.get("command", "")
        cmd = [self.config.executable, *self.config.global_args, "exec", *self.config.exec_args]

        work_dir = cwd or self.config.cwd
        if work_dir and work_dir != "/":
            cmd.extend(["--pwd", work_dir])

        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["--env", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])

        cmd.extend(["--writable", str(self.sandbox_dir), "bash", "-c", command])
        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def get_state(self) -> dict[str, Any]:
        return {"sandbox_dir": str(self.sandbox_dir), "owns_sandbox": self._owns_sandbox}

    def set_state(self, state: dict[str, Any]) -> None:
        sandbox_dir = state.get("sandbox_dir")
        if sandbox_dir:
            self.sandbox_dir = Path(sandbox_dir)
        self._owns_sandbox = state.get("owns_sandbox", False)

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        submission = _extract_submission(output)
        if submission is not None:
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self):
        if self._owns_sandbox:
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def __del__(self):
        """Cleanup sandbox when object is destroyed."""
        self.cleanup()
