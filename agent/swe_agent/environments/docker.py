import logging
import time
import os
import platform
import shlex
import subprocess
import uuid
from typing import Any

from pydantic import BaseModel

from swe_agent.exceptions import Submitted
from swe_agent.utils.serialize import recursive_merge


def docker_available(executable: str | None = None) -> bool:
    executable = executable or os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    try:
        result = subprocess.run(
            [executable, "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


class DockerEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = {}
    """Environment variables to set in the container."""
    forward_env: list[str] = []
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = ["--rm", "--memory=256g", "--memory-swap=256g"]
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images."""
    interpreter: list[str] = ["bash", "-lc"]
    """Interpreter to use to execute commands. Default is ["bash", "-lc"].
    The actual command will be appended as argument to this. Override this to e.g., modify shell flags
    (e.g., to remove the `-l` flag to disable login shell) or to use python instead of bash to interpret commands.
    """
    reuse_container_id: str | None = None
    """Reuse an existing container instead of starting a new one."""
    start_container_retries: int = 3
    """Retries for `docker run` (handles transient registry/daemon errors,
    e.g. missing image in lustre cache → docker.io fallback rate-limit)."""
    start_container_retry_delay: float = 5.0
    """Initial backoff between start-container retries; scaled by attempt#."""


class DockerEnvironment:
    def __init__(
        self,
        *,
        config_class: type = DockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `DockerEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("swe_agent.environment")
        self.container_id: str | None = None
        self._owns_container = False
        self.config = config_class(**kwargs)
        if self.config.reuse_container_id:
            self.container_id = self.config.reuse_container_id
        else:
            self._start_container()

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def _start_container(self):
        """Start the Docker container and return the container ID.

        Retries transient `docker run` failures (exit 125, timeouts) up to
        `start_container_retries` times. The original silent-fail behavior
        produced a stream of dummy rollouts when an image was missing from
        the lustre tarball cache; see slime job 57677.
        """
        container_name = f"swe_agent-{uuid.uuid4().hex[:8]}"
        cmd = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            container_name,
        ]
        if self.config.cwd:
            cmd.extend(["-w", self.config.cwd])
        cmd.extend([
            *self.config.run_args,
            self.config.image,
            "sleep",
            self.config.container_timeout,
        ])
        max_retries = max(1, int(self.config.start_container_retries))
        last_exc: BaseException | None = None
        for attempt in range(max_retries):
            self.logger.debug(
                f"Starting container (attempt {attempt + 1}/{max_retries}): {shlex.join(cmd)}"
            )
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.pull_timeout,  # docker pull might take a while
                    check=True,
                )
                self.logger.info(
                    f"Started container {container_name} with ID {result.stdout.strip()}"
                )
                self.container_id = result.stdout.strip()
                self._owns_container = True
                return
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                last_exc = exc
                stderr = ""
                raw = getattr(exc, "stderr", "") or ""
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                stderr = raw[:300]
                rc = getattr(exc, "returncode", "?")
                self.logger.warning(
                    f"start_container attempt {attempt + 1}/{max_retries} failed for image "
                    f"{self.config.image!r}: {type(exc).__name__} rc={rc} stderr={stderr!r}"
                )
                if attempt < max_retries - 1:
                    time.sleep(self.config.start_container_retry_delay * (attempt + 1))
        raise RuntimeError(
            f"docker start_container failed after {max_retries} attempts for "
            f"image={self.config.image!r}: {type(last_exc).__name__}: {last_exc}"
        ) from last_exc

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        cmd = [self.config.executable, "exec"]
        if cwd:
            cmd.extend(["-w", cwd])
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, *self.config.interpreter, command])

        _t_exec_start = time.perf_counter()
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
            _dt = time.perf_counter() - _t_exec_start
            if _dt >= 0.5:
                _cid = (self.container_id or "")[:12]
                _cmd_short = (command or "").splitlines()[0][:80] if command else ""
                self.logger.info(f"[TIMING] docker_exec took={_dt:.2f}s container={_cid} cmd={_cmd_short!r}")
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

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            if not submission.strip():
                output["returncode"] = 1
                output["output"] = (
                    "Submission rejected: COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT was printed, "
                    "but no non-empty patch/output was emitted after it.\n"
                    "Generate the final git diff first, then print the marker and the non-empty patch content."
                )
                return
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self):
        """Stop and remove the Docker container."""
        if self._owns_container and getattr(self, "container_id", None) is not None:
            cmd = f"(timeout 60 {self.config.executable} stop {self.container_id} || {self.config.executable} rm -f {self.container_id}) >/dev/null 2>&1 &"
            if getattr(subprocess, "Popen", None) is not None:
                subprocess.Popen(cmd, shell=True)

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()

    def get_state(self) -> dict[str, Any]:
        return {"container_id": self.container_id, "owns_container": self._owns_container}

    def set_state(self, state: dict[str, Any]) -> None:
        container_id = state.get("container_id")
        if container_id:
            self.container_id = container_id
        # IMPORTANT: do NOT overwrite _owns_container from `state`. Ownership is
        # a fact about THIS process (did *we* run `docker run`?) — not metadata
        # to be cloned across snapshots. When PDS forks M branches from the
        # same parent snapshot, all of them used to inherit owns=True from the
        # parent's state. Then the first branch to GC fired cleanup() (async
        # `docker stop ... &`) on the PARENT's container, and every still-live
        # branch + descendant round started getting "No such container" errors.
        # Init has already correctly set _owns_container based on whether
        # _start_container ran (owns=True) or reuse_container_id was used
        # (owns=False); leave it alone.
