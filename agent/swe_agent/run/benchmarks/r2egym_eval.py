from __future__ import annotations

import base64
import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any

from swe_agent.environments.singularity import (
    SingularityEnvironment,
    resolve_singularity_image,
)
from swe_agent.run.benchmarks.container_runtime import raise_for_container_error, select_container_backend


R2EGYM_DATASET_NAMES = {
    "R2E-Gym/R2E-Gym-Subset",
    "R2E-Gym/R2E-Gym-Lite",
    "R2E-Gym/R2E-Gym-Full",
}
R2EGYM_SRC = Path(__file__).resolve().parent / "r2e_gym" / "src"


def _load_official_r2egym() -> tuple[str, type, Any]:
    """Keep the vendored R2E-Gym runtime importable under RLER's newer deps."""
    import sys
    import types

    import swebench.harness.log_parsers as swebench_log_parsers

    if R2EGYM_SRC.exists() and str(R2EGYM_SRC) not in sys.path:
        sys.path.insert(0, str(R2EGYM_SRC))

    utils = types.ModuleType("r2egym.agenthub.utils.utils")
    utils.match_dockerimage_to_repo = lambda docker_image: (docker_image.split("/", 1)[0], True)
    utils.get_logger = lambda name="r2e", level=logging.INFO: logging.getLogger(name)
    sys.modules["r2egym.agenthub.utils.utils"] = utils

    swebench_utils = types.ModuleType("r2egym.agenthub.trajectory.swebench_utils")
    swebench_utils.make_test_spec = lambda *args, **kwargs: None
    swebench_utils.swebench_parse = lambda *args, **kwargs: None

    class TestSpec:
        pass

    swebench_utils.TestSpec = TestSpec
    sys.modules["r2egym.agenthub.trajectory.swebench_utils"] = swebench_utils

    if not hasattr(swebench_log_parsers, "get_eval_type"):
        swebench_log_parsers.get_eval_type = lambda *args, **kwargs: None

    from r2egym.agenthub.runtime.docker import DOCKER_PATH, DockerRuntime
    from r2egym.repo_analysis.execution_log_parser import decolor_dict_keys

    return DOCKER_PATH, DockerRuntime, decolor_dict_keys


DOCKER_PATH, DockerRuntime, decolor_dict_keys = _load_official_r2egym()


class R2ESingularityRuntime(DockerRuntime):
    """Singularity backend for the official R2E-Gym DockerRuntime methods."""

    def __init__(self, ds: dict[str, Any], *, timeout: int, logger: logging.Logger | None = None):
        self.ds = ds
        self.backend = "singularity"
        self.docker_image = str(ds["docker_image"])
        self.swebench_verified = False
        self.swesmith = False
        self.repo_path = "/testbed"
        self.alt_path = "/root"
        self.command = "/bin/bash"
        self.repo_name = str(ds.get("repo_name") or ds.get("repo") or "")
        self.docker_kwargs = {}
        self.timeout = timeout
        self.logger = logger or logging.getLogger("R2ESingularityRuntime")
        self.container_name = self._get_container_name(self.docker_image)
        self.container = None
        self.env = self._make_env()
        self.setup_env()

    def _make_env(self) -> SingularityEnvironment:
        return SingularityEnvironment(
            image=resolve_singularity_image(self.docker_image, self.ds),
            cwd=self.repo_path,
            env={"PATH": DOCKER_PATH},
            timeout=self.timeout,
        )

    def run(
        self,
        code: str,
        timeout: int = 120,
        args: str = "",
        workdir: str | None = None,
        type: str | None = None,
    ) -> tuple[str, str]:
        del type
        command = f"timeout {int(timeout)} {code} {args}".strip()
        result = self.env.execute({"command": command}, cwd=workdir or self.repo_path, timeout=timeout + 5)
        raise_for_container_error(result)
        output = str(result.get("output") or "")
        output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
        returncode = int(result.get("returncode", -1))
        if returncode == 124:
            self.logger.error("Internal Timeout: %ss", timeout)
            return f"The command took too long to execute (>{timeout}s)", "-1"
        if returncode != 0:
            self.logger.error("Error: Exit code %s\nError Message: %s", returncode, output)
            return output, f"Error: Exit code {returncode}"
        return output, str(returncode)

    def copy_to_container(self, src_path: str, dest_path: str) -> None:
        data = Path(src_path).read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        command = (
            f"mkdir -p {shlex.quote(str(Path(dest_path).parent))} && "
            f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(dest_path)}"
        )
        result = self.env.execute({"command": command}, cwd="/", timeout=120)
        raise_for_container_error(result)

    def reset(self) -> None:
        self.env.cleanup()
        self.env = self._make_env()
        self.setup_env()

    def close(self) -> None:
        self.env.cleanup()


def is_r2egym_dataset_name(dataset_name: str) -> bool:
    return dataset_name in R2EGYM_DATASET_NAMES


def is_r2egym_instance(instance: dict[str, Any]) -> bool:
    return bool(instance.get("docker_image")) and bool(instance.get("expected_output_json"))


def r2egym_instance_id(row: dict[str, Any]) -> str:
    if row.get("instance_id"):
        return str(row["instance_id"])
    repo = str(row.get("repo_name") or row.get("repo") or "r2egym").replace("/", "__")
    commit = str(row.get("commit_hash") or str(row.get("docker_image", "")).rsplit(":", 1)[-1])
    return f"{repo}__{commit}"


def r2egym_workdir(instance: dict[str, Any]) -> str:
    return str(instance.get("r2egym_workdir") or "/testbed")


def _status_map(status_map: dict[str, Any]) -> dict[str, str]:
    status_map = decolor_dict_keys(status_map)
    return {
        str(key).split(" - ", 1)[0]: str(status_map[key])
        for key in sorted(status_map)
        if str(key).split(" - ", 1)[0]
    }


def _expected_status_map(instance: dict[str, Any]) -> dict[str, str]:
    raw = instance.get("expected_output_json") or "{}"
    expected = json.loads(raw) if isinstance(raw, str) else raw if isinstance(raw, dict) else {}
    return _status_map(expected)


def convert_r2egym_instance(row: dict[str, Any]) -> dict[str, Any]:
    instance = dict(row)
    instance["instance_id"] = r2egym_instance_id(instance)
    instance.setdefault("repo", instance.get("repo_name") or "")
    instance.setdefault("base_commit", instance.get("commit_hash") or "")
    instance.setdefault("patch", "")
    instance.setdefault("test_patch", "")
    instance["FAIL_TO_PASS"] = sorted(_expected_status_map(instance))
    instance.setdefault("PASS_TO_PASS", [])
    instance["r2egym_workdir"] = r2egym_workdir(instance)
    return instance


def _make_runtime(instance: dict[str, Any], timeout: int):
    if select_container_backend() == "docker":
        return DockerRuntime(ds=instance, command="/bin/bash")
    return R2ESingularityRuntime(instance, timeout=timeout)


def _evaluate_with_runtime(runtime, instance: dict[str, Any], patch_text: str, timeout: int) -> dict[str, Any]:
    if patch_text.strip():
        patch_output, patch_status = runtime.apply_patch(patch_text)
        if patch_status != "0":
            output = patch_output
            reward = 0.0
        else:
            reward, output = runtime._calculate_reward_r2e(get_test_output=True, timeout=timeout)
    else:
        reward, output = runtime._calculate_reward_r2e(get_test_output=True, timeout=timeout)

    expected = _expected_status_map(instance)
    actual = _status_map(runtime.parse_logs(output))
    passed = sorted(name for name, status in expected.items() if actual.get(name) == status)
    failed = sorted(
        {name for name in expected if actual.get(name) != expected[name]}
        | {name for name in actual if name not in expected}
    )
    return {
        "instance_id": instance["instance_id"],
        "resolved": float(reward) == 1.0,
        "reward": float(reward),
        "exit_code": 0 if float(reward) == 1.0 else 1,
        "passed_actual": passed,
        "failed_actual": failed,
        "passed_expected": sorted(expected),
        "pass_to_pass_expected": [],
        "fail_to_pass_expected": sorted(expected),
        "evaluation_output": output,
    }


def evaluate_r2egym_instances(
    *,
    instance: dict[str, Any],
    patches_by_key: dict[str, str],
    max_workers: int,
    timeout: int,
    work_dir: Path,
) -> dict[str, dict[str, Any]]:
    del max_workers, work_dir
    instance = convert_r2egym_instance(instance)
    unique_patches: dict[str, list[str]] = {}
    for key, patch in patches_by_key.items():
        unique_patches.setdefault(patch or "", []).append(key)

    runtime = _make_runtime(instance, timeout)
    try:
        evaluations: dict[str, dict[str, Any]] = {}
        first = True
        for patch_text, keys in unique_patches.items():
            if not first:
                runtime.reset()
            result = _evaluate_with_runtime(runtime, instance, patch_text, timeout)
            for key in keys:
                evaluations[key] = result
            first = False
        return evaluations
    finally:
        runtime.close()
