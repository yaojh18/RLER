from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from swe_agent.run.benchmarks.container_runtime import make_bound_environment, raise_for_container_error


DEEPSWE_DATASET_NAMES = {"datacurve/deep-swe"}


def is_deepswe_dataset_name(dataset_name: str) -> bool:
    return dataset_name in DEEPSWE_DATASET_NAMES


def is_deepswe_instance(instance: dict[str, Any]) -> bool:
    return bool(instance.get("verifier_script")) and bool(instance.get("test_patch"))


def convert_deepswe_instance(row: dict[str, Any]) -> dict[str, Any]:
    instance = dict(row)
    instance["instance_id"] = str(instance.get("instance_id") or instance.get("task_id"))
    repo = str(instance.get("repo") or "")
    if repo.startswith("https://github.com/"):
        repo = repo.removeprefix("https://github.com/").removesuffix(".git")
    instance["repo"] = repo
    instance.setdefault("patch", instance.get("reference_patch") or "")
    instance.setdefault("swebench_workdir", "/app")
    return instance


def evaluate_deepswe_instances(
    *,
    instance: dict[str, Any],
    patches_by_key: dict[str, str],
    max_workers: int,
    timeout: int,
    work_dir: Path,
) -> dict[str, dict[str, Any]]:
    del max_workers
    instance = convert_deepswe_instance(instance)
    with tempfile.TemporaryDirectory(prefix=".deepswe-eval-", dir=work_dir) as tmp_dir:
        output_dir = Path(tmp_dir)
        unique_patches: dict[str, list[str]] = {}
        for key, patch in patches_by_key.items():
            unique_patches.setdefault(patch or "", []).append(key)

        evaluations: dict[str, dict[str, Any]] = {}
        for patch_text, keys in unique_patches.items():
            result = _evaluate_one(instance, patch_text, output_dir, timeout)
            for key in keys:
                evaluations[key] = result
        return evaluations


def _evaluate_one(instance: dict[str, Any], patch_text: str, output_dir: Path, timeout: int) -> dict[str, Any]:
    run_id = hashlib.sha1((instance["instance_id"] + patch_text).encode("utf-8")).hexdigest()[:12]
    workspace = output_dir / instance["instance_id"] / run_id
    tests_dir = workspace / "tests"
    logs_dir = workspace / "logs"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "verifier").mkdir(parents=True, exist_ok=True)
    (workspace / "model.patch").write_text(patch_text, encoding="utf-8")
    (workspace / "verifier.sh").write_text(str(instance["verifier_script"]), encoding="utf-8")
    (tests_dir / "test.patch").write_text(str(instance["test_patch"]), encoding="utf-8")

    env = make_bound_environment(
        image=str(instance["docker_image"]),
        instance=instance,
        binds=[(workspace, "/workspace"), (tests_dir, "/tests"), (logs_dir, "/logs")],
        cwd="/app",
        timeout=timeout,
    )
    command = f"""
set -euo pipefail
cd /app
git config --global --add safe.directory /app 2>/dev/null || true
git reset --hard {instance['base_commit']}
git clean -fd
if [ -s /workspace/model.patch ]; then
  git apply -v /workspace/model.patch
fi
bash /workspace/verifier.sh
"""
    try:
        result = env.execute({"command": command}, cwd="/app", timeout=timeout)
        raise_for_container_error(result)
    finally:
        if hasattr(env, "cleanup"):
            env.cleanup()

    output = str(result.get("output") or "")
    (workspace / "verifier_output.log").write_text(output, encoding="utf-8")
    reward = _read_reward(logs_dir / "verifier" / "reward.txt")
    resolved = reward == 1.0
    return {
        "instance_id": instance["instance_id"],
        "resolved": resolved,
        "reward": reward,
        "passed_actual": [],
        "failed_actual": [],
        "passed_expected": [],
        "pass_to_pass_expected": [],
        "fail_to_pass_expected": [],
        "evaluation_output": output,
    }


def _read_reward(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f"DeepSWE verifier did not write reward file: {path}")
    return float(path.read_text(encoding="utf-8").strip())
