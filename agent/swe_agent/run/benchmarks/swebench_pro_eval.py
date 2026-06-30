from __future__ import annotations

import ast
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

from swe_agent.run.benchmarks.container_runtime import (
    make_bound_environment,
    raise_for_container_error,
    select_container_backend,
)


SWEBENCH_PRO_DATASET_NAMES = {"ScaleAI/SWE-bench_Pro"}


def is_swebench_pro_dataset_name(dataset_name: str) -> bool:
    return dataset_name in SWEBENCH_PRO_DATASET_NAMES


def is_swebench_pro_instance(instance: dict[str, Any]) -> bool:
    return bool(instance.get("dockerhub_tag")) and bool(instance.get("before_repo_set_cmd"))


def convert_swebench_pro_instance(row: dict[str, Any]) -> dict[str, Any]:
    instance = dict(row)
    if instance.get("fail_to_pass") is not None:
        instance["FAIL_TO_PASS"] = json.dumps(_parse_test_list(instance["fail_to_pass"]))
    if instance.get("pass_to_pass") is not None:
        instance["PASS_TO_PASS"] = json.dumps(_parse_test_list(instance["pass_to_pass"]))
    if instance.get("dockerhub_tag") and not instance.get("docker_image"):
        instance["docker_image"] = f"jefzda/sweap-images:{instance['dockerhub_tag']}"
    instance.setdefault("swebench_workdir", "/app")
    return instance


def evaluate_swebench_pro_instances(
    *,
    instance: dict[str, Any],
    patches_by_key: dict[str, str],
    max_workers: int,
    timeout: int,
    work_dir: Path,
) -> dict[str, dict[str, Any]]:
    del max_workers
    instance = convert_swebench_pro_instance(instance)
    root = _official_repo_root()
    with _node_local_singularity_eval_lock(instance):
        official = _load_official_eval(root)
        with tempfile.TemporaryDirectory(prefix=".swebench-pro-eval-", dir=work_dir) as tmp_dir:
            output_dir = Path(tmp_dir)
            unique_patches: dict[str, list[str]] = {}
            for key, patch in patches_by_key.items():
                unique_patches.setdefault(patch or "", []).append(key)

            evaluations: dict[str, dict[str, Any]] = {}
            for index, (patch_text, keys) in enumerate(unique_patches.items()):
                prefix = f"rler_{index}_{hashlib.sha1(patch_text.encode('utf-8')).hexdigest()[:8]}"
                result = _evaluate_one(official, root, instance, patch_text, output_dir, prefix, timeout)
                for key in keys:
                    evaluations[key] = result
            return evaluations


@contextlib.contextmanager
def _node_local_singularity_eval_lock(instance: dict[str, Any]) -> Iterator[None]:
    if select_container_backend() != "singularity":
        yield
        return
    repo = str(instance.get("repo") or instance.get("dockerhub_tag") or "unknown")
    repo_hash = hashlib.sha1(repo.encode("utf-8")).hexdigest()[:12]
    lock_path = Path("/tmp") / f"rler-swebench-pro-eval-{os.getuid()}-{repo_hash}.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _evaluate_one(
    official: Any,
    root: Path,
    instance: dict[str, Any],
    patch_text: str,
    output_dir: Path,
    prefix: str,
    timeout: int,
) -> dict[str, Any]:
    uid = instance["instance_id"]
    scripts_dir = str(root / "run_scripts")
    if select_container_backend() == "docker":
        with _official_cwd(root):
            output = official.eval_with_docker(
                patch_text,
                instance,
                str(output_dir),
                os.environ.get("SWEBENCH_PRO_DOCKERHUB_USERNAME", "jefzda"),
                scripts_dir,
                prefix=prefix,
                redo=True,
            )
    else:
        output = _eval_with_singularity(official, root, instance, patch_text, output_dir, scripts_dir, prefix, timeout)

    stdout = _read_text(output_dir / uid / f"{prefix}_stdout.log")
    stderr = _read_text(output_dir / uid / f"{prefix}_stderr.log")
    evaluation_output = "\n".join(part for part in [stdout, stderr] if part)
    if output is None:
        return _result(instance, [], False, evaluation_output)
    tests = output.get("tests", []) if isinstance(output, dict) else []
    passed = {str(test.get("name")) for test in tests if test.get("status") == "PASSED" and test.get("name")}
    failed = {str(test.get("name")) for test in tests if test.get("status") != "PASSED" and test.get("name")}
    expected = set(_parse_test_list(instance.get("FAIL_TO_PASS", []))) | set(_parse_test_list(instance.get("PASS_TO_PASS", [])))
    return _result(instance, sorted(passed), expected <= passed, evaluation_output, failed_tests=sorted((expected - passed) | failed))


def _eval_with_singularity(
    official: Any,
    root: Path,
    instance: dict[str, Any],
    patch_text: str,
    output_dir: Path,
    scripts_dir: str,
    prefix: str,
    timeout: int,
) -> dict[str, Any] | None:
    uid = instance["instance_id"]
    with _official_cwd(root):
        _, _, workspace_dir = official.prepare_run(uid, str(output_dir), prefix, redo=True)
        files, entryscript_content = official.assemble_workspace_files(uid, scripts_dir, patch_text, instance)
        official.write_files_local(workspace_dir, files)
        official.write_patch_snapshot(str(output_dir), uid, prefix, patch_text)

    env = make_bound_environment(
        image=str(instance["docker_image"]),
        instance=instance,
        binds=[(Path(workspace_dir), "/workspace")],
        cwd="/",
        timeout=timeout,
    )
    try:
        result = env.execute({"command": "bash /workspace/entryscript.sh"}, cwd="/", timeout=timeout)
        raise_for_container_error(result)
    finally:
        if hasattr(env, "cleanup"):
            env.cleanup()

    output_text = str(result.get("output") or "")
    workspace_path = Path(workspace_dir)
    if output_text and not (workspace_path / "stdout.log").exists():
        (workspace_path / "stdout.log").write_text(output_text, encoding="utf-8")
    with _official_cwd(root):
        output = official.collect_outputs_local(workspace_dir, str(output_dir), uid, prefix)
        official.save_entryscript_copy(str(output_dir), uid, prefix, entryscript_content)
    return output


def _result(
    instance: dict[str, Any],
    passed_tests: list[str],
    resolved: bool,
    output: str,
    *,
    failed_tests: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "instance_id": instance["instance_id"],
        "resolved": resolved,
        "passed_actual": passed_tests,
        "failed_actual": failed_tests or [],
        "pass_to_pass_expected": _parse_test_list(instance.get("PASS_TO_PASS", [])),
        "fail_to_pass_expected": _parse_test_list(instance.get("FAIL_TO_PASS", [])),
        "evaluation_output": output,
    }


def _official_repo_root() -> Path:
    candidates = [
        os.environ.get("SWEBENCH_PRO_OFFICIAL_REPO"),
        os.environ.get("SWE_BENCH_PRO_OS_REPO"),
        str(Path.cwd() / "tmp" / "swebench_pro_official"),
        str(Path(__file__).resolve().parent / "swebench_pro_official"),
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "swe_bench_pro_eval.py").exists():
            return Path(candidate).resolve()
    raise FileNotFoundError(
        "Official SWE-bench Pro evaluator not found. Set SWEBENCH_PRO_OFFICIAL_REPO to a clone of "
        "https://github.com/scaleapi/SWE-bench_Pro-os."
    )


def _load_official_eval(root: Path) -> Any:
    module_name = f"_swebench_pro_official_{hashlib.sha1(str(root).encode('utf-8')).hexdigest()[:8]}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(module_name, root / "swe_bench_pro_eval.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official SWE-bench Pro evaluator from {root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _official_cwd(root: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def _parse_test_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    if not isinstance(value, str):
        return [str(value)]
    value = value.strip()
    if not value:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
            return [str(item) for item in parsed if str(item)]
        except Exception:
            pass
    return [value]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
