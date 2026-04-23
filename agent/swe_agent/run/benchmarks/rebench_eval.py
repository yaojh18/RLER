# TODO: I notice when you run rebench eval, you left a rebench_eval folder in the output files.
# TODO: I don't know ehere the cause is, but when evaluating on rebench, the outputs should be the same as swebench verified. Do not add additional files.

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

from swe_agent.run.benchmarks.swebench import get_swebench_docker_image_name

REBENCH_DATASET_NAMES = {
    "nebius/SWE-rebench",
    "nebius/SWE-rebench-V2",
}

_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


def is_rebench_dataset_name(dataset_name: str) -> bool:
    return dataset_name in REBENCH_DATASET_NAMES


def is_rebench_instance(instance: dict[str, Any]) -> bool:
    return bool(instance.get("image_name")) and isinstance(instance.get("install_config"), dict)


@lru_cache(maxsize=1)
def _load_log_parsers_module():
    vendor_root = Path(__file__).resolve().parent / "SWE-rebench-V2"
    if not vendor_root.exists():
        raise FileNotFoundError(f"Missing vendored SWE-rebench-V2 repository at {vendor_root}")
    sys.path.insert(0, str(vendor_root))
    return importlib.import_module("lib.agent.log_parsers")


def _normalize_test_name(name: str) -> str:
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def _normalize_command_list(value: object, field_name: str, instance_id: str) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"Task {instance_id} missing {field_name}.")
    commands = [item for item in value if isinstance(item, str) and item.strip()]
    if not commands:
        raise ValueError(f"Task {instance_id} missing {field_name}.")
    return commands


def evaluate_rebench_instance(
    *,
    instance: dict[str, Any],
    patch_text: str,
    timeout: int,
    work_dir: Path,
) -> dict[str, Any]:
    instance_id = instance["instance_id"]
    repo = instance.get("repo")
    if not repo or "/" not in repo:
        raise ValueError(f"Task {instance_id} missing repo.")
    image_name = get_swebench_docker_image_name(instance)

    install_config = instance.get("install_config") or {}
    test_cmds = _normalize_command_list(install_config.get("test_cmd", []), "install_config.test_cmd", instance_id)
    parser_name = install_config.get("log_parser")
    if not parser_name:
        raise ValueError(f"Task {instance_id} missing install_config.log_parser.")

    log_parsers = _load_log_parsers_module()
    parser = log_parsers.NAME_TO_PARSER.get(parser_name) or getattr(log_parsers, parser_name, None)
    if parser is None:
        raise ValueError(f"Unknown log parser: {parser_name}")

    resolved_work_dir = work_dir.resolve()
    eval_dir = resolved_work_dir / "rebench_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    workdir = f"/{repo.split('/')[1]}"
    test_patch = instance.get("test_patch", "")
    if not test_patch:
        raise ValueError(f"Task {instance_id} missing test_patch.")

    with tempfile.TemporaryDirectory(prefix=".rebench-eval-", dir=resolved_work_dir) as tmp_dir:
        patch_dir = Path(tmp_dir).resolve()
        (patch_dir / "patch.diff").write_text(patch_text, encoding="utf-8")
        (patch_dir / "test_patch.diff").write_text(test_patch, encoding="utf-8")
        script = "\n".join(
            [
                "set -e",
                "git reset --hard HEAD",
                "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/patch.diff",
                "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/test_patch.diff",
                *test_cmds,
            ]
        )
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "host",
            "-e",
            "_JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false",
            "-v",
            f"{patch_dir}:/patches:ro",
            "-w",
            workdir,
            image_name,
            "/bin/bash",
            "-c",
            script,
        ]
        try:
            completed = subprocess.run(
                docker_cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            exit_code = completed.returncode
            output = (completed.stdout or "") + (completed.stderr or "")
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            exit_code = -1
            output = stdout + stderr + f"\nTimed out after {timeout} seconds while evaluating patch.\n"

    log_path = eval_dir / f"{instance_id}.log"
    log_path.write_text(output, encoding="utf-8")
    parsed = parser(output)
    parsed = {_normalize_test_name(name): status for name, status in parsed.items()}
    passed = sorted(name for name, status in parsed.items() if status == "PASSED")
    failed = sorted(name for name, status in parsed.items() if status == "FAILED")
    expected_passed = sorted(
        _normalize_test_name(name) for name in (instance.get("PASS_TO_PASS", []) + instance.get("FAIL_TO_PASS", []))
    )
    return {
        "instance_id": instance_id,
        "resolved": passed == expected_passed,
        "exit_code": exit_code,
        "passed_actual": passed,
        "failed_actual": failed,
        "passed_expected": expected_passed,
        "log_path": str(log_path),
        "parser_name": parser_name,
        "image_name": image_name,
    }


def evaluate_rebench_instances(
    *,
    instance: dict[str, Any],
    patches_by_key: dict[str, str],
    max_workers: int,
    timeout: int,
    work_dir: Path,
) -> dict[str, float]:
    rewards: dict[str, float] = {}
    unique_patches: dict[str, dict[str, Any]] = {}
    for key, patch in patches_by_key.items():
        patch_text = patch or ""
        if not patch_text.strip():
            rewards[key] = 0.0
            continue
        entry = unique_patches.setdefault(patch_text, {"patch": patch_text, "keys": []})
        entry["keys"].append(key)
    if not unique_patches:
        return rewards

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(unique_patches)))) as executor:
        futures = {
            executor.submit(
                evaluate_rebench_instance,
                instance=instance,
                patch_text=entry["patch"],
                timeout=timeout,
                work_dir=work_dir,
            ): entry["keys"]
            for entry in unique_patches.values()
        }
        for future, keys in list(futures.items()):
            try:
                resolved = bool(future.result()["resolved"])
            except Exception:
                resolved = False
            score = 1.0 if resolved else 0.0
            for key in keys:
                rewards[key] = score
    return rewards
