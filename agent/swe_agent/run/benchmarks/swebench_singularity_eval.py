from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec

from swe_agent.environments.singularity import SingularityEnvironment, resolve_singularity_image


def evaluate_swebench_instances_singularity(
    *,
    instance: dict[str, Any],
    patches_by_key: dict[str, str],
    model_name: str,
    max_workers: int,
    namespace: str | None,
    work_dir: Path,
    timeout: int,
) -> dict[str, dict[str, Any]]:
    unique_patches: dict[str, dict[str, Any]] = {}
    for key, patch in patches_by_key.items():
        patch_text = patch or ""
        patch_hash = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
        unique_patches.setdefault(patch_hash, {"patch": patch_text, "keys": []})["keys"].append(key)

    test_spec = make_test_spec(instance, namespace=namespace or "swebench")
    eval_root = work_dir / ".swebench-sif-eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    evaluations: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(unique_patches) or 1))) as executor:
        future_map = {
            executor.submit(
                _evaluate_one,
                instance,
                test_spec,
                entry["patch"],
                model_name,
                eval_root,
                timeout,
            ): entry["keys"]
            for entry in unique_patches.values()
        }
        for future, keys in future_map.items():
            result = future.result()
            for key in keys:
                evaluations[key] = result
    return evaluations


def _evaluate_one(
    instance: dict[str, Any],
    test_spec: Any,
    patch_text: str,
    model_name: str,
    eval_root: Path,
    timeout: int,
) -> dict[str, Any]:
    run_id = f"sif-eval-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    run_dir = eval_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    env = SingularityEnvironment(
        image=resolve_singularity_image(test_spec.instance_image_key, instance),
        cwd="/testbed",
        timeout=timeout,
    )
    try:
        tmp_dir = env.sandbox_dir / "testbed" / ".rler_eval" / run_id
        tmp_dir.mkdir(parents=True, exist_ok=True)
        patch_path = tmp_dir / "model.patch"
        eval_path = tmp_dir / "eval.sh"
        patch_path.write_text(patch_text, encoding="utf-8")
        eval_path.write_text(test_spec.eval_script, encoding="utf-8")
        patch_container_path = f"/testbed/.rler_eval/{run_id}/model.patch"
        eval_container_path = f"/testbed/.rler_eval/{run_id}/eval.sh"
        result = env.execute(
            {
                "command": f"""
set -euo pipefail
cd /testbed
git config --global --add safe.directory /testbed 2>/dev/null || true
git apply -v {json.dumps(patch_container_path)}
bash {json.dumps(eval_container_path)}
"""
            },
            cwd="/testbed",
            timeout=timeout,
        )
    finally:
        env.cleanup()

    output = str(result.get("output") or "")
    log_path = run_dir / "test_output.txt"
    log_path.write_text(output, encoding="utf-8", errors="replace")
    prediction = {
        "model_name_or_path": model_name,
        "instance_id": test_spec.instance_id,
        "model_patch": patch_text,
    }
    report = get_eval_report(test_spec, prediction, str(log_path), include_tests_status=True)
    return {"report": report, "output": output}
