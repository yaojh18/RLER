#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

IMAGE="${IMAGE:-slimerl/slime:qwen35-route-fixed-20260416-v1}"
MODE="${MODE:-both}"
GPUS="${GPUS:-0,1}"
INSTANCE_ID="${INSTANCE_ID:-}"
INSTANCE_SEED="${INSTANCE_SEED:-42}"
PYTEST_PYTHON="${PYTEST_PYTHON:-${REPO_ROOT}/agent/.venv/bin/python}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
VLLM_MODEL="${VLLM_MODEL:-${MODEL_NAME}}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/slime/swe_agent/artifacts/step1_smoke_verify/qwen35_fullverify_20260415/models/Qwen3.5-9B}"
SLIME_PORT="${SLIME_PORT:-8021}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-80960}"
LITELLM_TIMEOUT="${LITELLM_TIMEOUT:-120}"
SEARCH_EXTRA_ARGS="${SEARCH_EXTRA_ARGS:---policy-temperature 0.0 --policy-top-p 1.0 --completion-max-tokens 2048 --rubric-temperature 0.0 --rubric-top-p 1.0 --rubric-max-tokens 2048 --judge-temperature 0.0 --judge-top-p 1.0 --judge-max-tokens 1024 --calculate-gt-reward false}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-$$}"
HOST_OUTPUT_DIR="${HOST_OUTPUT_DIR:-${REPO_ROOT}/slime/swe_agent/artifacts/step2_route_verify/${RUN_ID}}"
CONTAINER_NAME="slime-step2-route-${RUN_ID}"

cleanup() {
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker image inspect "${IMAGE}" >/dev/null
test -x "${PYTEST_PYTHON}"
test -d "${MODEL_DIR}"
case "${MODEL_DIR}" in
    "${REPO_ROOT}"/*)
        CONTAINER_MODEL_DIR="/workspace/rler/${MODEL_DIR#${REPO_ROOT}/}"
        ;;
    *)
        echo "MODEL_DIR must live under ${REPO_ROOT} so the verify container can mount it: ${MODEL_DIR}" >&2
        exit 2
        ;;
esac

mkdir -p "${HOST_OUTPUT_DIR}/logs"

if [[ -z "${INSTANCE_ID}" ]]; then
    echo "[0/7] selecting deterministic random SWE-rebench-V2 instance"
    INSTANCE_ID="$("${PYTEST_PYTHON}" - <<'PY' "${INSTANCE_SEED}" "${HOST_OUTPUT_DIR}" | tail -n 1
import json
import pathlib
import random
import subprocess
import sys

from swe_agent.run.benchmarks.swebench import load_swebench_instances

seed = int(sys.argv[1])
host_output = pathlib.Path(sys.argv[2])
instances = load_swebench_instances("rebench_v2", "train")
local_images = {
    line.strip()
    for line in subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if line.strip()
}
normalized_local_images = local_images | {
    f"docker.io/{image}" for image in local_images if not image.startswith("docker.io/")
}
supported = [
    instance
    for instance in instances
    if instance.get("language") == "python"
    and instance.get("image_name")
    and instance.get("image_name") in normalized_local_images
    and isinstance(instance.get("install_config"), dict)
    and (instance.get("install_config") or {}).get("test_cmd")
    and (instance.get("install_config") or {}).get("log_parser")
]
if not supported:
    raise SystemExit("No local Python SWE-rebench-V2 instances are available for search verification")
chosen = random.Random(seed).choice(supported)
(host_output / "logs" / "instance_selection.json").write_text(
    json.dumps(
        {
            "seed": seed,
            "candidate_count": len(supported),
            "instance_id": chosen["instance_id"],
            "repo": chosen.get("repo"),
            "base_commit": chosen.get("base_commit"),
            "image_name": chosen.get("image_name"),
        },
        indent=2,
    )
    + "\n"
)
print(chosen["instance_id"])
PY
)"
fi

echo "selected instance: ${INSTANCE_ID}"

echo "[1/7] running route pytest"
"${PYTEST_PYTHON}" -m pytest \
    "${REPO_ROOT}/agent/tests/test_route_textbased_model.py" \
    "${REPO_ROOT}/agent/tests/test_search_swe_agent.py" \
    "${REPO_ROOT}/agent/tests/test_run_swe_agent.py" -q \
    2>&1 | tee "${HOST_OUTPUT_DIR}/logs/pytest.log"

if [[ "${MODE}" == "both" || "${MODE}" == "vllm" ]]; then
    echo "[2/7] running vllm route verification"
    rm -rf "${HOST_OUTPUT_DIR}/vllm_outputs"
    LITELLM_DEFAULT_TIMEOUT="${LITELLM_TIMEOUT}" \
    "${PYTEST_PYTHON}" "${REPO_ROOT}/agent/swe_agent/run/search_swe_agent.py" \
        --backend vllm \
        --instance-id "${INSTANCE_ID}" \
        --subset rebench_v2 \
        --split train \
        --output-root "${HOST_OUTPUT_DIR}/vllm_outputs" \
        --workers 1 \
        --gpu-id "${GPUS}" \
        --vllm-model "${VLLM_MODEL}" \
        --m 2 \
        --k 20 \
        --p 1 \
        ${SEARCH_EXTRA_ARGS} \
        2>&1 | tee "${HOST_OUTPUT_DIR}/logs/vllm_driver.log"

    "${PYTEST_PYTHON}" - <<'PY' "${HOST_OUTPUT_DIR}" "${INSTANCE_ID}"
import json
import pathlib
import sys

host_output = pathlib.Path(sys.argv[1])
instance_id = sys.argv[2]
candidate_roots = sorted((host_output / "vllm_outputs").glob(f"rebench_v2_train_Qwen__Qwen3.5-9B/{instance_id}/*"))
if not candidate_roots:
    raise SystemExit("vllm verification did not produce an output directory")
run_dir = candidate_roots[-1]
evaluation = json.loads((run_dir / "evaluation.json").read_text())
rubric_files = sorted((run_dir / "rubrics").glob("round_*.json"))
if not rubric_files:
    raise SystemExit("vllm verification did not produce any rubric rounds")
rubrics = json.loads(rubric_files[-1].read_text())
node_count = sum(1 for _ in (run_dir / "node_index.jsonl").open())
model_patch = json.loads((run_dir / "model_patch.json").read_text())[instance_id]["model_patch"]
summary = {
    "run_dir": str(run_dir),
    "evaluation": evaluation,
    "rubric_count": len(rubrics.get("generated", [])),
    "node_count": node_count,
    "model_patch_empty": model_patch == "",
    "model_patch_chars": len(model_patch),
}
if summary["rubric_count"] == 0:
    raise SystemExit("vllm verification produced an empty rubric bank")
if summary["node_count"] <= 1:
    raise SystemExit("vllm verification did not progress beyond the root node")
if summary["model_patch_empty"]:
    raise SystemExit("vllm verification finished without a final patch")
(host_output / "logs" / "vllm_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
else
    echo "[2/7] skipping vllm route verification"
fi

if [[ "${MODE}" == "both" || "${MODE}" == "slime" ]]; then
    echo "[3/7] starting owned slime serving container from ${IMAGE}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker run -d --rm \
        --name "${CONTAINER_NAME}" \
        --runtime nvidia \
        --net=host \
        --shm-size=64g \
        -e NVIDIA_VISIBLE_DEVICES="${GPUS}" \
        -e FLASHINFER_USE_CUDA_NORM=1 \
        -v "${REPO_ROOT}:/workspace/rler" \
        -v "${REPO_ROOT}/slime:/workspace/slime" \
        -v "${HOST_OUTPUT_DIR}:/verify_outputs" \
        -w /workspace/slime \
        "${IMAGE}" \
        bash --noprofile --norc -lc "
            python -m sglang.launch_server \
                --model-path '${CONTAINER_MODEL_DIR}' \
                --host 127.0.0.1 \
                --port '${SLIME_PORT}' \
                --tensor-parallel-size 2 \
                --context-length '${CONTEXT_LENGTH}' \
                --mem-fraction-static 0.8 \
                --served-model-name '${MODEL_NAME}' \
                --reasoning-parser qwen3 \
                --skip-server-warmup 2>&1 | tee /verify_outputs/logs/sglang_server.log
        " >/dev/null

    echo "[4/7] waiting for slime route server to become ready"
    "${PYTEST_PYTHON}" - <<'PY' "${SLIME_PORT}" "${HOST_OUTPUT_DIR}"
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

port = sys.argv[1]
host_output = pathlib.Path(sys.argv[2])
ready = False
for _ in range(240):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
            payload = json.loads(response.read().decode())
        (host_output / "logs" / "sglang_models.json").write_text(json.dumps(payload, indent=2) + "\n")
        ready = True
        break
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        time.sleep(5)
if not ready:
    raise SystemExit("slime server did not become ready within 20 minutes")
PY

    echo "[5/7] probing slime route with a direct chat completion"
    "${PYTEST_PYTHON}" - <<'PY' "${SLIME_PORT}" "${HOST_OUTPUT_DIR}" "${MODEL_NAME}"
import json
import pathlib
import sys
import urllib.request

port = sys.argv[1]
host_output = pathlib.Path(sys.argv[2])
model_name = sys.argv[3]
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(
        {
            "model": model_name,
            "messages": [{"role": "user", "content": "Reply with OK only."}],
            "temperature": 0.0,
            "max_tokens": 8,
        }
    ).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    payload = json.loads(response.read().decode())
(host_output / "logs" / "sglang_probe.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

    echo "[6/7] running slime route verification"
    rm -rf "${HOST_OUTPUT_DIR}/slime_outputs"
    LITELLM_DEFAULT_TIMEOUT="${LITELLM_TIMEOUT}" \
    SEARCH_SWE_SLIME_API_BASE="http://127.0.0.1:${SLIME_PORT}" \
    SEARCH_SWE_SLIME_API_KEY="EMPTY" \
    "${PYTEST_PYTHON}" "${REPO_ROOT}/agent/swe_agent/run/search_swe_agent.py" \
        --backend slime \
        --instance-id "${INSTANCE_ID}" \
        --subset rebench_v2 \
        --split train \
        --output-root "${HOST_OUTPUT_DIR}/slime_outputs" \
        --slime-model "${MODEL_NAME}" \
        --rubric-model "${MODEL_NAME}" \
        --judge-model "${MODEL_NAME}" \
        --workers 1 \
        --model-retry-attempts 1 \
        --m 2 \
        --k 20 \
        --p 1 \
        ${SEARCH_EXTRA_ARGS} \
        2>&1 | tee "${HOST_OUTPUT_DIR}/logs/slime_driver.log"

    "${PYTEST_PYTHON}" - <<'PY' "${HOST_OUTPUT_DIR}" "${INSTANCE_ID}"
import json
import pathlib
import sys

host_output = pathlib.Path(sys.argv[1])
instance_id = sys.argv[2]
candidate_roots = sorted((host_output / "slime_outputs").glob(f"rebench_v2_train_Qwen__Qwen3.5-9B/{instance_id}/*"))
if not candidate_roots:
    raise SystemExit("slime verification did not produce an output directory")
run_dir = candidate_roots[-1]
evaluation = json.loads((run_dir / "evaluation.json").read_text())
rubric_files = sorted((run_dir / "rubrics").glob("round_*.json"))
if not rubric_files:
    raise SystemExit("slime verification did not produce any rubric rounds")
rubrics = json.loads(rubric_files[-1].read_text())
model_patch = json.loads((run_dir / "model_patch.json").read_text())[instance_id]["model_patch"]
gt_rewards = {}
for judge_path in sorted((run_dir / "nodes").glob("*/judge.json")):
    payload = json.loads(judge_path.read_text())
    if "ground_truth_reward" in payload:
        gt_rewards[judge_path.parent.name] = payload["ground_truth_reward"]
node_count = sum(1 for _ in (run_dir / "node_index.jsonl").open())
summary = {
    "run_dir": str(run_dir),
    "evaluation": evaluation,
    "rubric_count": len(rubrics.get("generated", [])),
    "node_count": node_count,
    "model_patch_empty": model_patch == "",
    "model_patch_chars": len(model_patch),
    "ground_truth_rewards": gt_rewards,
}
if summary["rubric_count"] == 0:
    raise SystemExit("slime verification produced an empty rubric bank")
if summary["node_count"] <= 1:
    raise SystemExit("slime verification did not progress beyond the root node")
if summary["model_patch_empty"]:
    raise SystemExit("slime verification finished without a final patch")
(host_output / "logs" / "slime_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
else
    echo "[3/7] skipping slime route verification"
fi

echo "[7/7] checking for leaked benchmark containers"
LEAKED_CONTAINERS="$(docker ps --format '{{.Names}}' | grep '^swe_agent-' || true)"
if [[ -n "${LEAKED_CONTAINERS}" ]]; then
    echo "Found leaked benchmark containers:" >&2
    echo "${LEAKED_CONTAINERS}" >&2
    exit 1
fi

echo "step2 route verification completed"
echo "host artifacts: ${HOST_OUTPUT_DIR}"
