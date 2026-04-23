#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

IMAGE="${IMAGE:-slimerl/slime:qwen35-route-fixed-20260416-v1}"
GPUS="${GPUS:-0,1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/slime/swe_agent/artifacts/step1_smoke_verify/qwen35_fullverify_20260415/models/Qwen3.5-9B}"
SLIME_PORT="${SLIME_PORT:-8021}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-80960}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-$$}"
HOST_OUTPUT_DIR="${HOST_OUTPUT_DIR:-${REPO_ROOT}/slime/swe_agent/artifacts/step4_data_pipeline_verify/${RUN_ID}}"
CONTAINER_NAME="slime-step4-pipeline-${RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/agent/.venv/bin/python}"

cleanup() {
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "${HOST_OUTPUT_DIR}/logs"
docker image inspect "${IMAGE}" >/dev/null
test -x "${PYTHON_BIN}"
test -d "${MODEL_DIR}"

case "${MODEL_DIR}" in
    "${REPO_ROOT}"/*)
        CONTAINER_MODEL_DIR="/workspace/rler/${MODEL_DIR#${REPO_ROOT}/}"
        ;;
    *)
        echo "MODEL_DIR must live under ${REPO_ROOT}: ${MODEL_DIR}" >&2
        exit 2
        ;;
esac

start_server() {
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
                2>&1 | tee /verify_outputs/logs/sglang_server.log
        " >/dev/null

    "${PYTHON_BIN}" - <<'PY' "${SLIME_PORT}" "${HOST_OUTPUT_DIR}"
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

port = sys.argv[1]
host_output = pathlib.Path(sys.argv[2])
for _ in range(240):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
            payload = json.loads(response.read().decode())
        (host_output / "logs" / "sglang_models.json").write_text(json.dumps(payload, indent=2) + "\n")
        break
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        time.sleep(5)
else:
    raise SystemExit("slime server did not become ready within 20 minutes")
PY
}

echo "[1/6] targeted pytest"
"${PYTHON_BIN}" -m pytest \
    "${REPO_ROOT}/slime/tests/swe_agent/test_teacher_student_export.py" \
    "${REPO_ROOT}/agent/tests/test_search_swe_agent.py" \
    "${REPO_ROOT}/agent/tests/test_run_swe_agent.py" -q \
    2>&1 | tee "${HOST_OUTPUT_DIR}/logs/pytest.log"

echo "[2/6] starting owned slime server for SFT log export"
start_server
SEARCH_SWE_SLIME_API_BASE="http://127.0.0.1:${SLIME_PORT}" \
SEARCH_SWE_SLIME_API_KEY="EMPTY" \
GEMINI_API_KEY="${GEMINI_API_KEY:-}" \
PYTHONPATH="${REPO_ROOT}/agent:${REPO_ROOT}" \
"${PYTHON_BIN}" "${REPO_ROOT}/slime/swe_agent/scripts/verify_step4_data_pipeline.py" \
    --mode sft \
    --instance-id psf__requests-1142 \
    --output-root "${HOST_OUTPUT_DIR}/outputs" \
    2>&1 | tee "${HOST_OUTPUT_DIR}/logs/sft_driver.log"

echo "[3/6] restarting slime server for RL runtime capture"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
start_server
SEARCH_SWE_SLIME_API_BASE="http://127.0.0.1:${SLIME_PORT}" \
SEARCH_SWE_SLIME_API_KEY="EMPTY" \
PYTHONPATH="${REPO_ROOT}/agent:${REPO_ROOT}" \
"${PYTHON_BIN}" "${REPO_ROOT}/slime/swe_agent/scripts/verify_step4_data_pipeline.py" \
    --mode rl \
    --instance-id psf__requests-1142 \
    --output-root "${HOST_OUTPUT_DIR}/outputs" \
    2>&1 | tee "${HOST_OUTPUT_DIR}/logs/rl_driver.log"

echo "[4/6] summarizing outputs"
find "${HOST_OUTPUT_DIR}/outputs" -maxdepth 3 -type f | sort | tee "${HOST_OUTPUT_DIR}/logs/output_files.txt" >/dev/null

echo "[5/6] resource checks"
docker ps --format '{{.Names}}' | tee "${HOST_OUTPUT_DIR}/logs/docker_ps_after.txt"
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader 2>/dev/null \
    | tee "${HOST_OUTPUT_DIR}/logs/gpu_processes_after.txt" >/dev/null || true

echo "[6/6] artifacts ready"
echo "artifacts: ${HOST_OUTPUT_DIR}"
