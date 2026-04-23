#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

IMAGE="${IMAGE:-slimerl/slime:qwen35-grpo-fixed-20260415-v4}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-2,3,4,5}"
MODEL_VARIANT="${MODEL_VARIANT:-qwen35_9b}"
PYTEST_PYTHON="${PYTEST_PYTHON:-${REPO_ROOT}/agent/.venv/bin/python}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-$$}"
HOST_OUTPUT_DIR="${HOST_OUTPUT_DIR:-${REPO_ROOT}/slime/swe_agent/artifacts/step1_smoke_verify/${RUN_ID}}"
CONTAINER_WORKSPACE="/workspace/slime"
CONTAINER_OUTPUT_DIR="/verify_outputs"
SFT_CONTAINER_NAME="slime-step1-sft-$$"
GRPO_CONTAINER_NAME="slime-step1-grpo-$$"

case "${MODEL_VARIANT}" in
    qwen35_9b)
        MODEL_NAME="Qwen3.5-9B"
        ;;
    qwen3_8b)
        MODEL_NAME="Qwen3-8B"
        ;;
    *)
        echo "Unsupported MODEL_VARIANT=${MODEL_VARIANT}" >&2
        exit 2
        ;;
esac

cleanup() {
    docker rm -f "${SFT_CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker rm -f "${GRPO_CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

run_in_verify_container() {
    local container_name="$1"
    local stage_command="$2"
    docker run --rm \
        --name "${container_name}" \
        --runtime nvidia \
        -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
        --ipc=host \
        --shm-size=16g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "${REPO_ROOT}/slime:${CONTAINER_WORKSPACE}" \
        -v "${HOST_OUTPUT_DIR}:${CONTAINER_OUTPUT_DIR}" \
        -w "${CONTAINER_WORKSPACE}" \
        "${IMAGE}" \
        bash --noprofile --norc -lc "${stage_command}"
}

assert_host_file() {
    local path="$1"
    test -f "${path}"
}

assert_log_contains() {
    local path="$1"
    local pattern="$2"
    grep -F -q -- "${pattern}" "${path}"
}

echo "[0/5] preparing host output directory"
docker image inspect "${IMAGE}" >/dev/null
mkdir -p "${HOST_OUTPUT_DIR}/logs" "${HOST_OUTPUT_DIR}/models" "${HOST_OUTPUT_DIR}/data"

echo "[1/5] running local config pytest"
"${PYTEST_PYTHON}" -m pytest "${REPO_ROOT}/slime/tests/swe_agent/test_qwen35_9b_config.py" -q

echo "[2/5] running SFT smoke in owned docker container"
run_in_verify_container "${SFT_CONTAINER_NAME}" "
    rm -rf '${CONTAINER_OUTPUT_DIR}/sft' '${CONTAINER_OUTPUT_DIR}/logs/sft.log' &&
    PYTHON_BIN=python3 \
    MODEL_VARIANT='${MODEL_VARIANT}' \
    NUM_GPUS=2 \
    DATA_DIR='${CONTAINER_OUTPUT_DIR}/data' \
    MODEL_DIR='${CONTAINER_OUTPUT_DIR}/models/${MODEL_NAME}' \
    MODEL_TORCH_DIST_DIR='${CONTAINER_OUTPUT_DIR}/models/${MODEL_NAME}_torch_dist' \
    SAVE_DIR='${CONTAINER_OUTPUT_DIR}/sft' \
    ./swe_agent/scripts/run_qwen35_9b_gsm8k_sft_smoke.sh 2>&1 | tee '${CONTAINER_OUTPUT_DIR}/logs/sft.log'
"
assert_host_file "${HOST_OUTPUT_DIR}/sft/latest_checkpointed_iteration.txt"
assert_host_file "${HOST_OUTPUT_DIR}/sft/iter_0000001/common.pt"
assert_host_file "${HOST_OUTPUT_DIR}/sft/iter_0000001/metadata.json"
assert_host_file "${HOST_OUTPUT_DIR}/sft/rollout/global_dataset_state_dict_1.pt"
grep -x '1' "${HOST_OUTPUT_DIR}/sft/latest_checkpointed_iteration.txt" >/dev/null
assert_log_contains "${HOST_OUTPUT_DIR}/logs/sft.log" "step 1:"
assert_log_contains "${HOST_OUTPUT_DIR}/logs/sft.log" "successfully saved checkpoint from iteration"
echo "[2/5] SFT smoke passed"

echo "[3/5] running GRPO smoke in owned docker container"
run_in_verify_container "${GRPO_CONTAINER_NAME}" "
    rm -rf '${CONTAINER_OUTPUT_DIR}/grpo' '${CONTAINER_OUTPUT_DIR}/logs/grpo.log' &&
    PYTHON_BIN=python3 \
    MODEL_VARIANT='${MODEL_VARIANT}' \
    NUM_GPUS=2 \
    DATA_DIR='${CONTAINER_OUTPUT_DIR}/data' \
    MODEL_DIR='${CONTAINER_OUTPUT_DIR}/models/${MODEL_NAME}' \
    MODEL_TORCH_DIST_DIR='${CONTAINER_OUTPUT_DIR}/models/${MODEL_NAME}_torch_dist' \
    INIT_LOAD_DIR='${CONTAINER_OUTPUT_DIR}/sft' \
    SAVE_DIR='${CONTAINER_OUTPUT_DIR}/grpo' \
    DYNAMIC_SAMPLING_FILTER_PATH= \
    USE_COLOCATE=0 \
    ACTOR_NUM_GPUS_PER_NODE=2 \
    ROLLOUT_NUM_GPUS=0 \
    ./swe_agent/scripts/run_qwen35_9b_gsm8k_grpo_smoke.sh 2>&1 | tee '${CONTAINER_OUTPUT_DIR}/logs/grpo.log'
"
assert_host_file "${HOST_OUTPUT_DIR}/grpo/latest_checkpointed_iteration.txt"
assert_host_file "${HOST_OUTPUT_DIR}/grpo/iter_0000000/common.pt"
assert_host_file "${HOST_OUTPUT_DIR}/grpo/iter_0000000/metadata.json"
assert_host_file "${HOST_OUTPUT_DIR}/grpo/rollout/global_dataset_state_dict_0.pt"
grep -x '0' "${HOST_OUTPUT_DIR}/grpo/latest_checkpointed_iteration.txt" >/dev/null
assert_log_contains "${HOST_OUTPUT_DIR}/logs/grpo.log" "model.py:664 - step 0:"
assert_log_contains "${HOST_OUTPUT_DIR}/logs/grpo.log" "successfully saved checkpoint from iteration"
echo "[3/5] GRPO smoke passed"

echo "[4/5] checking container-local training artifacts are visible on host"
find "${HOST_OUTPUT_DIR}/sft" -maxdepth 2 -type f | sort
find "${HOST_OUTPUT_DIR}/grpo" -maxdepth 2 -type f | sort

echo "[5/5] checking no verification containers remain"
if docker ps -a --format '{{.Names}}' | grep -E "^(${SFT_CONTAINER_NAME}|${GRPO_CONTAINER_NAME})$" >/dev/null; then
    echo "verification containers still exist unexpectedly" >&2
    docker ps -a --format '{{.Names}} {{.Status}}' | grep -E "^(${SFT_CONTAINER_NAME}|${GRPO_CONTAINER_NAME}) "
    exit 1
fi

echo "step1 smoke verification completed"
echo "host artifacts: ${HOST_OUTPUT_DIR}"
