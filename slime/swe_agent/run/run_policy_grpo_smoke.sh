#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-/workspace/rler:/workspace/rler/agent:/root/Megatron-LM}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

pkill -9 sglang >/dev/null 2>&1 || true
ray stop --force >/dev/null 2>&1 || true

cd /workspace/rler/slime
source /workspace/rler/slime/swe_agent/configs/qwen3.5-9B.sh

NUM_GPUS="${NUM_GPUS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-12288}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
DEBUG_ROLLOUT_PATH="${DEBUG_ROLLOUT_PATH:?DEBUG_ROLLOUT_PATH is required}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:?ROLLOUT_BATCH_SIZE is required}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$ROLLOUT_BATCH_SIZE}"
HF_CHECKPOINT="${HF_CHECKPOINT:?HF_CHECKPOINT is required}"
LOAD_DIR="${LOAD_DIR:?LOAD_DIR is required}"
REF_LOAD_DIR="${REF_LOAD_DIR:-$LOAD_DIR}"
SAVE_DIR="${SAVE_DIR:?SAVE_DIR is required}"
WANDB_DIR="${WANDB_DIR:-${SAVE_DIR}/wandb}"

ray start --head --node-ip-address 127.0.0.1 --num-gpus "${NUM_GPUS}" --num-cpus "${RAY_NUM_CPUS}" --disable-usage-stats >/dev/null

python3 train_async.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --load "${LOAD_DIR}" \
  --ref-load "${REF_LOAD_DIR}" \
  --save "${SAVE_DIR}" \
  --save-interval 1 \
  --num-rollout 1 \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --load-debug-rollout-data "${DEBUG_ROLLOUT_PATH}" \
  --custom-convert-samples-to-train-data-path swe_agent.run.search_grpo_convert.convert_samples_to_train_data \
  --loss-type policy_loss \
  --loss-mask-type qwen3_5 \
  --advantage-estimator grpo \
  --eps-clip 0.2 \
  --entropy-coef 0.0 \
  --calculate-per-token-loss \
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" \
  --sequence-parallel \
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}" \
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}" \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --min-lr 1e-6 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --use-wandb \
  --wandb-mode offline \
  --wandb-project swe-agent-grpo \
  --wandb-group policy-grpo-smoke \
  --wandb-dir "${WANDB_DIR}" \
  --disable-wandb-random-suffix
