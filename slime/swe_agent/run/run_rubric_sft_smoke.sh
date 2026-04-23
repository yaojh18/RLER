#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-/workspace/rler:/workspace/rler/agent:/root/Megatron-LM}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

pkill -9 sglang >/dev/null 2>&1 || true
ray stop --force >/dev/null 2>&1 || true

cd /workspace/rler/slime
source /workspace/rler/slime/swe_agent/configs/qwen3.5-9B.sh

NUM_GPUS="${NUM_GPUS:-4}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-12288}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
PROMPT_DATA="${PROMPT_DATA:?PROMPT_DATA is required}"
DATASET_SIZE="${DATASET_SIZE:?DATASET_SIZE is required}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$DATASET_SIZE}"
MODEL_DIR="${MODEL_DIR:?MODEL_DIR is required}"
MODEL_TORCH_DIST_DIR="${MODEL_TORCH_DIST_DIR:?MODEL_TORCH_DIST_DIR is required}"
SAVE_DIR="${SAVE_DIR:?SAVE_DIR is required}"
WANDB_DIR="${WANDB_DIR:-${SAVE_DIR}/wandb}"

ray start --head --node-ip-address 127.0.0.1 --num-gpus "${NUM_GPUS}" --num-cpus "${RAY_NUM_CPUS}" --disable-usage-stats >/dev/null

python3 train_async.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_DIR}" \
  --ref-load "${MODEL_TORCH_DIST_DIR}" \
  --save "${SAVE_DIR}" \
  --save-interval 1 \
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout \
  --prompt-data "${PROMPT_DATA}" \
  --input-key messages \
  --rollout-shuffle \
  --num-epoch 1 \
  --rollout-batch-size "${DATASET_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --loss-type sft_loss \
  --loss-mask-type qwen3_5 \
  --calculate-per-token-loss \
  --disable-compute-advantages-and-returns \
  --debug-train-only \
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" \
  --sequence-parallel \
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}" \
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}" \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
  --optimizer adam \
  --lr 1e-5 \
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
  --wandb-project swe-agent-sft \
  --wandb-group rubric-sft-smoke \
  --wandb-dir "${WANDB_DIR}" \
  --disable-wandb-random-suffix
