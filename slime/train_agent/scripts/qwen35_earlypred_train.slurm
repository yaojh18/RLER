#!/bin/bash
#SBATCH --job-name=q35-earlypred
#SBATCH --account=nemotron_rl_rm
#SBATCH --qos=normal
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=/lustre/fsw/portfolios/nemotron/projects/nemotron_rl_rm/users/jiyao/tmp/rler_earlypred_final/logs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nemotron/projects/nemotron_rl_rm/users/jiyao/tmp/rler_earlypred_final/logs/%x-%j.err

set -euo pipefail

# A Python subprocess timeout only terminates the direct Apptainer parent and
# can leave repository test children running indefinitely.  Make the existing
# environment timeout own the full process group so timed-out tool/evaluator
# commands cannot survive as node-wide memory leaks.
export RLER_SINGULARITY_RUNTIME_GUARD=1

# Final two-node Qwen3.5-9B configuration. Only cohort, actor-loss horizon and
# reward source differ across Baseline1, Baseline2 and Direct.
HOST_ROOT="${RLER_HOST_ROOT:-/lustre/fsw/portfolios/nemotron/projects/nemotron_rl_rm/users/jiyao}"
WORKSPACE=/workspace
IMAGE_HOST="${RUNTIME_IMAGE_HOST:-$HOST_ROOT/enroot_images/slime/rler_slime_qwen35_apptainer_v8.sqsh}"
RLER_SOURCE_REL="${RLER_SOURCE_REL:-RLER}"
LAUNCHER_ROOT_REL="$RLER_SOURCE_REL/slime/train_agent/scripts"
RUN_ROOT_REL="${RUN_ROOT_REL:-tmp/rler_earlypred_final/runs}"
DIRECT_ROLLOUT_CUTOFF="${DIRECT_ROLLOUT_CUTOFF:-40}"
DIRECT_HARD_ROLLOUT_CUTOFF="${DIRECT_HARD_ROLLOUT_CUTOFF:-0}"
DIRECT_SKIP_GT_EVALUATION="${DIRECT_SKIP_GT_EVALUATION:-0}"
case "$DIRECT_HARD_ROLLOUT_CUTOFF:$DIRECT_SKIP_GT_EVALUATION" in
  0:0|0:1|1:0|1:1) ;;
  *) echo "ERROR: Direct cutoff flags must be 0 or 1"; exit 2 ;;
esac
[ "$DIRECT_ROLLOUT_CUTOFF" -gt 0 ] || {
  echo "ERROR: DIRECT_ROLLOUT_CUTOFF must be positive"
  exit 2
}
case "$EXPERIMENT_METHOD" in
  baseline1)
    METHOD=baseline
    NAIVE_TRAIN_ASSISTANT_STEP_LIMIT=0
    SWE_AGENT_NAIVE_DIRECT_REWARD_MODE=none
    EPOCH_INSTANCE_COUNT=250
    ;;
  baseline2)
    METHOD=baseline
    NAIVE_TRAIN_ASSISTANT_STEP_LIMIT=40
    SWE_AGENT_NAIVE_DIRECT_REWARD_MODE=none
    EPOCH_INSTANCE_COUNT=250
    ;;
  direct)
    METHOD=judge_direct_naive_rollout40
    NAIVE_TRAIN_ASSISTANT_STEP_LIMIT="$DIRECT_ROLLOUT_CUTOFF"
    SWE_AGENT_NAIVE_DIRECT_REWARD_MODE=direct_judge
    case "$FOLD" in 0) EPOCH_INSTANCE_COUNT=222 ;; 1) EPOCH_INSTANCE_COUNT=226 ;; 2) EPOCH_INSTANCE_COUNT=230 ;; esac
    ;;
esac
if [ "$EXPERIMENT_METHOD" != direct ] \
  && { [ "$DIRECT_HARD_ROLLOUT_CUTOFF" = 1 ] \
    || [ "$DIRECT_SKIP_GT_EVALUATION" = 1 ]; }; then
  echo "ERROR: Direct cutoff flags require EXPERIMENT_METHOD=direct"
  exit 2
fi
NAIVE_ROLLOUT_STEP_LIMIT=250
if [ "$EXPERIMENT_METHOD" = direct ] \
  && [ "$DIRECT_HARD_ROLLOUT_CUTOFF" = 1 ]; then
  NAIVE_ROLLOUT_STEP_LIMIT="$DIRECT_ROLLOUT_CUTOFF"
fi
DATA_VARIANT=baseline
[ "$EXPERIMENT_METHOD" = direct ] && DATA_VARIANT=direct
FOLD_DATA_REL="tmp/rler_earlypred_final/inputs/fold_$FOLD/$DATA_VARIANT"
RLER_HOST="$HOST_ROOT/$RLER_SOURCE_REL"
RLER_WORKSPACE="$WORKSPACE/$RLER_SOURCE_REL"
LAUNCHER_ROOT_WORKSPACE="$WORKSPACE/$LAUNCHER_ROOT_REL"
SCRIPT_IN_CONTAINER="$LAUNCHER_ROOT_WORKSPACE/qwen35_earlypred_train.slurm"
MODEL_NAME=Qwen/Qwen3.5-9B
MODEL_CONFIG_NAME=qwen3.5-9B
HF_CHECKPOINT="${HF_CHECKPOINT:-$WORKSPACE/models/Qwen3.5-9B}"
LOAD_DIR="${LOAD_DIR:-$WORKSPACE/models/Qwen3.5-9B_torch_dist}"
REF_LOAD_DIR="${REF_LOAD_DIR:-$WORKSPACE/models/Qwen3.5-9B_torch_dist}"
FOLD_DATA="$WORKSPACE/$FOLD_DATA_REL"
TRAIN_DATA_REL="$FOLD_DATA_REL/train.jsonl"
SWE_AGENT_NAIVE_DIRECT_RUBRIC_BANK="$RLER_WORKSPACE/rubric/qwen/golden_reference_rubrics.json"
TRAIN_DATA="$WORKSPACE/$TRAIN_DATA_REL"
EVAL_SPLIT=validation
RAY_PORT="${RAY_PORT:-6379}"
RUN_PROFILE=actor1_cp4_64k_2n
TRAINING_RECIPE=dppo
TRAINING_RECIPE_TAG=-dppo
EXPECTED_LOG_PROBS_CHUNK_SIZE=256
ROLLOUT_MAX_RESPONSE_LEN=20480
ROLLOUT_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=128
VALIDATION_INSTANCE_WORKERS=50
CHECKPOINT_SAVE_INTERVAL=1
CHECKPOINT_RETAIN_LATEST=2
RLER_HOSTED_MODEL_CONTEXT_LENGTH=256000
RLER_HOSTED_MAX_COMPLETION_TOKENS=20480
DIRECT_JUDGE_MODEL=openai/azure/openai/gpt-5.6-luna
LITELLM_MAX_CONCURRENT_CALLS="${LITELLM_MAX_CONCURRENT_CALLS:-128}"
RLER_LITELLM_MIN_REQUEST_START_INTERVAL_SECONDS="${RLER_LITELLM_MIN_REQUEST_START_INTERVAL_SECONDS:-0}"
VALIDATION_TEMPERATURE=0.7
VALIDATION_TOP_P=0.95
VALIDATION_SAMPLES_PER_INSTANCE=4
VALIDATION_CONTEXT_LENGTH=128000
VALIDATION_COMPLETION_MAX_TOKENS=10240
OVER_SAMPLING_BATCH_SIZE=16
NUM_ROLLOUT=64
TRAIN_INSTANCE_BUDGET="${TRAIN_INSTANCE_BUDGET_OVERRIDE:-384}"
EVAL_INSTANCE_INTERVAL="${EVAL_INSTANCE_INTERVAL_OVERRIDE:-128}"
[ "$TRAIN_INSTANCE_BUDGET" -gt 0 ] || {
  echo "ERROR: TRAIN_INSTANCE_BUDGET must be positive"
  exit 2
}
[ "$EVAL_INSTANCE_INTERVAL" -gt 0 ] || {
  echo "ERROR: EVAL_INSTANCE_INTERVAL must be positive"
  exit 2
}
TRAIN_GT_EVAL_TIMEOUT=600
VALIDATION_GT_EVAL_TIMEOUT=600
TRAIN_SAMPLES_PER_PROMPT=8
TRAIN_POLICY_TOP_P=0.95
NAIVE_REWARD_KIND=joint
TRAINING_CONFIG_PATH="$RLER_WORKSPACE/slime/train_agent/configs/qwen3.5-9B.sh"
VALIDATION_HF_CONVERTER="$RLER_WORKSPACE/slime/tools/convert_torch_dist_to_hf.py"

if [ -z "${SLIME_WANDB_METRIC_ALLOWLIST:-}" ]; then
  SLIME_WANDB_METRIC_ALLOWLIST="train/step,train/pg_loss,train/entropy_loss,train/grad_norm,train/lr-pg_0,train/global_batch_size,train/pg_clipfrac,train/ppo_kl,train/tis,train/tis_abs,train/tis_clipfrac,train/train_rollout_logprob_abs_diff,train/dppo_mask_frac_kept,train/dppo_binary_divergence"
  SLIME_WANDB_METRIC_ALLOWLIST="$SLIME_WANDB_METRIC_ALLOWLIST,rollout/step,rollout/raw_reward,rollout/repetition_frac,rollout/truncated_ratio,perf/rollout_time,perf/actor_train_time,perf/update_weights_time,perf/train_wait_time,perf/wait_time_ratio"
  SLIME_WANDB_METRIC_ALLOWLIST="$SLIME_WANDB_METRIC_ALLOWLIST,eval/step,eval/train_instance_attempt,eval/attempted,eval/completed,eval/coverage,eval/errors,eval/incomplete,eval/resolved,eval/resolved_rate,eval/driver_error,eval/stale_1_instances,eval/stale_gt1_instances,eval/swebench_validation/truncated_ratio"
  SLIME_WANDB_METRIC_ALLOWLIST="$SLIME_WANDB_METRIC_ALLOWLIST,swe_agent/train_instances_attempted,swe_agent/train_epoch,swe_agent/last_validation_attempt,swe_agent/last_validation_scheduled_attempt,swe_agent/groups_attempted,swe_agent/groups_accepted_rate,swe_agent/groups_invalid_rate,swe_agent/groups_dynamic_filtered_rate,swe_agent/groups_excess_rate,swe_agent/stale_dropped_groups,swe_agent/stale_requeued_groups,swe_agent/stale_requeued_groups_total,swe_agent/failed_instances_total,swe_agent/truncated_oversized_samples_total"
  SLIME_WANDB_METRIC_ALLOWLIST="$SLIME_WANDB_METRIC_ALLOWLIST,swe_agent/reward_group_mean,swe_agent/reward_group_variance_mean,swe_agent/zero_variance_group_rate,swe_agent/sample_cont_steps_mean,swe_agent/sample_full_steps_mean,swe_agent/truncation_rate,swe_agent/infra_drop_rate,swe_agent/sample_gt_mean,swe_agent/sample_pass_at_0.5,swe_agent/instance_pass_at_0.5,swe_agent/submit_rate,swe_agent/ratio_zero_patch,swe_agent/format_error_rate_mean,swe_agent/ratio_fe_gated_trajectories,swe_agent/ratio_format_error_killed"
  SLIME_WANDB_METRIC_ALLOWLIST="$SLIME_WANDB_METRIC_ALLOWLIST,swe_agent/direct_judge_groups,swe_agent/direct_gt_pair_count,swe_agent/direct_gt_pairwise_accuracy,swe_agent/direct_predicted_zero_variance_rate,swe_agent/direct_explicit_abstain_rate,swe_agent/direct_variance_detector_count,swe_agent/direct_variance_detector_accuracy,swe_agent/direct_variance_detector_recall,swe_agent/direct_stage1_detector_count,swe_agent/direct_stage1_detector_accuracy,swe_agent/direct_stage1_detector_recall,swe_agent/direct_collapse_reward_rate,swe_agent/direct_collapse_reward_rollouts"
  SLIME_WANDB_METRIC_ALLOWLIST="$SLIME_WANDB_METRIC_ALLOWLIST,usage/event_step,usage/total_tokens_cumulative,usage/train_tokens_cumulative,usage/validation_tokens_cumulative,usage/qwen_tokens_cumulative"
fi

ROLLOUT_NUM_GPUS_PER_ENGINE=2
ROLLOUT_NUM_GPUS=8
TRAIN_TENSOR_PARALLEL_SIZE=2
TRAIN_PIPELINE_PARALLEL_SIZE=1
TRAIN_CONTEXT_PARALLEL_SIZE=4
NAIVE_INSTANCE_WORKERS=16
NAIVE_MAX_PENDING=16
MODEL_CONTEXT_LENGTH=65536
MAX_TOKENS_PER_GPU=16384
# The policy service accommodates validation's 128K requests even
# when training examples remain capped at 64K.  The collector and actor limits
# below continue to use MODEL_CONTEXT_LENGTH.
SGLANG_CONTEXT_LENGTH=$VALIDATION_CONTEXT_LENGTH
SGLANG_MEM_FRACTION_STATIC=0.85
SGLANG_MAX_PREFILL_TOKENS=16384
ACTOR_NUM_NODES=1
ACTOR_NUM_GPUS=8
TRAIN_MODEL_PARALLEL_SIZE=$((
  TRAIN_TENSOR_PARALLEL_SIZE
  * TRAIN_PIPELINE_PARALLEL_SIZE
  * TRAIN_CONTEXT_PARALLEL_SIZE
))
config_value() {
  python3 - "$1" <<'PY'
import sys

key = sys.argv[1]
try:
    with open("/workspace/exp/config.md", "r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith(key + ":"):
                print(stripped.split(":", 1)[1].strip())
                break
except FileNotFoundError:
    pass
PY
}

inside_container() {
  local host node_ip
  host=$(hostname -s)
  node_ip=$(getent ahostsv4 "$host" | awk '$1 ~ /^10[.]65[.]/ {print $1; exit}')
  if [ -z "$node_ip" ]; then
    node_ip=$(getent ahostsv4 "$host" | awk 'NR == 1 {print $1}')
  fi
  [ -n "$node_ip" ] || { echo "ERROR: cannot resolve node IP for $host"; exit 2; }

  export PYTHONUNBUFFERED=1
  export PYTHONPATH="$RLER_WORKSPACE/slime:$RLER_WORKSPACE:$RLER_WORKSPACE/agent:/root/Megatron-LM:${PYTHONPATH:-}"
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export MSWEA_DOCKER_EXECUTABLE=/bin/false
  export MSWEA_SINGULARITY_EXECUTABLE=/usr/bin/apptainer
  export RLER_SIF_DIR="$WORKSPACE/singularity_images"
  export SWE_AGENT_SIF_DIR="$WORKSPACE/singularity_images"
  export SWE_AGENT_PYTHON=python3
  export SEARCH_SWE_SLIME_API_KEY=EMPTY
  export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
  export RLER_USAGE_LEDGER_PATH="$USAGE_LEDGER"
  export RLER_HF_DATASET_LOCAL_ONLY=1
  export RLER_LITELLM_EXPLICIT_RETRIES="${RLER_LITELLM_EXPLICIT_RETRIES:-8}"
  export RLER_LITELLM_RETRY_BASE_SECONDS="${RLER_LITELLM_RETRY_BASE_SECONDS:-10}"
  export RLER_LITELLM_RETRY_MAX_SECONDS="${RLER_LITELLM_RETRY_MAX_SECONDS:-60}"
  export RLER_LITELLM_RETRY_JITTER_FRACTION="${RLER_LITELLM_RETRY_JITTER_FRACTION:-0.25}"
  export RLER_LITELLM_MIN_REQUEST_START_INTERVAL_SECONDS
  unset RLER_LITELLM_GLOBAL_COORDINATION_DIR
  unset RLER_LITELLM_GLOBAL_MAX_CONCURRENT_CALLS
  export LITELLM_MAX_CONCURRENT_CALLS
  export RLER_HOSTED_MODEL_CONTEXT_LENGTH
  export RLER_HOSTED_MAX_COMPLETION_TOKENS
  export RLER_POLICY_MODEL_CONTEXT_LENGTH="$MODEL_CONTEXT_LENGTH"
  export RLER_POLICY_VERSION_STATE_PATH="$RUN_DIR/policy_version.json"
  export SWE_AGENT_MODEL_CONTEXT_LENGTH="$MODEL_CONTEXT_LENGTH"
  export SWE_AGENT_NAIVE_SKIP_NODE_IPS="$HEAD_IP"
  export APPTAINER_CACHEDIR="$WORKSPACE/tmp/rler_earlypred_final/cache/apptainer/$SLURM_JOB_ID"
  export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
  export XDG_CONFIG_HOME="$WORKSPACE/tmp/rler_earlypred_final/cache/xdg/$SLURM_JOB_ID"
  export HF_HOME="$WORKSPACE/tmp/rler_earlypred_final/cache/huggingface"
  export HF_DATASETS_CACHE="$WORKSPACE/tmp/rler_earlypred_final/cache/hf_datasets"
  export TMPDIR="/tmp/rler-earlypred-$SLURM_JOB_ID"
  export WANDB_API_KEY="$(config_value "WanDB personal TOKEN")"
  export HF_TOKEN="${HF_TOKEN:-$(config_value "HF TOKEN")}"
  export WANDB_ENTITY="jihany2-university-of-washington"
  export WANDB_PROJECT="${WANDB_PROJECT:-rler-earlypred-final}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export TRAIN_TENSOR_PARALLEL_SIZE
  export TRAIN_PIPELINE_PARALLEL_SIZE
  export TRAIN_CONTEXT_PARALLEL_SIZE
  export TRAIN_MODEL_PARALLEL_SIZE
  if [ "$METHOD" = "judge_direct_naive_rollout40" ]; then
    export SWE_AGENT_NAIVE_DIRECT_JUDGE_API_KEY="${SWE_AGENT_NAIVE_DIRECT_JUDGE_API_KEY:-$(config_value "Model API TOKEN")}"
  fi

  mkdir -p \
    "$APPTAINER_CACHEDIR" \
    "$XDG_CONFIG_HOME" \
    "$HF_HOME" \
    "$HF_DATASETS_CACHE" \
    "$TMPDIR" \
    "$RUN_DIR/search"

  echo "[$(date -Is)] method=$METHOD host=$host node_ip=$node_ip head=$HEAD_NODE head_ip=$HEAD_IP"

  local ray_tmp="/tmp/ray-rler-earlypred-$SLURM_JOB_ID"
  mkdir -p "$ray_tmp"
  ray stop --force >/dev/null 2>&1 || true

  if [ "$host" = "$HEAD_NODE" ]; then
    cleanup_head() {
      ray stop --force >/dev/null 2>&1 || true
    }
    trap cleanup_head EXIT
    ray start \
      --head \
      --node-ip-address="$node_ip" \
      --port="$RAY_PORT" \
      --num-gpus=8 \
      --num-cpus="${SLURM_CPUS_PER_TASK:-128}" \
      --temp-dir="$ray_tmp" \
      --include-dashboard=false \
      --disable-usage-stats
    touch "$RUN_DIR/ray_head_ready.flag"

    local nodes=0
    for _ in $(seq 1 180); do
      nodes=$(python3 -c 'import os, ray; ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True, logging_level="ERROR"); print(sum(1 for node in ray.nodes() if node.get("Alive"))); ray.shutdown()')
      echo "[$(date -Is)] ray_alive_nodes=$nodes expected=$RUN_NUM_NODES"
      [ "$nodes" -ge "$RUN_NUM_NODES" ] && break
      sleep 2
    done
    [ "$nodes" -ge "$RUN_NUM_NODES" ] || {
      echo "ERROR: Ray cluster reached $nodes/$RUN_NUM_NODES nodes"
      exit 3
    }

    local ray_node_layout actor_node_ips rollout_node_ips
    ray_node_layout=$(python3 - <<'PY'
import ipaddress
import os

import ray

ray.init(
    address=os.environ["RAY_ADDRESS"],
    ignore_reinit_error=True,
    logging_level="ERROR",
)
gpu_ips = sorted(
    {
        node["NodeManagerAddress"]
        for node in ray.nodes()
        if node.get("Alive") and float(node.get("Resources", {}).get("GPU", 0)) > 0
    },
    key=ipaddress.ip_address,
)
ray.shutdown()
print("actor=" + gpu_ips[0])
print("rollout=" + ",".join(gpu_ips[1:]))
PY
)
    actor_node_ips=$(printf '%s\n' "$ray_node_layout" | awk -F= '$1 == "actor" {print $2}')
    rollout_node_ips=$(printf '%s\n' "$ray_node_layout" | awk -F= '$1 == "rollout" {print $2}')
    export SWE_AGENT_NAIVE_SKIP_NODE_IPS="$actor_node_ips"
    export SWE_AGENT_NAIVE_ROLLOUT_NODE_IPS="$rollout_node_ips"
    echo "[$(date -Is)] actor_node_ips=$actor_node_ips rollout_node_ips=$rollout_node_ips"

    cd "$RLER_WORKSPACE"

    local -a common_args method_args
    local entry dynamic_filter_path
    dynamic_filter_path=slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    if [[ "$METHOD" == judge_direct_* ]]; then
      dynamic_filter_path=slime.rollout.filter_hub.dynamic_sampling_filters.check_direct_judge_variance
    fi
    common_args=(
      --rler-root "$RLER_WORKSPACE"
      --ray-external
      --prompt-data "$TRAIN_DATA"
      --eval-instance-interval "$EVAL_INSTANCE_INTERVAL"
      --eval-prompt-data swebench_validation "$FOLD_DATA/$EVAL_SPLIT.jsonl"
      --n-samples-per-eval-prompt "$VALIDATION_SAMPLES_PER_INSTANCE"
      --train-instance-budget "$TRAIN_INSTANCE_BUDGET"
      --hf-checkpoint "$HF_CHECKPOINT"
      --load-dir "$LOAD_DIR"
      --ref-load-dir "$REF_LOAD_DIR"
      --save-dir "$RUN_DIR/policy_ckpt"
      --auto-resume
      --usage-ledger "$USAGE_LEDGER"
      --student-model "$MODEL_NAME"
      --num-nodes "$RUN_NUM_NODES"
      --num-gpus-per-node 8
      --actor-num-gpus "$ACTOR_NUM_GPUS"
      --actor-num-nodes "$ACTOR_NUM_NODES"
      --rollout-num-gpus "$ROLLOUT_NUM_GPUS"
      --rollout-num-gpus-per-engine "$ROLLOUT_NUM_GPUS_PER_ENGINE"
      --tensor-model-parallel-size "$TRAIN_TENSOR_PARALLEL_SIZE"
      --pipeline-model-parallel-size "$TRAIN_PIPELINE_PARALLEL_SIZE"
      --context-parallel-size "$TRAIN_CONTEXT_PARALLEL_SIZE"
      --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
      --log-probs-chunk-size "$EXPECTED_LOG_PROBS_CHUNK_SIZE"
      --rollout-max-context-len "$MODEL_CONTEXT_LENGTH"
      --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
      --sglang-context-length "$SGLANG_CONTEXT_LENGTH"
      --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC"
      --num-rollout "$NUM_ROLLOUT"
      --save-interval "$CHECKPOINT_SAVE_INTERVAL"
      --checkpoint-retain-latest "$CHECKPOINT_RETAIN_LATEST"
      --dynamic-sampling-filter-path "$dynamic_filter_path"
      --wandb-mode "$WANDB_MODE"
      --wandb-project "$WANDB_PROJECT"
      --wandb-team "$WANDB_ENTITY"
      --wandb-group "$RUN_NAME"
      --wandb-dir "$RUN_DIR/wandb"
      --validation-instance-workers "$VALIDATION_INSTANCE_WORKERS"
      --validation-step-limit 250
      --validation-completion-max-tokens "$VALIDATION_COMPLETION_MAX_TOKENS"
      --validation-context-length "$VALIDATION_CONTEXT_LENGTH"
      --validation-gt-eval-timeout "$VALIDATION_GT_EVAL_TIMEOUT"
      --validation-temperature "$VALIDATION_TEMPERATURE"
      --validation-top-p "$VALIDATION_TOP_P"
      --model-context-length "$MODEL_CONTEXT_LENGTH"
    )
    common_args+=(
      --require-train-instance-budget-exhaustion
      --defer-instance-validation
    )
    case "$METHOD" in
      baseline)
        entry=train_agent.run.grpo_async_naive
        method_args=(
          --naive-output-root "$RUN_DIR/search"
          --naive-m "$TRAIN_SAMPLES_PER_PROMPT"
          --naive-step-limit 250
          --naive-train-assistant-step-limit "$NAIVE_TRAIN_ASSISTANT_STEP_LIMIT"
          --naive-instance-workers "$NAIVE_INSTANCE_WORKERS"
          --naive-max-pending "$NAIVE_MAX_PENDING"
          --naive-gt-eval-workers 8
          --naive-gt-eval-timeout "$TRAIN_GT_EVAL_TIMEOUT"
          --naive-rollout-pool-size 8
          --naive-completion-max-tokens 20480
          --naive-policy-temperature 1.0
          --naive-policy-top-p "$TRAIN_POLICY_TOP_P"
          --naive-reward-kind "$NAIVE_REWARD_KIND"
          --naive-joint-alpha 1.0
          --naive-all-pass-reward 1.0
        )
        ;;
      judge_direct_naive_rollout40)
        entry=train_agent.run.grpo_async_naive
        method_args=(
          --naive-output-root "$RUN_DIR/search"
          --naive-m "$TRAIN_SAMPLES_PER_PROMPT"
          --naive-step-limit "$NAIVE_ROLLOUT_STEP_LIMIT"
          --naive-train-assistant-step-limit "$NAIVE_TRAIN_ASSISTANT_STEP_LIMIT"
          --naive-instance-workers "$NAIVE_INSTANCE_WORKERS"
          --naive-max-pending "$NAIVE_MAX_PENDING"
          --naive-gt-eval-workers 8
          --naive-gt-eval-timeout "$TRAIN_GT_EVAL_TIMEOUT"
          --naive-rollout-pool-size 8
          --naive-completion-max-tokens 20480
          --naive-policy-temperature 1.0
          --naive-policy-top-p "$TRAIN_POLICY_TOP_P"
          --naive-reward-kind joint
          --naive-joint-alpha 1.0
          --naive-all-pass-reward 1.0
          --naive-direct-reward-mode "$SWE_AGENT_NAIVE_DIRECT_REWARD_MODE"
          --naive-direct-rubric-bank "$SWE_AGENT_NAIVE_DIRECT_RUBRIC_BANK"
          --naive-direct-collapse-reward-margin 0.5
          --naive-direct-judge-context-length 256000
          --naive-direct-judge-max-tokens 20480
          --naive-direct-judge-temperature 0.02
          --naive-direct-judge-top-p 1.0
          --naive-direct-judge-model "$DIRECT_JUDGE_MODEL"
          --naive-direct-judge-api-base https://inference-api.nvidia.com/v1
        )
        if [ "$DIRECT_SKIP_GT_EVALUATION" = 1 ]; then
          method_args+=(--naive-skip-gt-evaluation)
        fi
        ;;
    esac

    local hf_export_pid hf_export_stop hf_export_rc=0
    hf_export_stop="$RUN_DIR/validation_hf_checkpoints/.training_finished"
    mkdir -p "$RUN_DIR/validation_hf_checkpoints"
    rm -f "$hf_export_stop"
    nice -n 10 python3 \
      "$RLER_WORKSPACE/slime/train_agent/scripts/export_validation_checkpoints.py" \
      --checkpoint-root "$RUN_DIR/policy_ckpt" \
      --output-root "$RUN_DIR/validation_hf_checkpoints" \
      --origin-hf-dir "$HF_CHECKPOINT" \
      --converter "$VALIDATION_HF_CONVERTER" \
      --stop-file "$hf_export_stop" \
      --release-source-marker \
      >> "$RUN_DIR/validation_hf_export.log" 2>&1 &
    hf_export_pid=$!
    echo "[$(date -Is)] validation HF exporter started pid=$hf_export_pid"

    set +e
    (
      cd "$RLER_WORKSPACE/slime"
      python3 -m "$entry" \
        "${common_args[@]}" \
        --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
        --over-sampling-batch-size "$OVER_SAMPLING_BATCH_SIZE" \
        --global-batch-size "$GLOBAL_BATCH_SIZE" \
        "${method_args[@]}"
    ) 2>&1 | tee -a "$RUN_DIR/driver.log"
    local driver_rc=${PIPESTATUS[0]}
    set -e
    touch "$hf_export_stop"
    set +e
    wait "$hf_export_pid"
    hf_export_rc=$?
    set -e
    if [ "$hf_export_rc" -ne 0 ]; then
      echo "WARNING: asynchronous validation HF export finished with rc=$hf_export_rc; training artifacts remain valid"
    fi
    printf '%s\n' "$driver_rc" > "$RUN_DIR/training_driver.rc"
    printf '%s\n' "$driver_rc" > "$RUN_DIR/driver.rc"
    if [ "$driver_rc" -eq 0 ]; then
      touch "$RUN_DIR/driver_done.flag"
    else
      touch "$RUN_DIR/driver_failed.flag"
    fi
    echo "[$(date -Is)] method=$METHOD driver_rc=$driver_rc"
    exit "$driver_rc"
  fi

  for _ in $(seq 1 240); do
    [ -f "$RUN_DIR/ray_head_ready.flag" ] && break
    sleep 1
  done
  [ -f "$RUN_DIR/ray_head_ready.flag" ] || {
    echo "ERROR: timed out waiting for Ray head"
    exit 3
  }
  cleanup_worker() {
    ray stop --force >/dev/null 2>&1 || true
  }
  trap cleanup_worker EXIT
  ray start \
    --address="$HEAD_IP:$RAY_PORT" \
    --node-ip-address="$node_ip" \
    --num-gpus=8 \
    --num-cpus="${SLURM_CPUS_PER_TASK:-128}" \
    --temp-dir="$ray_tmp" \
    --disable-usage-stats

  while [ ! -f "$RUN_DIR/driver_done.flag" ] \
    && [ ! -f "$RUN_DIR/driver_failed.flag" ]; do
    sleep 5
  done
  local worker_rc
  worker_rc=$(tr -d '[:space:]' < "$RUN_DIR/driver.rc")
  exit "${worker_rc:-4}"
}

if [ "${1:-}" = "inside" ]; then
  inside_container
  exit $?
fi

python3 "$RLER_HOST/slime/train_agent/scripts/prepare_fold_data.py" \
  --workspace-root "$HOST_ROOT" \
  --fold "$FOLD" \
  --output-root "$HOST_ROOT/tmp/rler_earlypred_final/inputs"
RUN_NODELIST="$SLURM_JOB_NODELIST"
RUN_NUM_NODES=2
HEAD_NODE=$(scontrol show hostnames "$RUN_NODELIST" | head -n 1)
HEAD_IP=$(getent ahostsv4 "$HEAD_NODE" | awk '$1 ~ /^10[.]65[.]/ {print $1; exit}')
if [ -z "$HEAD_IP" ]; then
  HEAD_IP=$(getent ahostsv4 "$HEAD_NODE" | awk 'NR == 1 {print $1}')
fi
[ -n "$HEAD_IP" ] || { echo "ERROR: cannot resolve head IP"; exit 2; }

EXPERIMENT_LABEL="f$FOLD-q35-$EXPERIMENT_METHOD"
RUN_PROFILE_TAG=a1c4-64k2n
RUN_SERIES="${RUN_SERIES:-earlypred-q35-final}"
NAIVE_STEP_TAG=
if [ "$EXPERIMENT_METHOD" = direct ] \
  && [ "$DIRECT_HARD_ROLLOUT_CUTOFF" = 1 ]; then
  NAIVE_STEP_TAG="-cutoff${DIRECT_ROLLOUT_CUTOFF}"
elif { [ "$METHOD" = "baseline" ] \
  || [ "$METHOD" = "judge_direct_naive_rollout40" ]; } \
  && [ "$NAIVE_TRAIN_ASSISTANT_STEP_LIMIT" -gt 0 ]; then
  NAIVE_STEP_TAG="-trainfirst${NAIVE_TRAIN_ASSISTANT_STEP_LIMIT}"
fi
[ "$DIRECT_SKIP_GT_EVALUATION" = 0 ] \
  || NAIVE_STEP_TAG="${NAIVE_STEP_TAG}-nogt"
RUN_NAME="${RUN_SERIES}-${EXPERIMENT_LABEL}${NAIVE_STEP_TAG}${TRAINING_RECIPE_TAG}-${RUN_PROFILE_TAG}-rb${ROLLOUT_BATCH_SIZE}-gb${GLOBAL_BATCH_SIZE}-nr${NUM_ROLLOUT}-vt${VALIDATION_TEMPERATURE}-ctx${MODEL_CONTEXT_LENGTH}-mt${MAX_TOKENS_PER_GPU}-lpc${EXPECTED_LOG_PROBS_CHUNK_SIZE}"
RUN_DIR_HOST="$HOST_ROOT/$RUN_ROOT_REL/$RUN_NAME"
RUN_DIR="$WORKSPACE/$RUN_ROOT_REL/$RUN_NAME"
USAGE_LEDGER="$RUN_DIR/usage.jsonl"
mkdir -p "$RUN_DIR_HOST"
if [ -s "$RUN_DIR_HOST/usage.jsonl" ]; then
  export RLER_USAGE_RESUME=1
else
  unset RLER_USAGE_RESUME
fi
export RLER_USAGE_ATTEMPT_NONCE="job-${SLURM_JOB_ID}-restart-${SLURM_RESTART_COUNT:-0}"
export WANDB_RUN_ID
WANDB_RUN_ID="$RUN_NAME"
# These are per-allocation control files, not experiment artifacts.
rm -f \
  "$RUN_DIR_HOST/ray_head_ready.flag" \
  "$RUN_DIR_HOST/driver_done.flag" \
  "$RUN_DIR_HOST/driver_failed.flag" \
  "$RUN_DIR_HOST/driver.rc" \
  "$RUN_DIR_HOST/training_driver.rc"

export METHOD EXPERIMENT_METHOD FOLD RUN_PROFILE RUN_NAME RUN_DIR USAGE_LEDGER
export FOLD_DATA_REL TRAIN_DATA_REL TRAIN_DATA
export DIRECT_ROLLOUT_CUTOFF DIRECT_HARD_ROLLOUT_CUTOFF
export DIRECT_SKIP_GT_EVALUATION NAIVE_ROLLOUT_STEP_LIMIT
export SWE_AGENT_NAIVE_DIRECT_REWARD_MODE
export TRAINING_RECIPE TRAINING_CONFIG_PATH TRAIN_SAMPLES_PER_PROMPT
export VALIDATION_HF_CONVERTER
export TRAIN_POLICY_TOP_P
export NAIVE_TRAIN_ASSISTANT_STEP_LIMIT
export MODEL_CONTEXT_LENGTH SGLANG_CONTEXT_LENGTH SGLANG_MEM_FRACTION_STATIC
export SGLANG_MAX_PREFILL_TOKENS
export MAX_TOKENS_PER_GPU
export ROLLOUT_MAX_RESPONSE_LEN
export RLER_LITELLM_MIN_REQUEST_START_INTERVAL_SECONDS
export RLER_HOSTED_MODEL_CONTEXT_LENGTH
export RLER_HOSTED_MAX_COMPLETION_TOKENS
export LITELLM_MAX_CONCURRENT_CALLS
export ACTOR_NUM_NODES ACTOR_NUM_GPUS EXPECTED_LOG_PROBS_CHUNK_SIZE
export TRAIN_TENSOR_PARALLEL_SIZE TRAIN_PIPELINE_PARALLEL_SIZE
export TRAIN_CONTEXT_PARALLEL_SIZE TRAIN_MODEL_PARALLEL_SIZE
export ROLLOUT_NUM_GPUS ROLLOUT_NUM_GPUS_PER_ENGINE
export NAIVE_INSTANCE_WORKERS NAIVE_MAX_PENDING
export VALIDATION_INSTANCE_WORKERS
export VALIDATION_TEMPERATURE VALIDATION_TOP_P
export VALIDATION_SAMPLES_PER_INSTANCE VALIDATION_CONTEXT_LENGTH
export VALIDATION_COMPLETION_MAX_TOKENS
export ROLLOUT_BATCH_SIZE OVER_SAMPLING_BATCH_SIZE GLOBAL_BATCH_SIZE NUM_ROLLOUT
export TRAIN_INSTANCE_BUDGET EPOCH_INSTANCE_COUNT
export EVAL_INSTANCE_INTERVAL
export TRAIN_GT_EVAL_TIMEOUT VALIDATION_GT_EVAL_TIMEOUT
export CHECKPOINT_SAVE_INTERVAL CHECKPOINT_RETAIN_LATEST
export NAIVE_REWARD_KIND
export HEAD_NODE HEAD_IP RAY_PORT RUN_NODELIST RUN_NUM_NODES
export MODEL_NAME MODEL_CONFIG_NAME DIRECT_JUDGE_MODEL
export SWE_AGENT_NAIVE_DIRECT_RUBRIC_BANK
export SLIME_WANDB_METRIC_ALLOWLIST
export HF_CHECKPOINT LOAD_DIR REF_LOAD_DIR
export FOLD_DATA

export ENROOT_CACHE_PATH="$HOST_ROOT/tmp/rler_earlypred_final/cache/enroot/cache"
export ENROOT_DATA_PATH="$HOST_ROOT/tmp/rler_earlypred_final/cache/enroot/data"
export ENROOT_RUNTIME_PATH="/tmp/rler-earlypred-enroot-runtime-$SLURM_JOB_ID"
export ENROOT_TEMP_PATH="/tmp/rler-earlypred-enroot-tmp-$SLURM_JOB_ID"
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH"

echo "[$(date -Is)] job=$SLURM_JOB_ID fold=$FOLD method=$EXPERIMENT_METHOD nodes=$RUN_NODELIST run_dir=$RUN_DIR_HOST"
requeue_for_walltime() {
  echo "[$(date -Is)] received pre-timeout USR1; requeueing job $SLURM_JOB_ID from the latest complete checkpoint pair"
  if scontrol requeue "$SLURM_JOB_ID"; then
    exit 0
  fi
  echo "ERROR: scontrol requeue failed for job $SLURM_JOB_ID"
  exit 4
}
trap requeue_for_walltime USR1

set +e
srun \
  --exclusive \
  --nodelist="$RUN_NODELIST" \
  --nodes="$RUN_NUM_NODES" \
  --ntasks="$RUN_NUM_NODES" \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  --container-image="$IMAGE_HOST" \
  --no-container-mount-home \
  --container-mounts="$HOST_ROOT:$WORKSPACE" \
  --container-workdir="$WORKSPACE" \
  bash "$SCRIPT_IN_CONTAINER" inside &
srun_pid=$!
wait "$srun_pid"
srun_rc=$?
set -e
trap - USR1
driver_rc=
if [ -s "$RUN_DIR_HOST/driver.rc" ]; then
  driver_rc=$(tr -d '[:space:]' < "$RUN_DIR_HOST/driver.rc")
fi
restart_count="${SLURM_RESTART_COUNT:-0}"
if [ "$driver_rc" = "137" ] \
  && [ "$restart_count" -lt 2 ]; then
  echo "[$(date -Is)] driver received SIGKILL; requeueing job $SLURM_JOB_ID (restart $restart_count/2)"
  if scontrol requeue "$SLURM_JOB_ID"; then
    exit 0
  fi
  echo "ERROR: automatic SIGKILL requeue failed for job $SLURM_JOB_ID"
fi
exit "$srun_rc"
