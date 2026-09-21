#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace
RLER=$ROOT/${RLER_SOURCE_REL:-RLER}
TASKS=${EVAL_TASKS_CONFIG:?}
OUTPUT_ROOT=${EVALUATION_OUTPUT_ROOT:-$ROOT/tmp/rler_earlypred_final/evaluations}
worker_rank=${SLURM_PROCID:?}
restart=${SLURM_RESTART_COUNT:-0}
claim_root=$OUTPUT_ROOT/.claims/job-$SLURM_JOB_ID-restart-$restart
mkdir -p "$claim_root"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$RLER/slime:$RLER:$RLER/agent:/root/Megatron-LM:${PYTHONPATH:-}"
export MSWEA_DOCKER_EXECUTABLE=/bin/false
export MSWEA_SINGULARITY_EXECUTABLE=/usr/bin/apptainer
export RLER_SIF_DIR=$ROOT/singularity_images
export SWE_AGENT_SIF_DIR=$ROOT/singularity_images
export RLER_HF_DATASET_LOCAL_ONLY=1
export HF_HOME=$ROOT/tmp/rler_earlypred_final/cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export APPTAINER_CACHEDIR=$ROOT/tmp/rler_earlypred_final/cache/apptainer/$SLURM_JOB_ID/node-$worker_rank
export SINGULARITY_CACHEDIR=$APPTAINER_CACHEDIR
export XDG_CONFIG_HOME=$ROOT/tmp/rler_earlypred_final/cache/xdg/$SLURM_JOB_ID/node-$worker_rank
export TMPDIR=/tmp/rler-q35-eval-$SLURM_JOB_ID-$worker_rank
mkdir -p "$OUTPUT_ROOT" "$APPTAINER_CACHEDIR" "$XDG_CONFIG_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

task_count=$(python3 - "$TASKS" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))["tasks"]))
PY
)

run_task() {
  local task_index=$1
  local values label checkpoint_rel fold split
  mapfile -t values < <(python3 - "$TASKS" "$task_index" <<'PY'
import json, sys
task=json.load(open(sys.argv[1], encoding="utf-8"))["tasks"][int(sys.argv[2])]
for key in ("label", "checkpoint", "fold", "split"):
    print(task[key])
PY
)
  label=${values[0]}
  checkpoint_rel=${values[1]}
  fold=${values[2]}
  split=${values[3]}
  local source=$ROOT/$checkpoint_rel
  local manifest=$ROOT/tmp/rler_earlypred_final/inputs/fold_$fold/baseline/$split.jsonl
  local task_root=$OUTPUT_ROOT/$label
  local results=$task_root/results
  local logs=$task_root/logs
  local hf_root=$source
  mkdir -p "$task_root" "$logs"

  if [ -s "$results/summary.json" ] && python3 - "$results/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if summary.get("status") == "complete" else 1)
PY
  then
    return 0
  fi

  if [ ! -s "$source/model.safetensors.index.json" ] && [ ! -s "$source/model.safetensors" ]; then
    hf_root=$task_root/hf_checkpoint
    if [ ! -s "$hf_root/model.safetensors.index.json" ] && [ ! -s "$hf_root/model.safetensors" ]; then
      local staging=$task_root/.hf_checkpoint-$SLURM_JOB_ID-r$restart-$worker_rank
      python3 "$RLER/slime/tools/convert_torch_dist_to_hf.py" \
        --input-dir "$source" --output-dir "$staging" \
        --origin-hf-dir "$ROOT/models/Qwen3.5-9B" --add-missing-from-origin-hf \
        >"$logs/conversion.log" 2>&1
      mv "$staging" "$hf_root"
    fi
  fi

  local server_pids=()
  cleanup_servers() {
    local pid
    for pid in "${server_pids[@]:-}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    for _ in {1..20}; do
      local alive=0
      for pid in "${server_pids[@]:-}"; do kill -0 "$pid" 2>/dev/null && alive=1 || true; done
      [ "$alive" -eq 0 ] && break
      sleep 1
    done
    for pid in "${server_pids[@]:-}"; do
      kill -KILL -- "-$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    done
  }
  trap cleanup_servers RETURN INT TERM
  local ports=(31000 31010 31020 31030)
  for replica in 0 1 2 3; do
    local gpu0=$((replica * 2))
    local gpu1=$((gpu0 + 1))
    local port=${ports[$replica]}
    CUDA_VISIBLE_DEVICES="$gpu0,$gpu1" setsid python3 -m sglang.launch_server \
      --model-path "$hf_root" --served-model-name Qwen/Qwen3.5-9B \
      --host 127.0.0.1 --port "$port" --tensor-parallel-size 2 \
      --context-length 128000 --reasoning-parser qwen3 --mem-fraction-static 0.85 \
      --cuda-graph-max-bs 16 --max-running-requests 16 \
      --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
      --watchdog-timeout 3600 >"$logs/sglang-$replica.log" 2>&1 &
    server_pids+=("$!")
  done
  local deadline=$((SECONDS + 900))
  for replica in 0 1 2 3; do
    local port=${ports[$replica]}
    while ! curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null; do
      kill -0 "${server_pids[$replica]}" 2>/dev/null || {
        tail -n 200 "$logs/sglang-$replica.log" >&2 || true
        return 30
      }
      [ "$SECONDS" -lt "$deadline" ] || return 31
      sleep 5
    done
  done

  local reuse=()
  [ ! -d "$results" ] || reuse=(--reuse-policy-root "$results")
  python3 "$RLER/slime/train_agent/scripts/run_checkpoint_evaluation.py" \
    --manifest "$manifest" \
    --output-root "$results" --policy-version "$label" \
    --endpoint http://127.0.0.1:31000 --endpoint http://127.0.0.1:31010 \
    --endpoint http://127.0.0.1:31020 --endpoint http://127.0.0.1:31030 \
    --samples-per-instance 4 --step-limit 250 --context-length 128000 \
    --completion-max-tokens 10240 --temperature 0.7 --top-p 0.95 \
    --gt-eval-timeout 600 --policy-workers 8 --gt-eval-workers 16 \
    --infrastructure-retries 3 "${reuse[@]}" >"$logs/evaluation.log" 2>&1
  local evaluation_rc=$?
  cleanup_servers
  trap - RETURN INT TERM
  return "$evaluation_rc"
}

while true; do
  claimed=
  for ((task_index=0; task_index<task_count; task_index++)); do
    claim=$claim_root/task-$task_index.claim
    if mkdir "$claim" 2>/dev/null; then
      claimed=$task_index
      break
    fi
  done
  [ -n "$claimed" ] || break
  set +e
  run_task "$claimed" >"$claim/worker.out" 2>"$claim/worker.err"
  status=$?
  set -e
  printf 'status=%s\nfinished_at=%s\n' "$status" "$(date -Is)" >"$claim/result.txt"
  [ "$status" -eq 0 ] || exit "$status"
done
