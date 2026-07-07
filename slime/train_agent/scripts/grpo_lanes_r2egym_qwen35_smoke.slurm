#!/bin/bash
#SBATCH --job-name=rler-lanes-qwen35-r2e-smoke
#SBATCH --account=nemotron_rl_rm
#SBATCH --qos=normal
#SBATCH --partition=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/projects/nemotron_rl_rm/users/jiyao/tmp/train_async_validation/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nemotron/projects/nemotron_rl_rm/users/jiyao/tmp/train_async_validation/%x-%j.err

set -euo pipefail

# Lane async counterpart to grpo_naive_r2egym_qwen35_smoke.slurm.
# Shared smoke settings intentionally match the naive run:
# Qwen3.5-9B, 2 H100 nodes, actor 8 GPUs, rollout 8 GPUs, TP=2, CP=4,
# 4 rollouts, one R2E-Gym instance, and trajectory-search defaults.

ROOT=${ROOT:-/lustre/fsw/portfolios/nemotron/projects/nemotron_rl_rm/users/jiyao}
WORKSPACE=/workspace
SLIME_IMAGE=${SLIME_IMAGE:-$ROOT/enroot_images/slime/rler_slime_qwen35_apptainer_v8.sqsh}
INSTANCE_ID=${INSTANCE_ID:-orange3__2d9617bd0cb1f0ba61771258410ab8fae8e7e24d}
SUBSET=${SUBSET:-r2egym}
SPLIT=${SPLIT:-train}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3.5-9B}
RUN_NAME=${RUN_NAME:-lanes_${SLURM_JOB_ID}_r2egym_qwen35_smoke}
RUN_DIR_HOST=${RUN_DIR_HOST:-$ROOT/tmp/train_async_validation/$RUN_NAME}
RUN_DIR=$WORKSPACE/tmp/train_async_validation/$RUN_NAME
PROMPT_DATA_HOST=$RUN_DIR_HOST/r2egym_one.jsonl
PROMPT_DATA=$RUN_DIR/r2egym_one.jsonl
NODE_TABLE_HOST=$RUN_DIR_HOST/node_ip_table.txt
NODE_TABLE=$RUN_DIR/node_ip_table.txt
INNER_HOST=$RUN_DIR_HOST/launch_inside_container.sh
INNER=$RUN_DIR/launch_inside_container.sh
RAY_PORT=${RAY_PORT:-6379}
RUN_NODELIST=${RUN_NODELIST:-$SLURM_JOB_NODELIST}
RUN_NUM_NODES=$(scontrol show hostnames "$RUN_NODELIST" | wc -l)

HF_CHECKPOINT=${HF_CHECKPOINT:-$WORKSPACE/models/Qwen3.5-9B}
LOAD_DIR=${LOAD_DIR:-$WORKSPACE/models/Qwen3.5-9B_torch_dist}
REF_LOAD_DIR=${REF_LOAD_DIR:-$LOAD_DIR}
SAVE_DIR=${SAVE_DIR:-$RUN_DIR/policy_ckpt}
LANE_OUTPUT_ROOT=${LANE_OUTPUT_ROOT:-$RUN_DIR/lane_out}

REQUIRED_SIF_HOST=$ROOT/singularity_images/r2egym/namanjain12_orange3_final_2d9617bd0cb1f0ba61771258410ab8fae8e7e24d.sif

[ -n "${SLURM_JOB_ID:-}" ] || { echo "ERROR: run this script under sbatch/salloc"; exit 1; }
[ "$RUN_NUM_NODES" -eq 2 ] || { echo "ERROR: lane smoke requires exactly 2 run nodes, got $RUN_NUM_NODES"; exit 1; }
[ -f "$SLIME_IMAGE" ] || { echo "ERROR: missing slime image: $SLIME_IMAGE"; exit 1; }
[ -f "$REQUIRED_SIF_HOST" ] || { echo "ERROR: missing R2E-Gym SIF: $REQUIRED_SIF_HOST"; exit 1; }
[ -d "$ROOT/RLER" ] || { echo "ERROR: missing RLER checkout: $ROOT/RLER"; exit 1; }
[ -d "$ROOT/models/Qwen3.5-9B" ] || { echo "ERROR: missing HF checkpoint: $ROOT/models/Qwen3.5-9B"; exit 1; }
[ -d "$ROOT/models/Qwen3.5-9B_torch_dist" ] || { echo "ERROR: missing torch_dist checkpoint: $ROOT/models/Qwen3.5-9B_torch_dist"; exit 1; }

mkdir -p "$RUN_DIR_HOST" "$RUN_DIR_HOST/rollout_dumps" "$RUN_DIR_HOST/train_dumps"

printf '%s\n' \
  "{\"input\":[{\"role\":\"user\",\"content\":\"$INSTANCE_ID\"}],\"metadata\":{\"instance_id\":\"$INSTANCE_ID\",\"subset\":\"$SUBSET\",\"split\":\"$SPLIT\"}}" \
  > "$PROMPT_DATA_HOST"

: > "$NODE_TABLE_HOST"
while IFS= read -r node; do
  ip=$(getent ahostsv4 "$node" | awk '$1 ~ /^10[.]65[.]/ {print $1; exit}')
  [ -n "$ip" ] || { echo "ERROR: could not resolve 10.65.* IP for $node"; exit 1; }
  printf '%s %s %s\n' "$ip" "$node" "$ip" >> "$NODE_TABLE_HOST"
done < <(scontrol show hostnames "$RUN_NODELIST")
sort -V -o "$NODE_TABLE_HOST" "$NODE_TABLE_HOST"
HEAD_IP=$(awk 'NR == 1 {print $3}' "$NODE_TABLE_HOST")
HEAD_NODE=$(awk 'NR == 1 {print $2}' "$NODE_TABLE_HOST")

cat > "$INNER_HOST" <<'INNER'
#!/bin/bash
set -euo pipefail

config_value() {
  python3 - "$1" <<'PY'
import sys
key = sys.argv[1]
path = "/workspace/exp/config.md"
try:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(key + ":"):
                print(stripped.split(":", 1)[1].strip())
                break
except FileNotFoundError:
    pass
PY
}

HOST=$(hostname -s)
NODE_IP=$(awk -v host="$HOST" '$2 == host {print $3; exit}' "$NODE_TABLE")
[ -n "$NODE_IP" ] || { echo "ERROR: node $HOST not found in $NODE_TABLE"; exit 1; }

export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/RLER/slime:/workspace/RLER:/workspace/RLER/agent:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MSWEA_DOCKER_EXECUTABLE=/bin/false
export MSWEA_SINGULARITY_EXECUTABLE=/usr/bin/apptainer
export RLER_SIF_DIR=/workspace/singularity_images
export SWE_AGENT_SIF_DIR=/workspace/singularity_images
export SWE_AGENT_PYTHON=python3
export APPTAINER_CACHEDIR=/workspace/tmp/train_async_validation/apptainer/cache/$SLURM_JOB_ID
export SINGULARITY_CACHEDIR=$APPTAINER_CACHEDIR
export XDG_CONFIG_HOME=/workspace/tmp/train_async_validation/xdg_config/$SLURM_JOB_ID
export HF_HOME=/workspace/tmp/train_async_validation/hf_home
export HF_DATASETS_CACHE=/workspace/tmp/train_async_validation/hf_datasets_cache
export TMPDIR=/tmp/rler-$SLURM_JOB_ID
export SEARCH_SWE_SLIME_API_KEY=EMPTY
export SWE_AGENT_LANES_SKIP_NODE_IPS="$HEAD_IP"
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
export WANDB_API_KEY="${WANDB_API_KEY:-$(config_value "WanDB TOKEN")}"
export HF_TOKEN="${HF_TOKEN:-$(config_value "HF TOKEN")}"
export WANDB_ENTITY="${WANDB_ENTITY:-nemo-llm-service}"
export WANDB_PROJECT="${WANDB_PROJECT:-rler-train-smoke}"
export WANDB_MODE=online

[ -n "$WANDB_API_KEY" ] || { echo "ERROR: missing WanDB TOKEN in /workspace/exp/config.md"; exit 1; }
mkdir -p "$APPTAINER_CACHEDIR" "$XDG_CONFIG_HOME" "$TMPDIR" "$RUN_DIR/rollout_dumps" "$RUN_DIR/train_dumps" "$RUN_DIR/lane_out"

echo "[$(date)] node=$HOST ip=$NODE_IP head=$HEAD_NODE head_ip=$HEAD_IP"
python3 --version
apptainer --version

RAY_TMPDIR=/tmp/ray-rler-$SLURM_JOB_ID
mkdir -p "$RAY_TMPDIR"

if [ "$NODE_IP" = "$HEAD_IP" ]; then
  trap 'touch "$RUN_DIR/driver_done.flag" 2>/dev/null || true' EXIT
  ray stop --force >/dev/null 2>&1 || true
  ray start --head \
    --node-ip-address="$NODE_IP" \
    --port="$RAY_PORT" \
    --num-gpus=8 \
    --num-cpus="${SLURM_CPUS_PER_TASK:-128}" \
    --temp-dir="$RAY_TMPDIR" \
    --include-dashboard=false \
    --disable-usage-stats
  touch "$RUN_DIR/ray_head_ready.flag"

  for _ in $(seq 1 120); do
    nodes=$(python3 - <<'PY'
import os
import ray
ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True, logging_level="ERROR")
print(sum(1 for n in ray.nodes() if n.get("Alive")))
ray.shutdown()
PY
)
    echo "[$(date)] ray nodes=$nodes waiting=$RUN_NUM_NODES"
    [ "$nodes" -ge "$RUN_NUM_NODES" ] && break
    sleep 2
  done
  [ "$nodes" -ge "$RUN_NUM_NODES" ] || { echo "ERROR: Ray cluster did not reach $RUN_NUM_NODES nodes"; exit 1; }

  cd /workspace/RLER/slime
  set +e
  # Two-node validation. Formal: num_rollout=3000, actor/rollout=64/64,
  # TP/PP/CP=4/2/4, rollout_num_gpus_per_engine=4.
  python3 -m train_agent.run.grpo_async_lanes \
    --target policy \
    --rler-root /workspace/RLER \
    --ray-external \
    --prompt-data "$PROMPT_DATA" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --load-dir "$LOAD_DIR" \
    --ref-load-dir "$REF_LOAD_DIR" \
    --save-dir "$SAVE_DIR" \
    --search-output-root "$LANE_OUTPUT_ROOT" \
    --lanes-output-root "$LANE_OUTPUT_ROOT" \
    --num-nodes "$RUN_NUM_NODES" \
    --num-gpus-per-node 8 \
    --actor-num-gpus 8 \
    --actor-num-nodes 1 \
    --rollout-num-gpus 8 \
    --rollout-num-gpus-per-engine 2 \
    --tensor-model-parallel-size 2 \
    --context-parallel-size 4 \
    --rollout-batch-size 128 \
    --global-batch-size 1024 \
    --num-rollout 4 \
    --use-tis \
    --tis-clip 2.0 \
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std \
    --student-model "$MODEL_NAME" \
    --wandb-mode online \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-team "$WANDB_ENTITY" \
    --wandb-group "lanes-r2egym-$SLURM_JOB_ID-smoke" \
    --lanes-m 8 \
    --lanes-max-mid-cps 6 \
    --lanes-steps-per-round 20 \
    --lanes-step-limit 250 \
    --lanes-instance-workers 1 \
    --lanes-max-pending 1 \
    --lanes-gt-eval-workers 8 \
    --lanes-completion-max-tokens 16384 \
    --lanes-policy-temperature 1.0 \
    --lanes-policy-top-p 0.95 \
    --lanes-lane-b-temperature 1.0 \
    --lanes-lane-b-top-p 0.95 \
    --lanes-reward-kind delta \
    --lanes-disable-rubric \
    --save-debug-rollout-data "$RUN_DIR/rollout_dumps/{rollout_id}.pt" \
    --save-debug-train-data "$RUN_DIR/train_dumps/{rollout_id}_{rank}.pt"
  rc=$?
  set -e
  echo "[$(date)] driver rc=$rc"
  touch "$RUN_DIR/driver_done.flag"
  ray stop --force >/dev/null 2>&1 || true
  exit "$rc"
fi

while [ ! -f "$RUN_DIR/ray_head_ready.flag" ]; do
  sleep 1
done
ray stop --force >/dev/null 2>&1 || true
ray start \
  --address="$HEAD_IP:$RAY_PORT" \
  --node-ip-address="$NODE_IP" \
  --num-gpus=8 \
  --num-cpus="${SLURM_CPUS_PER_TASK:-128}" \
  --temp-dir="$RAY_TMPDIR" \
  --disable-usage-stats

while [ ! -f "$RUN_DIR/driver_done.flag" ]; do
  sleep 5
done
ray stop --force >/dev/null 2>&1 || true
INNER
chmod +x "$INNER_HOST"

echo "[$(date)] job=$SLURM_JOB_ID nodes=$RUN_NUM_NODES head=$HEAD_NODE head_ip=$HEAD_IP"
cat "$NODE_TABLE_HOST"
echo "[$(date)] run_dir=$RUN_DIR_HOST"

export RUN_DIR PROMPT_DATA NODE_TABLE HEAD_IP HEAD_NODE RAY_PORT RUN_NUM_NODES
export HF_CHECKPOINT LOAD_DIR REF_LOAD_DIR SAVE_DIR LANE_OUTPUT_ROOT MODEL_NAME

export ENROOT_CACHE_PATH=$ROOT/tmp/train_async_validation/enroot/cache
export ENROOT_DATA_PATH=$ROOT/tmp/train_async_validation/enroot/data
export ENROOT_RUNTIME_PATH=$ROOT/tmp/train_async_validation/enroot/runtime
export ENROOT_TEMP_PATH=/tmp/jiyao_enroot_tmp_$SLURM_JOB_ID
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$ENROOT_TEMP_PATH"

srun \
  --exclusive \
  --nodelist="$RUN_NODELIST" \
  --nodes="$RUN_NUM_NODES" \
  --ntasks="$RUN_NUM_NODES" \
  --ntasks-per-node=1 \
  --container-image="$SLIME_IMAGE" \
  --no-container-mount-home \
  --container-mounts="$ROOT:$WORKSPACE" \
  --container-workdir="$WORKSPACE" \
  bash "$INNER"
