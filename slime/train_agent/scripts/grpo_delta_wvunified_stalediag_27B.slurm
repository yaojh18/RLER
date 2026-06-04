#!/bin/bash
#SBATCH --job-name=grpo-delta-wvunified-stalediag-27b
#SBATCH --account=ads_cpt
#SBATCH --qos=ads_cpt
#SBATCH --partition=a4
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/home/sihanzeng_meta_com/rler_runs/_slurm-logs/grpo-delta-wvunified-stalediag-27b-%j.log

# 5-node GRPO async NAIVE for Qwen3.5-27B (delta-stalediag recipe from 59762).
#
# Topology delta vs 59762 (9B, 4 nodes):
#   * NUM_NODES 4 -> 5
#   * ACTOR_NUM_NODES 1 -> 2 (16 actor GPUs)
#   * ROLLOUT_NUM_GPUS unchanged at 24 (3 rollout nodes)
#   * Actor parallelism: TP=4 PP=2 CP=2 DP=1 (TP*PP*CP*DP = 16). The 9B's
#     TP=1 CP=4 DP=2 layout can't host a 27B weight+Adam shard on 80GB H100;
#     TP=4*PP=2 = weight-shard factor 8 -> ~54GB per rank for fp32 master +
#     Adam states, fits with room for activations under recompute=full.
#   * sglang per-engine TP 1 -> 2 (27B BF16 = 54GB; one GPU can't hold weights
#     + an 80k KV cache). 24 rollout GPUs / TP=2 -> 12 inference engines.
#   * --decoder-last-pipeline-num-layers 30 mirrors slime's reference 27B
#     launcher (first PP stage gets 34 layers including embedding, last 30
#     including output head).
#
# Per-sample sequence cap is max_tokens_per_gpu * CP_SIZE. CP dropped 4 -> 2
# halves the cap; we keep max_tokens_per_gpu at 16384 (= 32k per sample) and
# lower sglang context to 32k to match. Long-tail rollouts that pushed past
# 80k under 9B WILL truncate here -- this is the cost of running 27B with
# 5 nodes instead of slime's reference 7+. Bump CP=4 (requires +2 actor
# nodes) once the recipe is otherwise validated.
#
# All other reward/search knobs (delta, format-error penalty, infra drop
# rate, KL anchor, variance filter, debug dumps) are unchanged from 59762.

set -u

# ====================================================================
# Tunable knobs -- mirror 59762 for everything except topology.
# ====================================================================
NAIVE_M=8
NAIVE_STEP_LIMIT=120
NAIVE_INSTANCE_WORKERS=24
NAIVE_GT_EVAL_WORKERS=8
NAIVE_COMPLETION_MAX_TOKENS=4096
NAIVE_POLICY_TEMPERATURE=1.0
NAIVE_POLICY_TOP_P=0.95
NAIVE_FALLBACK_PATCH_PENALTY=0.5
NAIVE_NO_ACTION_PATCH_PENALTY=0.0
NAIVE_REWARD_KIND=scaled_delta
NAIVE_FORMAT_ERROR_PER_STEP_PENALTY=0.02
NAIVE_FORMAT_OK_GATE_THRESHOLD=0.5

ROLLOUT_BATCH_SIZE=16
OVER_SAMPLING_BATCH_SIZE=24
NUM_ROLLOUT=120
NUM_NODES=5
NUM_GPUS_PER_NODE=8

# 27B-specific topology
ACTOR_NUM_NODES=2
ACTOR_NUM_GPUS=16
ROLLOUT_NUM_GPUS=24
GRPO_TARGET=policy
GRPO_MODEL_NAME="Qwen/Qwen3.5-27B"
MODEL_CONFIG_NAME=qwen3.5-27B
PROMPT_BUDGET=1150

# Actor parallel layout
TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=2
CONTEXT_PARALLEL_SIZE=2
# 27B has 64 decoder layers. PP=2 even split would be 32/32; slime's
# reference launcher uses 34/30 to balance the embedding (first stage) and
# output head (last stage) overhead.
DECODER_LAST_PIPELINE_NUM_LAYERS=30

# sglang per-engine TP. 27B BF16 weights = 54 GB; TP=2 -> 27 GB shard +
# room for KV cache at 32k context. 24 rollout GPUs / 2 = 12 engines.
ROLLOUT_NUM_GPUS_PER_ENGINE=2

# Memory / context tuning for 27B + CP=2.
# per-sample seq cap = MAX_TOKENS_PER_GPU * CP_SIZE = 32768
MAX_TOKENS_PER_GPU=16384
SGLANG_CONTEXT_LENGTH=32768
# ====================================================================

# === IP-sorted node ordering ===
# slime's placement_group.sort_key orders bundles by IPv4 numerically, so
# the lowest-IP nodes become the actor. Pick HEAD_NODE = lowest-IP node so
# the Ray head + driver land on an actor node, and pass BOTH actor IPs to
# SWE_AGENT_NAIVE_SKIP_NODE_IPS so neither hosts SWE-rebench instance
# subprocesses (the round-1 OOM vector that motivated the head-skip env var
# in the first place).
ALL_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
declare -A NODE_TO_IP
SORTED_IPS=$(
  for n in $ALL_NODES; do
    ip=$(getent ahostsv4 $n | awk '{print $1; exit}')
    [ -z "$ip" ] && ip=$(srun --nodes=1 --ntasks=1 -w $n hostname -I | awk '{print $1}')
    echo "$ip $n"
    NODE_TO_IP[$n]=$ip
  done | sort -t. -k1,1n -k2,2n -k3,3n -k4,4n
)
HEAD_IP=$(echo "$SORTED_IPS" | head -n 1 | awk '{print $1}')
HEAD_NODE=$(echo "$SORTED_IPS" | head -n 1 | awk '{print $2}')
ACTOR_IPS_CSV=$(echo "$SORTED_IPS" | head -n $ACTOR_NUM_NODES | awk '{print $1}' | paste -sd, -)
RAY_PORT=6379

RLER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/RLER
BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
SUBSET=$BASE/datasets/v2_python_train_plus_sweep_middle_v2
RUN_DIR=/home/sihanzeng_meta_com/rler_runs/${SLURM_JOB_ID}-grpo-delta-wvunified-stalediag-27b
SLURM_LOGS=/home/sihanzeng_meta_com/rler_runs/_slurm-logs
mkdir -p $RUN_DIR $SLURM_LOGS $RUN_DIR/_logs

# 27B HF + torch_dist locations -- override at sbatch time if these change.
HF_CHECKPOINT=${HF_CHECKPOINT:-/mnt/lustre/metavmds0lstre/teams/mrs_ranking_core_modeling/daili1/deepeyes/models/Qwen3.5-27B}
LOAD_DIR=${LOAD_DIR:-/mnt/lustre/metavmds0lstre/share/sihanzeng_meta_com/slime_qwen35/ckpt/Qwen3.5-27B_torch_dist_latest}
SAVE_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler_ckpts/${SLURM_JOB_ID}-grpo-delta-wvunified-stalediag-27b/policy_ckpt
PROMPT_DATA=$RUN_DIR/grpo_prompts.jsonl

echo "[$(date)] jobid=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES head=$HEAD_NODE head_ip=$HEAD_IP"
echo "Actor IPs (skipped from naive worker pool): $ACTOR_IPS_CSV"
echo "Sorted node list (by IP):"
echo "$SORTED_IPS"

[ ! -d "$HF_CHECKPOINT" ] && { echo "ERROR: HF_CHECKPOINT=$HF_CHECKPOINT does not exist (run convert_qwen3.5-27B_hf_to_torch_dist.slurm prereq?)"; exit 1; }
[ ! -d "$LOAD_DIR" ] && { echo "ERROR: LOAD_DIR=$LOAD_DIR does not exist (run convert_qwen3.5-27B_hf_to_torch_dist.slurm prereq?)"; exit 1; }

mkdir -p $RUN_DIR/_snapshot
( cd $RLER && git rev-parse HEAD 2>/dev/null || sl id -i ) > $RUN_DIR/_snapshot/RLER_HEAD.txt 2>&1
cp $0 $RUN_DIR/_snapshot/ 2>/dev/null || true

CACHE_BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/.cache/grpo-${SLURM_JOB_ID}
mkdir -p $CACHE_BASE
chmod -R 777 $CACHE_BASE 2>/dev/null || true

# Generate prompt data on the head (NFS-shared via Lustre)
python3 - <<EOF
import json, os
from datasets import load_from_disk
DENYLIST = {
    "getmoto__moto-7283",
    "kozea__weasyprint-2405",
    "slackapi__python-slack-sdk-1638",
}
candidate = "$SUBSET/train" if os.path.isdir("$SUBSET/train") else "$SUBSET"
ds = load_from_disk(candidate)
if hasattr(ds, "keys") and "train" in ds.keys():
    ds = ds["train"]
ids = sorted([r["instance_id"] for r in ds if r["instance_id"] not in DENYLIST])[:$PROMPT_BUDGET]
with open("$PROMPT_DATA", "w") as f:
    for iid in ids:
        f.write(json.dumps({
            "input": [{"role": "user", "content": iid}],
            "metadata": {"instance_id": iid, "subset": "$SUBSET", "split": "train"},
        }) + "\n")
print("Wrote", len(ids), "prompt rows to $PROMPT_DATA (denylist drops:", len(DENYLIST), ")")
EOF

# ----- Per-node setup script (run inside container on each node) -----
PER_NODE=$RUN_DIR/_snapshot/per_node.sh
cat > $PER_NODE <<TOP
#!/bin/bash
set -u
GRPO_TARGET="$GRPO_TARGET"
GRPO_MODEL_NAME="$GRPO_MODEL_NAME"
MODEL_CONFIG_NAME="$MODEL_CONFIG_NAME"
NAIVE_M=$NAIVE_M
NAIVE_STEP_LIMIT=$NAIVE_STEP_LIMIT
NAIVE_INSTANCE_WORKERS=$NAIVE_INSTANCE_WORKERS
NAIVE_GT_EVAL_WORKERS=$NAIVE_GT_EVAL_WORKERS
NAIVE_COMPLETION_MAX_TOKENS=$NAIVE_COMPLETION_MAX_TOKENS
NAIVE_POLICY_TEMPERATURE=$NAIVE_POLICY_TEMPERATURE
NAIVE_POLICY_TOP_P=$NAIVE_POLICY_TOP_P
NAIVE_FALLBACK_PATCH_PENALTY=$NAIVE_FALLBACK_PATCH_PENALTY
NAIVE_NO_ACTION_PATCH_PENALTY=$NAIVE_NO_ACTION_PATCH_PENALTY
NAIVE_REWARD_KIND=$NAIVE_REWARD_KIND
NAIVE_FORMAT_ERROR_PER_STEP_PENALTY=$NAIVE_FORMAT_ERROR_PER_STEP_PENALTY
NAIVE_FORMAT_OK_GATE_THRESHOLD=$NAIVE_FORMAT_OK_GATE_THRESHOLD
ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE
OVER_SAMPLING_BATCH_SIZE=$OVER_SAMPLING_BATCH_SIZE
NUM_ROLLOUT=$NUM_ROLLOUT
NUM_NODES=$NUM_NODES
NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE
ACTOR_NUM_NODES=$ACTOR_NUM_NODES
ACTOR_NUM_GPUS=$ACTOR_NUM_GPUS
ROLLOUT_NUM_GPUS=$ROLLOUT_NUM_GPUS
TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE
PIPELINE_PARALLEL_SIZE=$PIPELINE_PARALLEL_SIZE
CONTEXT_PARALLEL_SIZE=$CONTEXT_PARALLEL_SIZE
DECODER_LAST_PIPELINE_NUM_LAYERS=$DECODER_LAST_PIPELINE_NUM_LAYERS
ROLLOUT_NUM_GPUS_PER_ENGINE=$ROLLOUT_NUM_GPUS_PER_ENGINE
MAX_TOKENS_PER_GPU=$MAX_TOKENS_PER_GPU
SGLANG_CONTEXT_LENGTH=$SGLANG_CONTEXT_LENGTH
ACTOR_IPS_CSV="$ACTOR_IPS_CSV"
export SLURM_JOB_NAME="$SLURM_JOB_NAME"
export SLURM_JOB_ID="$SLURM_JOB_ID"
TOP
cat >> $PER_NODE <<'EOF'
HEAD_NODE=$1
HEAD_IP=$2
RAY_PORT=$3
CACHE_BASE=$4
RUN_DIR=$5
RLER=$6
BASE=$7
PROMPT_DATA=$8
HF_CHECKPOINT=$9
LOAD_DIR=${10}
SAVE_DIR=${11}

NODE=$(hostname)
NODE_CACHE=/mnt/localssd/grpo-${SLURM_JOB_ID}/$NODE
mkdir -p $NODE_CACHE/{flashinfer,triton,hf,torch} $SAVE_DIR
export FLASHINFER_WORKSPACE_DIR=$NODE_CACHE/flashinfer
export TRITON_CACHE_DIR=$NODE_CACHE/triton
export HF_HOME=$NODE_CACHE/hf
export TORCH_HOME=$NODE_CACHE/torch
export XDG_CACHE_HOME=$NODE_CACHE
export HOME=$NODE_CACHE

export MSWEA_DOCKER_EXECUTABLE=$BASE/bin/docker
export DOCKER_LUSTRE_TARDIR=$BASE/docker_tarballs
export DOCKER_LUSTRE_REAL=/usr/bin/docker
export MSWEA_COST_TRACKING=ignore_errors
export SEARCH_SWE_SLIME_API_KEY=EMPTY
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1

WANDB_KEY_FILE=/mnt/lustre/metavmds0lstre/share/sihanzeng_meta_com/secrets/wandb_api_key
if [ -r "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY=$(cat "$WANDB_KEY_FILE" | tr -d '[:space:]')
    export WANDB_MODE=online
    export WANDB_ENTITY=chenqinghan2003-university-of-washington
else
    echo "[node=$NODE] WARN: wandb key file missing at $WANDB_KEY_FILE, staying offline"
    export WANDB_MODE=offline
fi

export PYTHONPATH=$BASE/SWE-rebench-V2:$RLER:$RLER/agent:$RLER/slime:/root/Megatron-LM:${PYTHONPATH:-}
export EXTRA_PYTHONPATH=$BASE/SWE-rebench-V2:$RLER:$RLER/agent

export SWE_AGENT_NAIVE_OUTPUT_ROOT=$RUN_DIR/naive_out
export SWE_AGENT_GRPO_OUTPUT_ROOT=$RUN_DIR/naive_out
export SWE_AGENT_GRPO_TARGET=$GRPO_TARGET
export SWE_AGENT_GRPO_MODEL_NAME=$GRPO_MODEL_NAME
export SWE_AGENT_NAIVE_M=$NAIVE_M
export SWE_AGENT_NAIVE_STEP_LIMIT=$NAIVE_STEP_LIMIT
export SWE_AGENT_NAIVE_INSTANCE_WORKERS=$NAIVE_INSTANCE_WORKERS
export SWE_AGENT_NAIVE_MAX_PENDING=$OVER_SAMPLING_BATCH_SIZE
export NAIVE_INFRA_DROP_RATE_LIMIT=0.7
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$RUN_DIR/_logs/nccl_trace
export NCCL_DEBUG=WARN
# Skip BOTH actor nodes (computed in the launcher from IP-sorted SLURM
# allocation) -- no SWE-rebench instance subprocesses on any actor node.
export SWE_AGENT_NAIVE_SKIP_NODE_IPS=$ACTOR_IPS_CSV
export SWE_AGENT_NAIVE_GT_EVAL_WORKERS=$NAIVE_GT_EVAL_WORKERS
export SWE_AGENT_NAIVE_COMPLETION_MAX_TOKENS=$NAIVE_COMPLETION_MAX_TOKENS
export SWE_AGENT_NAIVE_POLICY_TEMPERATURE=$NAIVE_POLICY_TEMPERATURE
export SWE_AGENT_NAIVE_POLICY_TOP_P=$NAIVE_POLICY_TOP_P
export SWE_AGENT_NAIVE_FALLBACK_PATCH_PENALTY=$NAIVE_FALLBACK_PATCH_PENALTY
export SWE_AGENT_NAIVE_NO_ACTION_PATCH_PENALTY=$NAIVE_NO_ACTION_PATCH_PENALTY
export SWE_AGENT_NAIVE_REWARD_KIND=$NAIVE_REWARD_KIND
export SWE_AGENT_NAIVE_FORMAT_ERROR_PER_STEP_PENALTY=$NAIVE_FORMAT_ERROR_PER_STEP_PENALTY
export SWE_AGENT_NAIVE_FORMAT_OK_GATE_THRESHOLD=$NAIVE_FORMAT_OK_GATE_THRESHOLD
export CALCULATE_PER_TOKEN_LOSS=0

MY_IP=$(hostname -I | awk '{print $1}')
echo "[node=$NODE] my_ip=$MY_IP head=$HEAD_NODE head_ip=$HEAD_IP"

MEM_LOG=$RUN_DIR/_logs/mem-${NODE}.log
echo "[node=$NODE] mem-monitor starting -> $MEM_LOG"
(
  while true; do
    {
      printf "==== %s ====\n" "$(date '+%Y-%m-%d %H:%M:%S')"
      free -g | head -3
      echo "-- top 6 RSS processes --"
      ps -eo pid,rss,comm --sort=-rss --no-headers | head -6 | awk '{printf "  pid=%s rss_kB=%s cmd=%s\n", $1, $2, $3}'
    } >> $MEM_LOG 2>&1
    sleep 30
  done
) &
MEM_MONITOR_PID=$!
trap "kill $MEM_MONITOR_PID 2>/dev/null" EXIT

SGLANG_USAGE=/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/usage_processor.py
if [ -f "$SGLANG_USAGE" ]; then
  if ! grep -q 'meta_info.get."prompt_tokens"' "$SGLANG_USAGE"; then
    echo "[node=$NODE] patching $SGLANG_USAGE for KeyError-safe prompt_tokens"
    sed -i 's/responses\[i\]\["meta_info"\]\["prompt_tokens"\]/responses[i]["meta_info"].get("prompt_tokens", 0)/g' "$SGLANG_USAGE"
    sed -i 's/responses\[i\]\["meta_info"\]\["completion_tokens"\]/responses[i]["meta_info"].get("completion_tokens", 0)/g' "$SGLANG_USAGE"
  else
    echo "[node=$NODE] sglang usage_processor already patched"
  fi
else
  echo "[node=$NODE] WARNING: $SGLANG_USAGE not found; KeyError patch skipped"
fi

DONE_FLAG=$RUN_DIR/_logs/driver_done.flag

if [ "$NODE" = "$HEAD_NODE" ]; then
    echo "[node=$NODE] role=HEAD: starting ray head"
    ray start --head --node-ip-address=$MY_IP --num-gpus=8 --num-cpus=128 \
        --port=$RAY_PORT --object-store-memory=200000000000 \
        --disable-usage-stats 2>&1 | sed 's/^/[ray-head] /'
    for i in $(seq 1 60); do
        n=$(ray status 2>/dev/null | grep -E '^ *[0-9]+ node_' | wc -l)
        echo "[node=$NODE] ray sees $n nodes (waiting for $NUM_NODES)"
        if [ "$n" -ge "$NUM_NODES" ]; then
            echo "[node=$NODE] cluster size $NUM_NODES reached!"
            break
        fi
        sleep 2
    done
    echo "=== ray status ==="
    ray status

    cd $RLER/slime
    DRIVER_LOG=$RUN_DIR/_logs/training.log
    echo "[node=$NODE] launching driver -> $DRIVER_LOG"
    export RAY_ADDRESS=$MY_IP:$RAY_PORT
    python -m train_agent.run.grpo_async_naive \
      --target $GRPO_TARGET \
      --prompt-data $PROMPT_DATA \
      --hf-checkpoint $HF_CHECKPOINT \
      --load-dir $LOAD_DIR \
      --save-dir $SAVE_DIR \
      --rler-root $RLER \
      --config-path $RLER/slime/train_agent/configs/grpo.sh \
      --model-config-name $MODEL_CONFIG_NAME \
      --search-output-root $RUN_DIR/naive_out \
      --student-model "$GRPO_MODEL_NAME" \
      --rollout-batch-size $ROLLOUT_BATCH_SIZE \
      --over-sampling-batch-size $OVER_SAMPLING_BATCH_SIZE \
      --global-batch-size $((ROLLOUT_BATCH_SIZE * NAIVE_M)) \
      --num-rollout $NUM_ROLLOUT \
      --num-nodes $NUM_NODES \
      --num-gpus-per-node $NUM_GPUS_PER_NODE \
      --actor-num-nodes $ACTOR_NUM_NODES \
      --actor-num-gpus $ACTOR_NUM_GPUS \
      --rollout-num-gpus $ROLLOUT_NUM_GPUS \
      --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
      --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
      --decoder-last-pipeline-num-layers $DECODER_LAST_PIPELINE_NUM_LAYERS \
      --context-parallel-size $CONTEXT_PARALLEL_SIZE \
      --rollout-num-gpus-per-engine $ROLLOUT_NUM_GPUS_PER_ENGINE \
      --max-tokens-per-gpu $MAX_TOKENS_PER_GPU \
      --sglang-context-length $SGLANG_CONTEXT_LENGTH \
      --ray-external \
      --naive-instance-workers $NAIVE_INSTANCE_WORKERS \
      --naive-m $NAIVE_M \
      --naive-step-limit $NAIVE_STEP_LIMIT \
      --naive-policy-temperature $NAIVE_POLICY_TEMPERATURE \
      --naive-fallback-patch-penalty $NAIVE_FALLBACK_PATCH_PENALTY \
      --naive-no-action-patch-penalty $NAIVE_NO_ACTION_PATCH_PENALTY \
      --naive-format-error-per-step-penalty $NAIVE_FORMAT_ERROR_PER_STEP_PENALTY \
      --naive-format-ok-gate-threshold $NAIVE_FORMAT_OK_GATE_THRESHOLD \
      --naive-output-root $RUN_DIR/naive_out \
      --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std \
      --save-debug-train-data "$RUN_DIR/_train_dumps/{rollout_id}_{rank}.pt" \
      --save-debug-rollout-data "$RUN_DIR/_rollout_dumps/{rollout_id}.pt" \
      2>&1 | tee $DRIVER_LOG
    DRIVER_RC=${PIPESTATUS[0]}
    echo "[node=$NODE] driver exited rc=$DRIVER_RC"
    touch $DONE_FLAG
    sleep 5
    ray stop --force >/dev/null 2>&1 || true
else
    echo "[node=$NODE] role=WORKER: waiting for head $HEAD_IP:$RAY_PORT"
    for i in $(seq 1 60); do
        if nc -z -w 2 $HEAD_IP $RAY_PORT 2>/dev/null; then
            echo "[node=$NODE] head reachable"
            break
        fi
        sleep 2
    done
    ray start --address=$HEAD_IP:$RAY_PORT --node-ip-address=$MY_IP \
        --num-gpus=8 --num-cpus=128 --object-store-memory=200000000000 \
        --disable-usage-stats
    echo "[node=$NODE] joined ray cluster, waiting for driver done flag"
    while [ ! -f "$DONE_FLAG" ]; do
        sleep 10
    done
    echo "[node=$NODE] driver done, stopping ray"
    ray stop --force >/dev/null 2>&1 || true
fi
EOF
chmod +x $PER_NODE

echo "[$(date)] === srun fan-out across $SLURM_JOB_NUM_NODES nodes ==="
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
    --container-image=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh \
    --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,/var/run/docker.sock:/var/run/docker.sock,/usr/bin/docker:/usr/bin/docker,/mnt/localssd:/mnt/localssd,/home/sihanzeng_meta_com:/home/sihanzeng_meta_com \
    --container-workdir=$RLER/slime \
    bash $PER_NODE \
        "$HEAD_NODE" "$HEAD_IP" "$RAY_PORT" \
        "$CACHE_BASE" "$RUN_DIR" "$RLER" "$BASE" \
        "$PROMPT_DATA" "$HF_CHECKPOINT" "$LOAD_DIR" "$SAVE_DIR"

echo "[$(date)] DONE jobid=$SLURM_JOB_ID"
