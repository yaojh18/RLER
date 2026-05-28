#!/bin/bash
#SBATCH --job-name=grpo-naive-soft-fmtfix
#SBATCH --account=ads_cpt
#SBATCH --qos=ads_cpt
#SBATCH --partition=a4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/home/sihanzeng_meta_com/rler_runs/_slurm-logs/grpo-naive-soft-fmtfix-%j.log

# 4-node GRPO async NAIVE — V7b (soft + format-error penalties).
# Same as V6b except: format-error per-step penalty (k=0.02) AND format-ok
# hard gate (threshold=0.5) both ON. Mitigation for the 58062 (v6b) mode
# collapse — soft reward made things worse (giving credit for
# baseline-preserving rollouts let degenerate FE trajectories with
# fallback patches pull positive gradient). With both penalties on, an
# FE-spam trajectory is hard-zeroed regardless of patch.

set -u

# ====================================================================
# Single source of truth for all tunable knobs. Both the env exports
# (which Ray workers inherit at fork time) AND the driver CLI flags
# below reference these — change a value here and both stay in sync.
# ====================================================================
# NAIVE / search
NAIVE_M=8                       # rollouts per instance (= n_samples_per_prompt)
NAIVE_STEP_LIMIT=120            # reverted from 40 (was cutting off 56% of would-be submitters; v1 baseline)
# Round 2: actor is on its own node (8 GPUs), instance subprocesses only run
# on the 3 rollout nodes. 18 instance workers / 3 rollout nodes = 6 per node.
# This matches round-1 per-node concurrency but excludes the actor node where
# round-1 OOMed at step 23 due to actor + 6 instance subprocs on same node.
NAIVE_INSTANCE_WORKERS=18       # concurrent instance subprocesses across 3 rollout nodes
NAIVE_GT_EVAL_WORKERS=8         # per-instance GT eval threads
NAIVE_COMPLETION_MAX_TOKENS=4096
NAIVE_POLICY_TEMPERATURE=1.0    # naive baseline — single sampling temp
NAIVE_POLICY_TOP_P=0.95
NAIVE_FALLBACK_PATCH_PENALTY=0.5
NAIVE_NO_ACTION_PATCH_PENALTY=0.0  # v2: hard-zero reward for rollouts with zero env actions
NAIVE_REWARD_KIND=soft             # v6b: scored = raw_reward (no baseline subtraction)
# v7b: format-error mitigation (job 58062 mode-collapse fix).
NAIVE_FORMAT_ERROR_PER_STEP_PENALTY=0.02  # linear decay; zeros reward at 50 FE turns
NAIVE_FORMAT_OK_GATE_THRESHOLD=0.5        # hard zero when >50% of asst turns are FormatError
# GRPO / training
# rollout_batch_size = #naive-groups returned per training step.
# n_samples_per_prompt MUST equal NAIVE_M (set in slime/train_agent/configs/grpo.sh)
# so slime's rewards.reshape(-1, n_samples_per_prompt) groups by sibling for the
# GRPO baseline. With ROLLOUT_BATCH_SIZE=16 + n_samples_per_prompt=8 (=NAIVE_M):
#   global_batch_size = 16 * 8 = 128 samples per step.
ROLLOUT_BATCH_SIZE=16
OVER_SAMPLING_BATCH_SIZE=32
NUM_ROLLOUT=60
NUM_NODES=4
NUM_GPUS_PER_NODE=8
# Pure-actor / pure-rollout split: node 0 runs the actor on ALL 8 GPUs
# (DP=2, CP=4 — actor_num_gpus / cp_size = 8 / 4 = 2 data-parallel groups).
# Nodes 1-3 are pure-rollout (3 nodes x 8 sglang engines = 24 rollout GPUs).
# Round-1 round had a co-resident actor + 4 sglang + 6 instance subprocs on
# node 0 and OOMed at step 23 — splitting cleanly fixes that.
ACTOR_NUM_NODES=1
ACTOR_NUM_GPUS=8
ROLLOUT_NUM_GPUS=24
GRPO_TARGET=policy
GRPO_MODEL_NAME="Qwen/Qwen3.5-9B"
PROMPT_BUDGET=1150          # v2 dataset: 1316 - 166 (registry-deleted images)
CONTEXT_PARALLEL_SIZE=4
# ====================================================================

NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
HEAD_NODE=$(echo "$NODES" | head -n 1)
HEAD_IP=$(getent ahostsv4 $HEAD_NODE | awk '{print $1}' | head -n 1)
[ -z "$HEAD_IP" ] && HEAD_IP=$(srun --nodes=1 --ntasks=1 -w $HEAD_NODE hostname -I | awk '{print $1}')
RAY_PORT=6379

RLER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/RLER
BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
SUBSET=$BASE/datasets/v2_python_train_plus_sweep_middle_v2
RUN_DIR=/home/sihanzeng_meta_com/rler_runs/${SLURM_JOB_ID}-grpo-naive-soft-fmtfix
SLURM_LOGS=/home/sihanzeng_meta_com/rler_runs/_slurm-logs
mkdir -p $RUN_DIR $SLURM_LOGS $RUN_DIR/_logs

HF_CHECKPOINT=/mnt/lustre/metavmds0lstre/teams/mrs_ranking_core_modeling/daili1/deepeyes/models/Qwen3.5-9B
LOAD_DIR=/mnt/lustre/metavmds0lstre/share/sihanzeng_meta_com/slime_qwen35/ckpt/Qwen3.5-9B_torch_dist_latest
SAVE_DIR=/mnt/localssd/rler-runs/${SLURM_JOB_ID}-grpo-naive-soft-fmtfix/policy_ckpt
PROMPT_DATA=$RUN_DIR/grpo_prompts.jsonl

echo "[$(date)] jobid=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES head=$HEAD_NODE head_ip=$HEAD_IP"
echo "Node list:"
echo "$NODES"

[ ! -d "$LOAD_DIR" ] && { echo "ERROR: LOAD_DIR=$LOAD_DIR does not exist"; exit 1; }

mkdir -p $RUN_DIR/_snapshot
( cd $RLER && git rev-parse HEAD ) > $RUN_DIR/_snapshot/RLER_HEAD.txt
cp $0 $RUN_DIR/_snapshot/ 2>/dev/null || true

CACHE_BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/.cache/grpo-${SLURM_JOB_ID}
mkdir -p $CACHE_BASE
chmod -R 777 $CACHE_BASE 2>/dev/null || true

# Generate prompt data on the head (NFS-shared via Lustre)
python3 - <<EOF
import json, os
from datasets import load_from_disk
candidate = "$SUBSET/train" if os.path.isdir("$SUBSET/train") else "$SUBSET"
ds = load_from_disk(candidate)
if hasattr(ds, "keys") and "train" in ds.keys():
    ds = ds["train"]
ids = sorted([r["instance_id"] for r in ds])[:$PROMPT_BUDGET]
with open("$PROMPT_DATA", "w") as f:
    for iid in ids:
        f.write(json.dumps({
            "input": [{"role": "user", "content": iid}],
            "metadata": {"instance_id": iid, "subset": "$SUBSET", "split": "train"},
        }) + "\n")
print("Wrote", len(ids), "prompt rows to $PROMPT_DATA")
EOF

# ----- Per-node setup script (run inside container on each node) -----
PER_NODE=$RUN_DIR/_snapshot/per_node.sh
cat > $PER_NODE <<TOP
#!/bin/bash
set -u
# === single-source-of-truth values from SLURM script ===
GRPO_TARGET="$GRPO_TARGET"
GRPO_MODEL_NAME="$GRPO_MODEL_NAME"
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
CONTEXT_PARALLEL_SIZE=$CONTEXT_PARALLEL_SIZE
# Propagate SLURM job identity so grpo.py can default the wandb run name
# to "\$SLURM_JOB_NAME-\$SLURM_JOB_ID" instead of the generic "policy-grpo".
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

# Lustre docker shim env
export MSWEA_DOCKER_EXECUTABLE=$BASE/bin/docker
export DOCKER_LUSTRE_TARDIR=$BASE/docker_tarballs
export DOCKER_LUSTRE_REAL=/usr/bin/docker
export MSWEA_COST_TRACKING=ignore_errors
export SEARCH_SWE_SLIME_API_KEY=EMPTY
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1

# Wandb online sync — same recipe as the lanes launcher.
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

# NAIVE / agent env vars — Ray workers fork from this shell BEFORE the
# driver runs, so they don't see env vars exported later by the driver.
# Export all NAIVE knobs here.
export SWE_AGENT_NAIVE_OUTPUT_ROOT=$RUN_DIR/naive_out
export SWE_AGENT_GRPO_OUTPUT_ROOT=$RUN_DIR/naive_out
export SWE_AGENT_GRPO_TARGET=$GRPO_TARGET
export SWE_AGENT_GRPO_MODEL_NAME=$GRPO_MODEL_NAME
export SWE_AGENT_NAIVE_M=$NAIVE_M
export SWE_AGENT_NAIVE_STEP_LIMIT=$NAIVE_STEP_LIMIT
export SWE_AGENT_NAIVE_INSTANCE_WORKERS=$NAIVE_INSTANCE_WORKERS
export SWE_AGENT_NAIVE_MAX_PENDING=$OVER_SAMPLING_BATCH_SIZE
# Tell the rollout fn to NOT spawn a NodeWorker on the actor node (head).
# Without this, the head node would get an instance-subprocess pool on
# top of Megatron + Ray driver — exactly the round-1 OOM vector.
export SWE_AGENT_NAIVE_SKIP_NODE_IPS=$HEAD_IP
export SWE_AGENT_NAIVE_GT_EVAL_WORKERS=$NAIVE_GT_EVAL_WORKERS
export SWE_AGENT_NAIVE_COMPLETION_MAX_TOKENS=$NAIVE_COMPLETION_MAX_TOKENS
export SWE_AGENT_NAIVE_POLICY_TEMPERATURE=$NAIVE_POLICY_TEMPERATURE
export SWE_AGENT_NAIVE_POLICY_TOP_P=$NAIVE_POLICY_TOP_P
export SWE_AGENT_NAIVE_FALLBACK_PATCH_PENALTY=$NAIVE_FALLBACK_PATCH_PENALTY
export SWE_AGENT_NAIVE_NO_ACTION_PATCH_PENALTY=$NAIVE_NO_ACTION_PATCH_PENALTY
export SWE_AGENT_NAIVE_REWARD_KIND=$NAIVE_REWARD_KIND
export SWE_AGENT_NAIVE_FORMAT_ERROR_PER_STEP_PENALTY=$NAIVE_FORMAT_ERROR_PER_STEP_PENALTY
export SWE_AGENT_NAIVE_FORMAT_OK_GATE_THRESHOLD=$NAIVE_FORMAT_OK_GATE_THRESHOLD

MY_IP=$(hostname -I | awk '{print $1}')
echo "[node=$NODE] my_ip=$MY_IP head=$HEAD_NODE head_ip=$HEAD_IP"

# === Per-node memory monitor (sidecar) ===
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

# === Defensive sglang patch: tolerate missing prompt_tokens (same as lanes) ===
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
      --context-parallel-size $CONTEXT_PARALLEL_SIZE \
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
