#!/bin/bash
#SBATCH --job-name=grpo-lanes-gts
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/grpo-lanes-gts-%j.log

# 4-node GRPO async LANES with GT-only training (rubric+judge SKIPPED).
# SOFT variant of grpo_lanes_gtjoint_4node_cp4.slurm — same plumbing, but
# uses RLER_REWARD_SCHEME=soft (legacy) instead of joint.
#
# Motivation: joint reward (frac_F2P * frac_P2P) produces high intra-group
# variance — branches that pass everything score 1.0, branches that miss
# anything score 0.0. Result on job 56245: two grad-norm spikes (4.2, 6.4)
# in 25 steps + gt_mean drifting DOWN, not up. Soft reward gives partial
# credit and smaller advantages -> smaller gradient steps.
#
# Requires: RLER clone has both
#   (a) disable_rubric / gt_only_reward plumbing (this branch)
#   (b) RLER_REWARD_SCHEME plumbing from naive r5 commit b89f417

set -u

# ====================================================================
# Single source of truth for all tunable knobs.
# ====================================================================
LANES_M=8
LANES_MAX_MID_CPS=6
LANES_STEPS_PER_ROUND=20
LANES_STEP_LIMIT=120
LANES_INSTANCE_WORKERS=24
LANES_GT_EVAL_WORKERS=8
LANES_COMPLETION_MAX_TOKENS=4096
LANES_POLICY_TEMPERATURE=0.5
LANES_POLICY_TOP_P=0.95
LANES_LANE_B_TEMPERATURE=0.5
LANES_LANE_B_TOP_P=0.95
LANES_FALLBACK_PATCH_PENALTY=0.5
LANES_DISABLE_RUBRIC=1
ROLLOUT_BATCH_SIZE=16
OVER_SAMPLING_BATCH_SIZE=28
NUM_ROLLOUT=60
NUM_NODES=4
NUM_GPUS_PER_NODE=8
ACTOR_NUM_NODES=1
ACTOR_NUM_GPUS=4
ROLLOUT_NUM_GPUS=28
GRPO_TARGET=policy
GRPO_MODEL_NAME="Qwen/Qwen3.5-9B"
PROMPT_BUDGET=1316
CONTEXT_PARALLEL_SIZE=4
# Soft (legacy) reward: weighted average of F2P/P2P pass rates rather than
# their product. Smoother signal, smaller per-group advantage variance.
REWARD_SCHEME=soft
# ====================================================================

NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
HEAD_NODE=$(echo "$NODES" | head -n 1)
HEAD_IP=$(getent ahostsv4 $HEAD_NODE | awk '{print $1}' | head -n 1)
[ -z "$HEAD_IP" ] && HEAD_IP=$(srun --nodes=1 --ntasks=1 -w $HEAD_NODE hostname -I | awk '{print $1}')
RAY_PORT=6379

RLER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/RLER-zsh-trim
BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
SUBSET=$BASE/datasets/v2_python_train_plus_sweep_middle
RUN_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/${SLURM_JOB_ID}-grpo-lanes-gtsoft
SLURM_LOGS=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs
mkdir -p $RUN_DIR $SLURM_LOGS $RUN_DIR/_logs

HF_CHECKPOINT=/mnt/lustre/metavmds0lstre/teams/mrs_ranking_core_modeling/daili1/deepeyes/models/Qwen3.5-9B
LOAD_DIR=/mnt/lustre/metavmds0lstre/share/sihanzeng_meta_com/slime_qwen35/ckpt/Qwen3.5-9B_torch_dist_latest
SAVE_DIR=$RUN_DIR/policy_ckpt
PROMPT_DATA=$RUN_DIR/grpo_prompts.jsonl

echo "[$(date)] jobid=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES head=$HEAD_NODE head_ip=$HEAD_IP"
echo "Node list:"
echo "$NODES"

[ ! -d "$LOAD_DIR" ] && { echo "ERROR: LOAD_DIR=$LOAD_DIR does not exist"; exit 1; }

mkdir -p $RUN_DIR/_snapshot
( cd $RLER && git rev-parse HEAD ) > $RUN_DIR/_snapshot/RLER_HEAD.txt
cp $0 $RUN_DIR/_snapshot/ 2>/dev/null || true

CACHE_BASE=/mnt/localssd/sihanzeng-grpo-cache-${SLURM_JOB_ID}
mkdir -p $CACHE_BASE
chmod -R 777 $CACHE_BASE 2>/dev/null || true

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

PER_NODE=$RUN_DIR/_snapshot/per_node.sh
cat > $PER_NODE <<TOP
#!/bin/bash
set -u
GRPO_TARGET="$GRPO_TARGET"
GRPO_MODEL_NAME="$GRPO_MODEL_NAME"
LANES_M=$LANES_M
LANES_MAX_MID_CPS=$LANES_MAX_MID_CPS
LANES_STEPS_PER_ROUND=$LANES_STEPS_PER_ROUND
LANES_STEP_LIMIT=$LANES_STEP_LIMIT
LANES_INSTANCE_WORKERS=$LANES_INSTANCE_WORKERS
LANES_GT_EVAL_WORKERS=$LANES_GT_EVAL_WORKERS
LANES_COMPLETION_MAX_TOKENS=$LANES_COMPLETION_MAX_TOKENS
LANES_POLICY_TEMPERATURE=$LANES_POLICY_TEMPERATURE
LANES_POLICY_TOP_P=$LANES_POLICY_TOP_P
LANES_LANE_B_TEMPERATURE=$LANES_LANE_B_TEMPERATURE
LANES_LANE_B_TOP_P=$LANES_LANE_B_TOP_P
LANES_FALLBACK_PATCH_PENALTY=$LANES_FALLBACK_PATCH_PENALTY
LANES_DISABLE_RUBRIC=$LANES_DISABLE_RUBRIC
ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE
OVER_SAMPLING_BATCH_SIZE=$OVER_SAMPLING_BATCH_SIZE
NUM_ROLLOUT=$NUM_ROLLOUT
NUM_NODES=$NUM_NODES
NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE
ACTOR_NUM_NODES=$ACTOR_NUM_NODES
ACTOR_NUM_GPUS=$ACTOR_NUM_GPUS
ROLLOUT_NUM_GPUS=$ROLLOUT_NUM_GPUS
CONTEXT_PARALLEL_SIZE=$CONTEXT_PARALLEL_SIZE
REWARD_SCHEME=$REWARD_SCHEME
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
NODE_CACHE=$CACHE_BASE/$NODE
mkdir -p $NODE_CACHE/{flashinfer,triton,hf,torch}
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
export MALLOC_ARENA_MAX=2

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

export SWE_AGENT_LANES_OUTPUT_ROOT=$RUN_DIR/lane_out
export SWE_AGENT_GRPO_OUTPUT_ROOT=$RUN_DIR/lane_out
export SWE_AGENT_GRPO_TARGET=$GRPO_TARGET
export SWE_AGENT_GRPO_MODEL_NAME=$GRPO_MODEL_NAME
export SWE_AGENT_LANES_M=$LANES_M
export SWE_AGENT_LANES_MAX_MID_CPS=$LANES_MAX_MID_CPS
export SWE_AGENT_LANES_STEPS_PER_ROUND=$LANES_STEPS_PER_ROUND
export SWE_AGENT_LANES_STEP_LIMIT=$LANES_STEP_LIMIT
export SWE_AGENT_LANES_INSTANCE_WORKERS=$LANES_INSTANCE_WORKERS
export SWE_AGENT_LANES_MAX_PENDING=$OVER_SAMPLING_BATCH_SIZE
export SWE_AGENT_LANES_GT_EVAL_WORKERS=$LANES_GT_EVAL_WORKERS
export SWE_AGENT_LANES_COMPLETION_MAX_TOKENS=$LANES_COMPLETION_MAX_TOKENS
export SWE_AGENT_LANES_POLICY_TEMPERATURE=$LANES_POLICY_TEMPERATURE
export SWE_AGENT_LANES_POLICY_TOP_P=$LANES_POLICY_TOP_P
export SWE_AGENT_LANES_LANE_B_TEMPERATURE=$LANES_LANE_B_TEMPERATURE
export SWE_AGENT_LANES_LANE_B_TOP_P=$LANES_LANE_B_TOP_P
export SWE_AGENT_LANES_FALLBACK_PATCH_PENALTY=$LANES_FALLBACK_PATCH_PENALTY
export SWE_AGENT_LANES_DISABLE_RUBRIC=$LANES_DISABLE_RUBRIC
export RLER_REWARD_SCHEME=$REWARD_SCHEME

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
        --port=$RAY_PORT --disable-usage-stats 2>&1 | sed 's/^/[ray-head] /'
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
    python -m train_agent.run.grpo_async_lanes \
      --target $GRPO_TARGET \
      --prompt-data $PROMPT_DATA \
      --hf-checkpoint $HF_CHECKPOINT \
      --load-dir $LOAD_DIR \
      --save-dir $SAVE_DIR \
      --rler-root $RLER \
      --config-path $RLER/slime/train_agent/configs/grpo.sh \
      --search-output-root $RUN_DIR/lane_out \
      --student-model "$GRPO_MODEL_NAME" \
      --rollout-batch-size $ROLLOUT_BATCH_SIZE \
      --over-sampling-batch-size $OVER_SAMPLING_BATCH_SIZE \
      --global-batch-size $((ROLLOUT_BATCH_SIZE * LANES_M)) \
      --num-rollout $NUM_ROLLOUT \
      --num-nodes $NUM_NODES \
      --num-gpus-per-node $NUM_GPUS_PER_NODE \
      --actor-num-nodes $ACTOR_NUM_NODES \
      --actor-num-gpus $ACTOR_NUM_GPUS \
      --rollout-num-gpus $ROLLOUT_NUM_GPUS \
      --context-parallel-size $CONTEXT_PARALLEL_SIZE \
      --ray-external \
      --lanes-instance-workers $LANES_INSTANCE_WORKERS \
      --lanes-m $LANES_M \
      --lanes-max-mid-cps $LANES_MAX_MID_CPS \
      --lanes-steps-per-round $LANES_STEPS_PER_ROUND \
      --lanes-step-limit $LANES_STEP_LIMIT \
      --lanes-policy-temperature $LANES_POLICY_TEMPERATURE \
      --lanes-lane-b-temperature $LANES_LANE_B_TEMPERATURE \
      --lanes-fallback-patch-penalty $LANES_FALLBACK_PATCH_PENALTY \
      --lanes-disable-rubric \
      --lanes-output-root $RUN_DIR/lane_out \
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
        --num-gpus=8 --num-cpus=128 --disable-usage-stats
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
    --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,/mnt/localssd:/mnt/localssd,/var/run/docker.sock:/var/run/docker.sock,/usr/bin/docker:/usr/bin/docker \
    --container-workdir=$RLER/slime \
    bash $PER_NODE \
        "$HEAD_NODE" "$HEAD_IP" "$RAY_PORT" \
        "$CACHE_BASE" "$RUN_DIR" "$RLER" "$BASE" \
        "$PROMPT_DATA" "$HF_CHECKPOINT" "$LOAD_DIR" "$SAVE_DIR"

echo "[$(date)] DONE jobid=$SLURM_JOB_ID"
