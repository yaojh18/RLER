#!/bin/bash
#SBATCH --job-name=grpo-lanes-cp4
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/grpo-lanes-cp4-%j.log

# 4-node GRPO async LANES (v1, target=policy). 4 × 8 = 32 GPUs:
#   16 actor GPUs (4 nodes × 4 actor GPUs)
#   16 rollout GPUs (4 nodes × 4 sglang GPUs)
# Multi-node ray cluster is brought up via srun before the driver runs.

set -u

# ====================================================================
# Single source of truth for all tunable knobs. Both the env exports
# (which Ray workers inherit at fork time) AND the driver CLI flags
# below reference these — change a value here and both stay in sync.
# ====================================================================
# LANES / search (v1)
LANES_M=8                       # Lane B forks per mid_cp
LANES_MAX_MID_CPS=6             # cap on Lane A mid_cp emissions per instance
LANES_STEPS_PER_ROUND=20        # asst turns between mid_cps
LANES_STEP_LIMIT=120            # hard cap per Lane B branch
LANES_INSTANCE_WORKERS=24       # concurrent instance subprocesses across 4 nodes
LANES_GT_EVAL_WORKERS=8         # per-instance GT eval threads
LANES_COMPLETION_MAX_TOKENS=4096
LANES_POLICY_ALPHA=1.0          # DEPRECATED — reward is now pure rubric in
                                # lane_to_grpo_bundle._build_branch_sample;
                                # this knob is ignored. Kept for sbatch back-compat.
LANES_POLICY_TEMPERATURE=0.5    # Lane A (spine) sampling — match v0 51639 known-good
LANES_POLICY_TOP_P=0.95
LANES_LANE_B_TEMPERATURE=0.5    # Lane B (forks) sampling — temp=1.0 caused NaN
                                #   (51783 step 5, 51840 step 1). Drop to 0.5.
LANES_LANE_B_TOP_P=0.95
# Same fallback-patch penalty recipe as v0.
LANES_FALLBACK_PATCH_PENALTY=0.5
# GRPO / training
# rollout_batch_size = #LANES-groups returned per training step.
# n_samples_per_prompt MUST equal LANES_M (set in slime/train_agent/configs/grpo.sh)
# so slime's rewards.reshape(-1, n_samples_per_prompt) groups by sibling for the
# GRPO baseline. With ROLLOUT_BATCH_SIZE=16 + n_samples_per_prompt=8 (=LANES_M):
#   global_batch_size = 16 * 8 = 128 samples per step.
# OVER_SAMPLING_BATCH_SIZE: must be >= ROLLOUT_BATCH_SIZE. Mirror into
# SWE_AGENT_LANES_MAX_PENDING so the runner keeps that many tasks concurrent.
# LANES_INSTANCE_WORKERS=24 → 6 per node × 4 nodes. Each instance produces up
# to LANES_MAX_MID_CPS × LANES_M = 6 × 8 = 48 Lane B agents, but they are
# serialized per-instance through lane_b_pool inside the runner.
ROLLOUT_BATCH_SIZE=16
OVER_SAMPLING_BATCH_SIZE=28
NUM_ROLLOUT=60
NUM_NODES=4
NUM_GPUS_PER_NODE=8
# Asymmetric 2-node: actor on node 0 (GPUs 0-3), rollout uses GPUs 4-15
# across both nodes (4 on node 0 + 8 on node 1). Requires the
# physical-IP grouping fix in slime/ray/rollout.py:_allocate_rollout_engine_addr_and_ports_normal.
ACTOR_NUM_NODES=1
ACTOR_NUM_GPUS=4
ROLLOUT_NUM_GPUS=28
# Rollout target
GRPO_TARGET=policy
GRPO_MODEL_NAME="Qwen/Qwen3.5-9B"
# Number of unique instance prompts to feed PDS (>= ROLLOUT_BATCH_SIZE)
PROMPT_BUDGET=400
# Context parallel — split each sample sequence across CP GPUs.
# CP=4 with 4 actor GPUs means dp=1 + per-GPU activation memory is 1/4 of
# what CP=2 would use, fixing OOM on long trajectories.
CONTEXT_PARALLEL_SIZE=4
# ====================================================================

NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
HEAD_NODE=$(echo "$NODES" | head -n 1)
HEAD_IP=$(getent ahostsv4 $HEAD_NODE | awk '{print $1}' | head -n 1)
[ -z "$HEAD_IP" ] && HEAD_IP=$(srun --nodes=1 --ntasks=1 -w $HEAD_NODE hostname -I | awk '{print $1}')
RAY_PORT=6379

RLER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/RLER
BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
SUBSET=$BASE/datasets/v2_python1k_split_train
RUN_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/${SLURM_JOB_ID}-grpo-lanes-cp4
SLURM_LOGS=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs
mkdir -p $RUN_DIR $SLURM_LOGS $RUN_DIR/_logs

HF_CHECKPOINT=/mnt/lustre/metavmds0lstre/teams/mrs_ranking_core_modeling/daili1/deepeyes/models/Qwen3.5-9B
# Resume from 51483's iter_0000009 (latest after the OOM-fix run reached 10 steps).
# To cold-start, swap back to /mnt/lustre/.../slime_qwen35/ckpt/Qwen3.5-9B_torch_dist_latest
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

# Cache base on Lustre
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
# Bake the single-source-of-truth values into per_node.sh as constants so
# the script run by srun on each node sees them. Use unquoted heredoc here
# (interpolation at SLURM-script time), then APPEND the body with quoted
# heredoc to keep $NODE / $MY_IP / etc. unevaluated until per_node runs.
cat > $PER_NODE <<TOP
#!/bin/bash
set -u
# === single-source-of-truth values from SLURM script ===
GRPO_TARGET="$GRPO_TARGET"
GRPO_MODEL_NAME="$GRPO_MODEL_NAME"
LANES_M=$LANES_M
LANES_MAX_MID_CPS=$LANES_MAX_MID_CPS
LANES_STEPS_PER_ROUND=$LANES_STEPS_PER_ROUND
LANES_STEP_LIMIT=$LANES_STEP_LIMIT
LANES_INSTANCE_WORKERS=$LANES_INSTANCE_WORKERS
LANES_GT_EVAL_WORKERS=$LANES_GT_EVAL_WORKERS
LANES_COMPLETION_MAX_TOKENS=$LANES_COMPLETION_MAX_TOKENS
LANES_POLICY_ALPHA=$LANES_POLICY_ALPHA
LANES_POLICY_TEMPERATURE=$LANES_POLICY_TEMPERATURE
LANES_POLICY_TOP_P=$LANES_POLICY_TOP_P
LANES_LANE_B_TEMPERATURE=$LANES_LANE_B_TEMPERATURE
LANES_LANE_B_TOP_P=$LANES_LANE_B_TOP_P
LANES_FALLBACK_PATCH_PENALTY=$LANES_FALLBACK_PATCH_PENALTY
ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE
OVER_SAMPLING_BATCH_SIZE=$OVER_SAMPLING_BATCH_SIZE
NUM_ROLLOUT=$NUM_ROLLOUT
NUM_NODES=$NUM_NODES
NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE
ACTOR_NUM_NODES=$ACTOR_NUM_NODES
ACTOR_NUM_GPUS=$ACTOR_NUM_GPUS
ROLLOUT_NUM_GPUS=$ROLLOUT_NUM_GPUS
CONTEXT_PARALLEL_SIZE=$CONTEXT_PARALLEL_SIZE
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

# Per-node cache subdir (writable inside container).
NODE=$(hostname)
NODE_CACHE=$CACHE_BASE/$NODE
mkdir -p $NODE_CACHE/{flashinfer,triton,hf,torch}
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

# Wandb online sync — load the API key from a Lustre-mounted secrets
# file (NOT committed to git). The per-node container overrides HOME
# to a fresh cache dir so ~/.netrc on the login node is invisible
# inside the container; load directly into WANDB_API_KEY env instead.
# Provisioned by:
#   mkdir -p /mnt/lustre/.../secrets && chmod 700 ...
#   grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}' > .../wandb_api_key
WANDB_KEY_FILE=/mnt/lustre/metavmds0lstre/share/sihanzeng_meta_com/secrets/wandb_api_key
if [ -r "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY=$(cat "$WANDB_KEY_FILE" | tr -d '[:space:]')
    export WANDB_MODE=online
    export WANDB_ENTITY=chenqinghan2003-university-of-washington
else
    echo "[node=$NODE] WARN: wandb key file missing at $WANDB_KEY_FILE, staying offline"
    export WANDB_MODE=offline
fi

# Ray worker processes inherit PYTHONPATH and env from the per-node shell.
# They need RLER/slime (for slime imports) and /root/Megatron-LM (for
# megatron.training) in addition to our agent paths — these get added to
# grpo.py's *driver* env but the Ray actors are forked from the ray-start
# parent process which sees only this PYTHONPATH.
export PYTHONPATH=$BASE/SWE-rebench-V2:$RLER:$RLER/agent:$RLER/slime:/root/Megatron-LM:${PYTHONPATH:-}
export EXTRA_PYTHONPATH=$BASE/SWE-rebench-V2:$RLER:$RLER/agent

# LANES / agent env vars — must match the CLI args below since
# grpo_async_lanes.py only exports them in its own process (driver), not
# in the Ray worker procs that were forked from `ray start` earlier.
export SWE_AGENT_LANES_OUTPUT_ROOT=$RUN_DIR/lane_out
export SWE_AGENT_GRPO_OUTPUT_ROOT=$RUN_DIR/lane_out
export SWE_AGENT_GRPO_TARGET=$GRPO_TARGET
export SWE_AGENT_GRPO_MODEL_NAME=$GRPO_MODEL_NAME
# CRITICAL: Ray workers fork from this shell BEFORE grpo_async_lanes.py
# runs, so they don't see env vars exported later by the driver. Export
# all LANES knobs here — values come from the single-source-of-truth
# block at the top of the script.
export SWE_AGENT_LANES_M=$LANES_M
export SWE_AGENT_LANES_MAX_MID_CPS=$LANES_MAX_MID_CPS
export SWE_AGENT_LANES_STEPS_PER_ROUND=$LANES_STEPS_PER_ROUND
export SWE_AGENT_LANES_STEP_LIMIT=$LANES_STEP_LIMIT
export SWE_AGENT_LANES_INSTANCE_WORKERS=$LANES_INSTANCE_WORKERS
# Long-tail buffer: keep this many instances in flight even though the
# trainer only needs ROLLOUT_BATCH_SIZE groups per step. Mirror of the
# slime --over-sampling-batch-size CLI arg (which governs slime's
# data-source pull; lanes scheduling lives here).
export SWE_AGENT_LANES_MAX_PENDING=$OVER_SAMPLING_BATCH_SIZE
export SWE_AGENT_LANES_GT_EVAL_WORKERS=$LANES_GT_EVAL_WORKERS
export SWE_AGENT_LANES_COMPLETION_MAX_TOKENS=$LANES_COMPLETION_MAX_TOKENS
export SWE_AGENT_LANES_POLICY_ALPHA=$LANES_POLICY_ALPHA
export SWE_AGENT_LANES_POLICY_TEMPERATURE=$LANES_POLICY_TEMPERATURE
export SWE_AGENT_LANES_POLICY_TOP_P=$LANES_POLICY_TOP_P
export SWE_AGENT_LANES_LANE_B_TEMPERATURE=$LANES_LANE_B_TEMPERATURE
export SWE_AGENT_LANES_LANE_B_TOP_P=$LANES_LANE_B_TOP_P
export SWE_AGENT_LANES_FALLBACK_PATCH_PENALTY=$LANES_FALLBACK_PATCH_PENALTY

MY_IP=$(hostname -I | awk '{print $1}')
echo "[node=$NODE] my_ip=$MY_IP head=$HEAD_NODE head_ip=$HEAD_IP"

# === Per-node memory monitor (sidecar) ===
# Logs free -g + top RSS processes every 30s to a per-node file. Lets us
# diagnose rollout-side OOM separately from train-side Megatron logging
# (which only covers the actor node). Run in background, killed on script exit.
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

# === Defensive sglang patch: make usage_processor tolerate missing prompt_tokens ===
# Upstream bug: srt/entrypoints/openai/usage_processor.py:26 raises KeyError on
# degenerate responses missing meta_info.prompt_tokens. Switch to .get() so a
# single bad response doesn't 500 and doesn't poison the agent's litellm retry
# loop. Patch is idempotent.
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
    # Wait for all NUM_NODES nodes to join
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

    # Now run the training driver
    cd $RLER/slime
    DRIVER_LOG=$RUN_DIR/_logs/training.log
    echo "[node=$NODE] launching driver -> $DRIVER_LOG"
    # Tell train_async.py to attach to our existing multi-node Ray cluster
    # instead of spinning up a local one.
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
      --lanes-policy-alpha $LANES_POLICY_ALPHA \
      --lanes-policy-temperature $LANES_POLICY_TEMPERATURE \
      --lanes-lane-b-temperature $LANES_LANE_B_TEMPERATURE \
      --lanes-fallback-patch-penalty $LANES_FALLBACK_PATCH_PENALTY \
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
    --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,/var/run/docker.sock:/var/run/docker.sock,/usr/bin/docker:/usr/bin/docker \
    --container-workdir=$RLER/slime \
    bash $PER_NODE \
        "$HEAD_NODE" "$HEAD_IP" "$RAY_PORT" \
        "$CACHE_BASE" "$RUN_DIR" "$RLER" "$BASE" \
        "$PROMPT_DATA" "$HF_CHECKPOINT" "$LOAD_DIR" "$SAVE_DIR"

echo "[$(date)] DONE jobid=$SLURM_JOB_ID"
