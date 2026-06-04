#!/bin/bash
#SBATCH --job-name=slime-fa-q3-4b-smoke
#SBATCH --account=ads_cpt
#SBATCH --qos=ads_cpt
#SBATCH --partition=a4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/slime-fa-q3-4b-smoke-%j.log

# Smoke run for slime's `fully_async` rollout example with Qwen3-4B-Instruct-2507.
# Single-node, 8-GPU (actor=4 + rollout=4). Verifies that the slime stack
# trains end-to-end and that GRPO reward trends up on DAPO-Math-17k.
#
# Mirrors slime/examples/fully_async/run-qwen3-4b-fully_async.sh but:
#   - uses Qwen3-4B-Instruct-2507 (HF rope=5M) so we don't need to download Qwen3-4B
#   - lustre paths instead of /root/...
#   - per-job /mnt/localssd cache dirs (lustre OST exhaustion mitigation)
#   - NUM_ROLLOUT=30 (smoke), eval visible via reward in training.log

set -u

# --- Tunables ---
NUM_ROLLOUT=30
ROLLOUT_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
GLOBAL_BATCH_SIZE=256
SAVE_INTERVAL=10
# ----------------

BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng
RLER=$BASE/swe-rebench/RLER
HF_CHECKPOINT=$BASE/models/Qwen3-4B-Instruct-2507
TORCH_DIST=$BASE/models/Qwen3-4B-Instruct-2507_torch_dist
PROMPT_DATA=$BASE/datasets/dapo-math-17k/dapo-math-17k.jsonl
RUN_DIR=$BASE/rler-runs/${SLURM_JOB_ID}-slime-fa-q3-4b-smoke
SLURM_LOGS=$BASE/rler-runs/_slurm-logs
SAVE_DIR=$RUN_DIR/policy_ckpt
LOAD_DIR=$SAVE_DIR
mkdir -p $RUN_DIR/_logs $RUN_DIR/_snapshot $SLURM_LOGS $SAVE_DIR

# Snapshot for reproducibility
( cd $RLER && git rev-parse HEAD ) > $RUN_DIR/_snapshot/RLER_HEAD.txt 2>/dev/null || true
cp $0 $RUN_DIR/_snapshot/ 2>/dev/null || true

[ ! -d "$HF_CHECKPOINT" ] && { echo "ERROR: HF_CHECKPOINT=$HF_CHECKPOINT missing"; exit 1; }
[ ! -f "$PROMPT_DATA" ]   && { echo "ERROR: PROMPT_DATA=$PROMPT_DATA missing";   exit 1; }

CONTAINER_IMAGE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh

echo "[$(date)] jobid=$SLURM_JOB_ID host=$(hostname)"
echo "[$(date)] container=$CONTAINER_IMAGE"
echo "[$(date)] HF=$HF_CHECKPOINT  TORCH_DIST=$TORCH_DIST"
echo "[$(date)] PROMPT_DATA=$PROMPT_DATA"
echo "[$(date)] RUN_DIR=$RUN_DIR"

INNER=$RUN_DIR/_snapshot/inner.sh
cat > $INNER <<TOP
#!/bin/bash
set -eu
NUM_ROLLOUT=$NUM_ROLLOUT
ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE
N_SAMPLES_PER_PROMPT=$N_SAMPLES_PER_PROMPT
GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE
SAVE_INTERVAL=$SAVE_INTERVAL
RLER=$RLER
RUN_DIR=$RUN_DIR
HF_CHECKPOINT=$HF_CHECKPOINT
TORCH_DIST=$TORCH_DIST
PROMPT_DATA=$PROMPT_DATA
SAVE_DIR=$SAVE_DIR
LOAD_DIR=$LOAD_DIR
TOP
cat >> $INNER <<'EOF'

NODE=$(hostname)
echo "[node=$NODE] inner start"

# Transient caches on local SSD (Lustre OSTs frequently full; sglang/triton/hf
# hammer these paths during init and OSError 28 will kill the run).
NODE_CACHE=/mnt/localssd/.cache/slime-fa-${SLURM_JOB_ID}
mkdir -p $NODE_CACHE/{flashinfer,triton,hf,torch,home,ray_tmp}
export FLASHINFER_WORKSPACE_DIR=$NODE_CACHE/flashinfer
export TRITON_CACHE_DIR=$NODE_CACHE/triton
export HF_HOME=$NODE_CACHE/hf
export TORCH_HOME=$NODE_CACHE/torch
export XDG_CACHE_HOME=$NODE_CACHE
export HOME=$NODE_CACHE/home
export TMPDIR=$NODE_CACHE
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1

# Wandb online if key is on lustre (mirrors grpo_naive_4node_cp4.slurm)
WANDB_KEY_FILE=/mnt/lustre/metavmds0lstre/share/sihanzeng_meta_com/secrets/wandb_api_key
if [ -r "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY=$(cat "$WANDB_KEY_FILE" | tr -d '[:space:]')
    export WANDB_MODE=online
    export WANDB_ENTITY=chenqinghan2003-university-of-washington
    WANDB_FLAGS=(--use-wandb --wandb-project slime-fully-async --wandb-group qwen3-4b-Instruct-2507 --wandb-run-name "fa-smoke-${SLURM_JOB_ID}")
    echo "[node=$NODE] wandb online enabled"
else
    export WANDB_MODE=offline
    WANDB_FLAGS=()
    echo "[node=$NODE] wandb key missing, staying offline"
fi

# === Step 1: convert HF -> torch_dist (idempotent, atomic via rename) ===
if [ ! -f "$TORCH_DIST/latest_checkpointed_iteration.txt" ]; then
    echo "[node=$NODE] no torch_dist at $TORCH_DIST, converting..."
    TMP_TD=${TORCH_DIST}.tmp.${SLURM_JOB_ID}
    rm -rf $TMP_TD && mkdir -p $TMP_TD
    cd $RLER/slime
    # Use Instruct-2507 model args (rope=5M matches HF config.json)
    source scripts/models/qwen3-4B-Instruct-2507.sh
    export PYTHONPATH="/root/Megatron-LM:${PYTHONPATH:-}"
    set -o pipefail
    python tools/convert_hf_to_torch_dist.py \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint $HF_CHECKPOINT \
        --save $TMP_TD \
        2>&1 | tee $RUN_DIR/_logs/convert.log
    CONV_RC=${PIPESTATUS[0]}
    set +o pipefail
    if [ "$CONV_RC" -ne 0 ]; then
        echo "[node=$NODE] FATAL: convert failed rc=$CONV_RC, see $RUN_DIR/_logs/convert.log"
        exit $CONV_RC
    fi
    if [ ! -f "$TMP_TD/latest_checkpointed_iteration.txt" ]; then
        echo "[node=$NODE] FATAL: convert produced no latest_checkpointed_iteration.txt"
        ls -la $TMP_TD
        exit 1
    fi
    mv $TMP_TD $TORCH_DIST
    echo "[node=$NODE] torch_dist ready at $TORCH_DIST"
else
    echo "[node=$NODE] torch_dist already present at $TORCH_DIST, skipping convert"
fi

# === Step 2: source model args + launch training ===
cd $RLER/slime
source scripts/models/qwen3-4B-Instruct-2507.sh

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "[node=$NODE] HAS_NVLINK=$HAS_NVLINK"

# Match the example's runtime env via plain shell exports BEFORE `ray
# start` so the raylet inherits them — Ray actors spawned by the driver
# (e.g. RolloutManager) then see PYTHONPATH and can import
# `fully_async_rollout`. Job 60247 failed here because exports happened
# AFTER `ray start`.
EXAMPLE_DIR=$RLER/slime/examples/fully_async
export PYTHONPATH="/root/Megatron-LM:${EXAMPLE_DIR}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=$HAS_NVLINK

# Ray on localhost (single-node). Disable dashboard — its MetricsHead
# module crashes in this container (job 60246), and we don't need it
# since we invoke train_async.py directly with RAY_ADDRESS instead of
# going through `ray job submit`.
MY_IP=$(hostname -I | awk '{print $1}')
RAY_PORT=6379
ray stop --force >/dev/null 2>&1 || true
sleep 2
ray start --head --node-ip-address=$MY_IP --port=$RAY_PORT \
    --num-gpus 8 --num-cpus 128 \
    --temp-dir $NODE_CACHE/ray_tmp \
    --disable-usage-stats --include-dashboard=false
sleep 3

export RAY_ADDRESS=$MY_IP:$RAY_PORT

DRIVER_LOG=$RUN_DIR/_logs/training.log
echo "[node=$NODE] launching driver -> $DRIVER_LOG"

set -o pipefail
python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint $HF_CHECKPOINT \
   --ref-load $TORCH_DIST \
   --load $LOAD_DIR \
   --save $SAVE_DIR \
   --save-interval $SAVE_INTERVAL \
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async \
   --prompt-data $PROMPT_DATA \
   --input-key prompt \
   --label-key label \
   --apply-chat-template \
   --rollout-shuffle \
   --rm-type dapo \
   --reward-key score \
   --rollout-max-prompt-len 2048 \
   --num-rollout $NUM_ROLLOUT \
   --rollout-batch-size $ROLLOUT_BATCH_SIZE \
   --n-samples-per-prompt $N_SAMPLES_PER_PROMPT \
   --rollout-max-response-len 8192 \
   --rollout-temperature 1 \
   --global-batch-size $GLOBAL_BATCH_SIZE \
   --balance-data \
   --tensor-model-parallel-size 2 \
   --sequence-parallel \
   --pipeline-model-parallel-size 1 \
   --context-parallel-size 1 \
   --expert-model-parallel-size 1 \
   --expert-tensor-parallel-size 1 \
   --recompute-granularity full \
   --recompute-method uniform \
   --recompute-num-layers 1 \
   --use-dynamic-batch-size \
   --max-tokens-per-gpu 12288 \
   --advantage-estimator grpo \
   --use-kl-loss \
   --kl-loss-coef 0.00 \
   --kl-loss-type low_var_kl \
   --entropy-coef 0.00 \
   --eps-clip 0.2 \
   --eps-clip-high 0.28 \
   --use-tis \
   --optimizer adam \
   --lr 1e-6 \
   --lr-decay-style constant \
   --weight-decay 0.1 \
   --adam-beta1 0.9 \
   --adam-beta2 0.98 \
   --rollout-num-gpus-per-engine 1 \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --accumulate-allreduce-grads-in-fp32 \
   --attention-softmax-in-fp32 \
   --attention-backend flash \
   "${WANDB_FLAGS[@]}" \
   2>&1 | tee $DRIVER_LOG
DRIVER_RC=${PIPESTATUS[0]}
set +o pipefail
echo "[node=$NODE] driver exited rc=$DRIVER_RC"

ray stop --force >/dev/null 2>&1 || true
exit $DRIVER_RC
EOF
chmod +x $INNER

echo "[$(date)] === srun into container ==="
srun --ntasks=1 --ntasks-per-node=1 \
    --container-image=$CONTAINER_IMAGE \
    --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,/mnt/localssd:/mnt/localssd \
    --container-workdir=$RLER/slime \
    bash $INNER

echo "[$(date)] DONE jobid=$SLURM_JOB_ID rc=$?"
