#!/bin/bash
#SBATCH --job-name=slime-q35-9b-hle
#SBATCH --account=ads_cpt
#SBATCH --qos=ads_cpt
#SBATCH --partition=a4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --exclude=metavmds1-a4-21,metavmds1-a4-9
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/slime-q35-9b-hle-%j.log

# Slime GRPO on Qwen3.5-9B + HLE multi-choice (464 train / 117 eval) with
# 32K total per-sample context (8K prompt + 24K response) and held-out
# eval every 5 rollouts. Same async + miles-trick eval recipe as
# run_qwen35_9b_dapo_eval.slurm. RM is the built-in `gpqa` scorer which
# reads metadata.choices + metadata.valid_letters and matches the
# "Final Answer: X" format that the HLE prompt asks for.
#
# Parameterised by env DYNAMIC_FILTER=on|off (default on) so the same
# script can be sbatch'd twice to run the filter A/B comparison:
#
#   sbatch --export=ALL,DYNAMIC_FILTER=on  run_qwen35_9b_hle_eval.slurm
#   sbatch --export=ALL,DYNAMIC_FILTER=off run_qwen35_9b_hle_eval.slurm

set -u

# --- Tunables ---
NUM_ROLLOUT=60
ROLLOUT_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
GLOBAL_BATCH_SIZE=256
SAVE_INTERVAL=20
EVAL_INTERVAL=5
DYNAMIC_FILTER=${DYNAMIC_FILTER:-on}
# ----------------

BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng
RLER=$BASE/swe-rebench/RLER
HF_CHECKPOINT=/mnt/lustre/metavmds0lstre/teams/mrs_ranking_core_modeling/daili1/deepeyes/models/Qwen3.5-9B
TORCH_DIST=/mnt/lustre/metavmds0lstre/share/sihanzeng_meta_com/slime_qwen35/ckpt/Qwen3.5-9B_torch_dist_latest
PROMPT_DATA=$BASE/datasets/hle/hle_mc_train.jsonl
EVAL_DATA=$BASE/datasets/hle/hle_mc_eval.jsonl
RUN_DIR=$BASE/rler-runs/${SLURM_JOB_ID}-slime-q35-9b-hle-${DYNAMIC_FILTER}
SLURM_LOGS=$BASE/rler-runs/_slurm-logs
SAVE_DIR=$RUN_DIR/policy_ckpt
LOAD_DIR=$SAVE_DIR
mkdir -p $RUN_DIR/_logs $RUN_DIR/_snapshot $SLURM_LOGS $SAVE_DIR

( cd $RLER && git rev-parse HEAD ) > $RUN_DIR/_snapshot/RLER_HEAD.txt 2>/dev/null || true
cp $0 $RUN_DIR/_snapshot/ 2>/dev/null || true

[ ! -d "$HF_CHECKPOINT" ] && { echo "ERROR: HF_CHECKPOINT=$HF_CHECKPOINT missing"; exit 1; }
[ ! -f "$PROMPT_DATA" ]   && { echo "ERROR: PROMPT_DATA=$PROMPT_DATA missing";   exit 1; }
[ ! -f "$EVAL_DATA" ]     && { echo "ERROR: EVAL_DATA=$EVAL_DATA missing";       exit 1; }

CONTAINER_IMAGE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh

echo "[$(date)] jobid=$SLURM_JOB_ID host=$(hostname) DYNAMIC_FILTER=$DYNAMIC_FILTER"
echo "[$(date)] HF=$HF_CHECKPOINT  TORCH_DIST=$TORCH_DIST"
echo "[$(date)] TRAIN=$PROMPT_DATA  EVAL=$EVAL_DATA"
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
EVAL_INTERVAL=$EVAL_INTERVAL
DYNAMIC_FILTER=$DYNAMIC_FILTER
RLER=$RLER
RUN_DIR=$RUN_DIR
HF_CHECKPOINT=$HF_CHECKPOINT
TORCH_DIST=$TORCH_DIST
PROMPT_DATA=$PROMPT_DATA
EVAL_DATA=$EVAL_DATA
SAVE_DIR=$SAVE_DIR
LOAD_DIR=$LOAD_DIR
TOP
cat >> $INNER <<'EOF'

NODE=$(hostname)
echo "[node=$NODE] inner start DYNAMIC_FILTER=$DYNAMIC_FILTER"

NODE_CACHE=/mnt/localssd/.cache/slime-${SLURM_JOB_ID}
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
# 9B + sglang on a single 4-GPU placement group fragments PyTorch's
# allocator and OOMs without expandable_segments (matches miles 9B fix).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WANDB_KEY_FILE=/mnt/lustre/metavmds0lstre/share/sihanzeng_meta_com/secrets/wandb_api_key
if [ -r "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY=$(cat "$WANDB_KEY_FILE" | tr -d '[:space:]')
    export WANDB_MODE=online
    export WANDB_ENTITY=chenqinghan2003-university-of-washington
    WANDB_FLAGS=(--use-wandb --wandb-project slime-fully-async --wandb-group qwen3.5-9B-hle --wandb-run-name "q35-9b-hle-${DYNAMIC_FILTER}-${SLURM_JOB_ID}")
    echo "[node=$NODE] wandb online enabled"
else
    export WANDB_MODE=offline
    WANDB_FLAGS=()
    echo "[node=$NODE] wandb key missing, staying offline"
fi

# Conditional --dynamic-sampling-filter-path
if [ "$DYNAMIC_FILTER" = "on" ]; then
    FILTER_FLAGS=(--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
else
    FILTER_FLAGS=()
fi

# === Step 1: convert HF -> torch_dist (idempotent) ===
if [ ! -f "$TORCH_DIST/latest_checkpointed_iteration.txt" ]; then
    echo "[node=$NODE] converting HF -> torch_dist..."
    TMP_TD=${TORCH_DIST}.tmp.${SLURM_JOB_ID}
    rm -rf $TMP_TD && mkdir -p $TMP_TD
    cd $RLER/slime
    source train_agent/configs/qwen3.5-9B.sh
    export PYTHONPATH="/root/Megatron-LM:${PYTHONPATH:-}"
    set -o pipefail
    python tools/convert_hf_to_torch_dist.py \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint $HF_CHECKPOINT --save $TMP_TD \
        2>&1 | tee $RUN_DIR/_logs/convert.log
    CONV_RC=${PIPESTATUS[0]}
    set +o pipefail
    [ "$CONV_RC" -ne 0 ] && { echo "convert failed rc=$CONV_RC"; exit $CONV_RC; }
    [ ! -f "$TMP_TD/latest_checkpointed_iteration.txt" ] && { echo "no checkpoint marker"; exit 1; }
    mv $TMP_TD $TORCH_DIST
fi
echo "[node=$NODE] torch_dist ready at $TORCH_DIST"

# === Step 2: source model args + launch training ===
cd $RLER/slime
source train_agent/configs/qwen3.5-9B.sh

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

EXAMPLE_DIR=$RLER/slime/examples/fully_async
export PYTHONPATH="/root/Megatron-LM:${EXAMPLE_DIR}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=$HAS_NVLINK

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
echo "[node=$NODE] launching train_async.py -> $DRIVER_LOG"

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
   --eval-function-path slime.rollout.sglang_rollout.generate_rollout \
   --prompt-data $PROMPT_DATA \
   --input-key prompt \
   --label-key label \
   --metadata-key metadata \
   --apply-chat-template \
   --rollout-shuffle \
   --rm-type gpqa \
   --rollout-max-prompt-len 8192 \
   --num-rollout $NUM_ROLLOUT \
   --rollout-batch-size $ROLLOUT_BATCH_SIZE \
   --n-samples-per-prompt $N_SAMPLES_PER_PROMPT \
   --rollout-max-response-len 24576 \
   --rollout-temperature 1 \
   --global-batch-size $GLOBAL_BATCH_SIZE \
   --balance-data \
   --eval-interval $EVAL_INTERVAL \
   --eval-prompt-data hle-held-out $EVAL_DATA \
   --n-samples-per-eval-prompt 1 \
   --eval-max-response-len 24576 \
   --eval-top-k 1 \
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
   --max-tokens-per-gpu 32768 \
   --advantage-estimator grpo \
   --use-kl-loss \
   --kl-loss-coef 0.00 \
   --kl-loss-type low_var_kl \
   --entropy-coef 0.00 \
   --eps-clip 0.2 \
   --eps-clip-high 0.28 \
   --optimizer adam \
   --lr 1e-6 \
   --lr-decay-style constant \
   --weight-decay 0.1 \
   --adam-beta1 0.9 \
   --adam-beta2 0.98 \
   --rollout-num-gpus-per-engine 1 \
   --sglang-mem-fraction-static 0.5 \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --accumulate-allreduce-grads-in-fp32 \
   --attention-softmax-in-fp32 \
   --attention-backend flash \
   "${FILTER_FLAGS[@]}" \
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
