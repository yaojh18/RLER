#!/bin/bash
#SBATCH --job-name=slime-q35-9b-hle-evalonly-64k
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/slime-q35-9b-hle-evalonly-64k-%j.log

# Eval-only run on HLE 117-prompt held-out set with the cold Qwen3.5-9B
# (no GRPO updates). Goal: characterize the per-sample response-length
# distribution at a much larger 64K response cap to see (a) how the
# truncated_ratio at 24K (~44%) breaks down — model genuinely needs more
# than 24K vs stuck in infinite generation, and (b) typical length of
# CORRECT (reward=1) vs INCORRECT (reward=0) answers.
#
# Uses train.py (NOT train_async.py) which natively supports eval-only
# via `num_rollout=0` (train.py:40-41 short-circuits to a single
# rollout_manager.eval call and exits). No filter, no training.
#
# Also enables --save-debug-rollout-data so the per-sample Sample
# objects are dumped to disk for post-hoc length-distribution analysis.

set -u

BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng
RLER=$BASE/swe-rebench/RLER
HF_CHECKPOINT=/mnt/lustre/metavmds0lstre/teams/mrs_ranking_core_modeling/daili1/deepeyes/models/Qwen3.5-9B
TORCH_DIST=/mnt/lustre/metavmds0lstre/share/sihanzeng_meta_com/slime_qwen35/ckpt/Qwen3.5-9B_torch_dist_latest
EVAL_DATA=$BASE/datasets/hle/hle_mc_eval.jsonl
PROMPT_DATA=$BASE/datasets/hle/hle_mc_train.jsonl  # required by slime even when num_rollout=0
RUN_DIR=$BASE/rler-runs/${SLURM_JOB_ID}-slime-q35-9b-hle-evalonly-64k
SLURM_LOGS=$BASE/rler-runs/_slurm-logs
DUMP_DIR=$RUN_DIR/_eval_dumps
EMPTY_LOAD=$RUN_DIR/empty_load
mkdir -p $RUN_DIR/_logs $RUN_DIR/_snapshot $SLURM_LOGS $DUMP_DIR $EMPTY_LOAD

( cd $RLER && git rev-parse HEAD ) > $RUN_DIR/_snapshot/RLER_HEAD.txt 2>/dev/null || true
cp $0 $RUN_DIR/_snapshot/ 2>/dev/null || true

[ ! -d "$HF_CHECKPOINT" ] && { echo "ERROR: HF_CHECKPOINT=$HF_CHECKPOINT missing"; exit 1; }
[ ! -f "$EVAL_DATA" ]     && { echo "ERROR: EVAL_DATA=$EVAL_DATA missing";       exit 1; }

CONTAINER_IMAGE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh

echo "[$(date)] jobid=$SLURM_JOB_ID host=$(hostname) (eval-only, 64K response cap)"
echo "[$(date)] HF=$HF_CHECKPOINT  TORCH_DIST=$TORCH_DIST"
echo "[$(date)] EVAL=$EVAL_DATA  DUMP=$DUMP_DIR"

INNER=$RUN_DIR/_snapshot/inner.sh
cat > $INNER <<TOP
#!/bin/bash
set -eu
RLER=$RLER
RUN_DIR=$RUN_DIR
HF_CHECKPOINT=$HF_CHECKPOINT
TORCH_DIST=$TORCH_DIST
PROMPT_DATA=$PROMPT_DATA
EVAL_DATA=$EVAL_DATA
DUMP_DIR=$DUMP_DIR
EMPTY_LOAD=$EMPTY_LOAD
TOP
cat >> $INNER <<'EOF'

NODE=$(hostname)
echo "[node=$NODE] inner start (eval-only 64K)"

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WANDB_KEY_FILE=/mnt/lustre/metavmds0lstre/share/sihanzeng_meta_com/secrets/wandb_api_key
if [ -r "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY=$(cat "$WANDB_KEY_FILE" | tr -d '[:space:]')
    export WANDB_MODE=online
    export WANDB_ENTITY=chenqinghan2003-university-of-washington
    WANDB_FLAGS=(--use-wandb --wandb-project slime-fully-async --wandb-group qwen3.5-9B-hle-evalonly --wandb-run-name "q35-9b-hle-evalonly-64k-${SLURM_JOB_ID}")
else
    export WANDB_MODE=offline
    WANDB_FLAGS=()
fi

# torch_dist must exist already (we have the cached one)
[ ! -f "$TORCH_DIST/latest_checkpointed_iteration.txt" ] && { echo "FATAL: torch_dist missing"; exit 1; }

cd $RLER/slime
source train_agent/configs/qwen3.5-9B.sh

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

export PYTHONPATH="/root/Megatron-LM:${PYTHONPATH:-}"
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
echo "[node=$NODE] launching train.py (eval-only, num_rollout=0) -> $DRIVER_LOG"

set -o pipefail
python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint $HF_CHECKPOINT \
   --ref-load $TORCH_DIST \
   --load $EMPTY_LOAD \
   --prompt-data $PROMPT_DATA \
   --input-key prompt \
   --label-key label \
   --metadata-key metadata \
   --apply-chat-template \
   --rollout-shuffle \
   --rm-type gpqa \
   --rollout-max-prompt-len 8192 \
   --num-rollout 1 \
   --rollout-batch-size 32 \
   --n-samples-per-prompt 8 \
   --rollout-max-response-len 65536 \
   --rollout-temperature 1 \
   --global-batch-size 256 \
   --balance-data \
   --eval-interval 1 \
   --eval-prompt-data hle-held-out $EVAL_DATA \
   --n-samples-per-eval-prompt 1 \
   --eval-max-response-len 65536 \
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
   --max-tokens-per-gpu 73728 \
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
   --save-debug-rollout-data "${DUMP_DIR}/{rollout_id}.pt" \
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
