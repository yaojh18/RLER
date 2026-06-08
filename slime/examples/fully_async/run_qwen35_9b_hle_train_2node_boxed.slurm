#!/bin/bash
#SBATCH --job-name=q35-9b-hle-train-2n-boxed
#SBATCH --account=ads_cpt
#SBATCH --qos=ads_cpt
#SBATCH --partition=a4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --exclude=metavmds1-a4-21,metavmds1-a4-9,metavmds1-a4-94
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/q35-9b-hle-train-2n-boxed-%j.log

# 2-node GRPO training on Qwen3.5-9B + HLE-MC with STRICT \boxed{X} format.
#
# Mirrors miles' 2-node layout: node0=8 actor GPUs (TP=2, DP=4),
# node1=8 sglang engines (TP=1 each). Reward via the strict
# boxed_letter_rm custom RM (truncated → 0; otherwise pull the LAST
# \boxed{X} via regex and 1-or-0 against label). Closes the substring
# fallback + <|im_end|> normalization bugs that caused 61647's policy
# collapse to 2-3 token "stub" outputs at eval 49.
#
# Dataset is the boxed-prompt variant (hle_mc_{train,eval}_boxed.jsonl)
# generated from the original HLE-MC with a system message asking for
# 'final line: \boxed{X}'.
#
# Knobs via env:
#   DYNAMIC_FILTER=on|off          # default on (matches miles)
#   ENABLE_THINKING=true|false     # default true (Qwen3.5 thinking)
#
#   sbatch --export=ALL,DYNAMIC_FILTER=on,ENABLE_THINKING=true  run_qwen35_9b_hle_train_2node_boxed.slurm

set -u

# --- Tunables ---
NUM_ROLLOUT=200
ROLLOUT_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
GLOBAL_BATCH_SIZE=256
SAVE_INTERVAL=50
EVAL_INTERVAL=5
ACTOR_GPUS_PER_NODE=8
ROLLOUT_NUM_GPUS=8
DYNAMIC_FILTER=${DYNAMIC_FILTER:-on}
ENABLE_THINKING=${ENABLE_THINKING:-true}
# ----------------

BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng
RLER=$BASE/swe-rebench/RLER
HF_CHECKPOINT=/mnt/lustre/metavmds0lstre/teams/mrs_ranking_core_modeling/daili1/deepeyes/models/Qwen3.5-9B
TORCH_DIST=/mnt/lustre/metavmds0lstre/share/sihanzeng_meta_com/slime_qwen35/ckpt/Qwen3.5-9B_torch_dist_latest
PROMPT_DATA=$BASE/datasets/hle/hle_mc_train_boxed.jsonl
EVAL_DATA=$BASE/datasets/hle/hle_mc_eval_boxed.jsonl
RUN_DIR=$BASE/rler-runs/${SLURM_JOB_ID}-q35-9b-hle-train-2n-boxed-think${ENABLE_THINKING}-filter${DYNAMIC_FILTER}
SLURM_LOGS=$BASE/rler-runs/_slurm-logs
SAVE_DIR=$RUN_DIR/policy_ckpt
LOAD_DIR=$SAVE_DIR
DUMP_DIR=$RUN_DIR/_dumps
HEAD_IP_FILE=$RUN_DIR/_snapshot/head_ip.txt
RAY_PORT=6379
mkdir -p $RUN_DIR/_logs $RUN_DIR/_snapshot $SLURM_LOGS $SAVE_DIR $DUMP_DIR

( cd $RLER && git rev-parse HEAD ) > $RUN_DIR/_snapshot/RLER_HEAD.txt 2>/dev/null || true
cp $0 $RUN_DIR/_snapshot/ 2>/dev/null || true
rm -f $HEAD_IP_FILE

[ ! -d "$HF_CHECKPOINT" ] && { echo "ERROR: HF_CHECKPOINT missing"; exit 1; }
[ ! -f "$PROMPT_DATA" ]   && { echo "ERROR: PROMPT_DATA missing";   exit 1; }
[ ! -f "$EVAL_DATA" ]     && { echo "ERROR: EVAL_DATA missing";     exit 1; }

CONTAINER_IMAGE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh

echo "[$(date)] jobid=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST DYNAMIC_FILTER=$DYNAMIC_FILTER ENABLE_THINKING=$ENABLE_THINKING"
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
ACTOR_GPUS_PER_NODE=$ACTOR_GPUS_PER_NODE
ROLLOUT_NUM_GPUS=$ROLLOUT_NUM_GPUS
DYNAMIC_FILTER=$DYNAMIC_FILTER
ENABLE_THINKING=$ENABLE_THINKING
RLER=$RLER
RUN_DIR=$RUN_DIR
HF_CHECKPOINT=$HF_CHECKPOINT
TORCH_DIST=$TORCH_DIST
PROMPT_DATA=$PROMPT_DATA
EVAL_DATA=$EVAL_DATA
SAVE_DIR=$SAVE_DIR
LOAD_DIR=$LOAD_DIR
DUMP_DIR=$DUMP_DIR
HEAD_IP_FILE=$HEAD_IP_FILE
RAY_PORT=$RAY_PORT
NUM_NODES=$SLURM_NNODES
TOP
cat >> $INNER <<'EOF'

NODE=$(hostname)
NODE_RANK=${SLURM_NODEID:-0}
echo "[node=$NODE rank=$NODE_RANK/$NUM_NODES] inner start (boxed RM, 2-node)"

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
    WANDB_FLAGS=(--use-wandb --wandb-project slime-fully-async --wandb-group qwen3.5-9B-hle-2n-boxed --wandb-run-name "q35-9b-hle-2n-boxed-think${ENABLE_THINKING}-filter${DYNAMIC_FILTER}-${SLURM_JOB_ID}")
else
    export WANDB_MODE=offline
    WANDB_FLAGS=()
fi

if [ "$DYNAMIC_FILTER" = "on" ]; then
    FILTER_FLAGS=(--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
else
    FILTER_FLAGS=()
fi
if [ "$ENABLE_THINKING" = "false" ]; then
    THINK_FLAGS=(--apply-chat-template-kwargs '{"enable_thinking": false}')
else
    THINK_FLAGS=()
fi

[ ! -f "$TORCH_DIST/latest_checkpointed_iteration.txt" ] && { echo "FATAL: torch_dist missing"; exit 1; }

cd $RLER/slime
source train_agent/configs/qwen3.5-9B.sh

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

EXAMPLE_DIR=$RLER/slime/examples/fully_async
export PYTHONPATH="/root/Megatron-LM:${EXAMPLE_DIR}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=$HAS_NVLINK

MY_IP=$(hostname -I | tr ' ' '\n' | grep -E '^192\.168\.' | head -1)
if [ -z "$MY_IP" ]; then MY_IP=$(hostname -I | awk '{print $1}'); fi
ray stop --force >/dev/null 2>&1 || true
sleep 2

if [ "$NODE_RANK" = "0" ]; then
    echo "[head $NODE] my_ip=$MY_IP starting ray head"
    ray start --head --node-ip-address=$MY_IP --port=$RAY_PORT \
        --num-gpus 8 --num-cpus 128 \
        --temp-dir $NODE_CACHE/ray_tmp \
        --disable-usage-stats --include-dashboard=false
    sleep 5
    echo "$MY_IP" > $HEAD_IP_FILE
    export RAY_ADDRESS=$MY_IP:$RAY_PORT
    echo "[head $NODE] waiting for workers to join..."
    EXPECTED_GPUS=$((NUM_NODES * 8))
    for i in $(seq 1 60); do
        STATUS=$(ray status 2>&1)
        AVAIL_GPUS=$(echo "$STATUS" | awk '/^ +[0-9.]+\/[0-9.]+ GPU/{split($1,a,"/"); print int(a[2]); exit}')
        AVAIL_GPUS=${AVAIL_GPUS:-0}
        echo "[head $NODE] ray: $AVAIL_GPUS / $EXPECTED_GPUS GPUs"
        if [ "$AVAIL_GPUS" -ge "$EXPECTED_GPUS" ]; then break; fi
        sleep 5
    done
    ray status 2>&1 | head -40

    DRIVER_LOG=$RUN_DIR/_logs/training.log
    echo "[head $NODE] launching train_async.py -> $DRIVER_LOG"

    set -o pipefail
    python3 train_async.py \
       --actor-num-nodes 1 \
       --actor-num-gpus-per-node $ACTOR_GPUS_PER_NODE \
       --rollout-num-gpus $ROLLOUT_NUM_GPUS \
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
       --custom-rm-path boxed_letter_rm.custom_rm \
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
       --eval-temperature 1.0 \
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
       --sglang-mem-fraction-static 0.85 \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --accumulate-allreduce-grads-in-fp32 \
       --attention-softmax-in-fp32 \
       --attention-backend flash \
       --save-debug-rollout-data "${DUMP_DIR}/{rollout_id}.pt" \
       "${THINK_FLAGS[@]}" \
       "${FILTER_FLAGS[@]}" \
       "${WANDB_FLAGS[@]}" \
       2>&1 | tee $DRIVER_LOG
    DRIVER_RC=${PIPESTATUS[0]}
    set +o pipefail
    echo "[head $NODE] driver exited rc=$DRIVER_RC"
    # Signal workers
    touch ${HEAD_IP_FILE}.done
    ray stop --force >/dev/null 2>&1 || true
    exit $DRIVER_RC
else
    echo "[worker $NODE] my_ip=$MY_IP waiting for head IP file..."
    for i in $(seq 1 120); do
        if [ -f $HEAD_IP_FILE ]; then HEAD_IP=$(cat $HEAD_IP_FILE); break; fi
        sleep 5
    done
    [ -z "${HEAD_IP:-}" ] && { echo "[worker $NODE] FATAL: no head IP"; exit 1; }
    echo "[worker $NODE] joining ray head at $HEAD_IP:$RAY_PORT"
    ray start --address=$HEAD_IP:$RAY_PORT --node-ip-address=$MY_IP \
        --num-gpus 8 --num-cpus 128 \
        --temp-dir $NODE_CACHE/ray_tmp --disable-usage-stats
    echo "[worker $NODE] joined; waiting for driver-done flag"
    for i in $(seq 1 17280); do
        [ -f ${HEAD_IP_FILE}.done ] && break
        sleep 10
    done
    echo "[worker $NODE] driver finished; stopping ray"
    ray stop --force >/dev/null 2>&1 || true
fi
EOF
chmod +x $INNER

echo "[$(date)] === srun fan-out across $SLURM_NNODES nodes ==="
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    --container-image=$CONTAINER_IMAGE \
    --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,/mnt/localssd:/mnt/localssd \
    --container-workdir=$RLER/slime \
    bash $INNER

echo "[$(date)] DONE jobid=$SLURM_JOB_ID rc=$?"
