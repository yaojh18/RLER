#!/bin/bash
#SBATCH --job-name=sft_collector_cpu
#SBATCH --account=ads_cpt
#SBATCH --qos=ads_cpt
#SBATCH --partition=a4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gpus-per-node=0
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/sft_collector_cpu-%j.log

# CPU-only collector variant — shareable with other GPU jobs on the same
# node (no --exclusive, no GPU ask). The collector talks to an EXTERNAL
# DSv4 teacher; all model compute is on the teacher node, so there's no
# reason to hold a full B200 node here.
#
# Why: the original sbatch_collector_sft_parallel.sbatch was --exclusive
# + --mem=0 (full node, 3.85TB RAM), which idled the 8 GPUs and blocked
# other workloads from using them. The collector's actual footprint is
# Python ThreadPoolExecutor (24 instances × ~50 MB) + a docker exec
# per agent step (~1-2 GB transient per running container). Even with
# 192 simultaneous containers (24 × 8 Lane B), we never approach 512G.
#
# Required env (passed via --export):
#   ARTIFACT_ROOT    output dir on Lustre
#   DSV4_DISCOVERY   path to DSv4 server discovery JSON
#   INSTANCE_IDS     space-separated list of swe-rebench instance IDs
# Optional env: same as sbatch_collector_sft_parallel.sbatch
#   (INSTANCE_WORKERS, SEARCH_M, SEARCH_N, LANES_MAX_MID_CPS,
#    LANES_STEPS_PER_ROUND, SEARCH_STEP_LIMIT, LANES_GT_EVAL_WORKERS,
#    RUBRIC_BANK_STRATEGY)

set -euo pipefail

: "${ARTIFACT_ROOT:?ARTIFACT_ROOT is required}"
: "${DSV4_DISCOVERY:?DSV4_DISCOVERY is required}"
: "${INSTANCE_IDS:?INSTANCE_IDS is required (space-separated)}"

SUBSET=${SUBSET:-rebench_v2}
SPLIT=${SPLIT:-train}
INSTANCE_WORKERS=${INSTANCE_WORKERS:-16}
SEARCH_M=${SEARCH_M:-8}
SEARCH_N=${SEARCH_N:-2}
LANES_MAX_MID_CPS=${LANES_MAX_MID_CPS:-6}
LANES_STEPS_PER_ROUND=${LANES_STEPS_PER_ROUND:-20}
SEARCH_STEP_LIMIT=${SEARCH_STEP_LIMIT:-120}
LANES_GT_EVAL_WORKERS=${LANES_GT_EVAL_WORKERS:-8}
RUBRIC_BANK_STRATEGY=${RUBRIC_BANK_STRATEGY:-score}

RLER=${RLER:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/RLER-zsh-trim}
BASE=${BASE:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench}
IMAGE_SQSH=${IMAGE_SQSH:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh}

if [ ! -f "$RLER/slime/train_agent/scripts/run_teacher_data_collect.py" ]; then
    echo "ERROR: run_teacher_data_collect.py not under \$RLER=$RLER" >&2
    exit 2
fi

mkdir -p "$ARTIFACT_ROOT"
echo "[$(date)] jobid=$SLURM_JOB_ID node=$SLURMD_NODENAME artifact=$ARTIFACT_ROOT"
echo "  dsv4_discovery=$DSV4_DISCOVERY"
echo "  instances=$(echo $INSTANCE_IDS | wc -w)"
echo "  workers=$INSTANCE_WORKERS m=$SEARCH_M n=$SEARCH_N max_mid_cps=$LANES_MAX_MID_CPS"
echo "  RESOURCE ASK: cpus=64 mem=512G gpus=0 (shareable node)"

# Wait for DSv4 discovery file + ready=true. --dependency=after only
# guarantees the teacher STARTED, not that sglang booted (~13 min cold).
echo "[$(date)] waiting for DSv4 ready..."
for i in $(seq 1 600); do
    if [ -f "$DSV4_DISCOVERY" ]; then
        if python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ready') else 1)" "$DSV4_DISCOVERY" 2>/dev/null; then
            echo "[$(date)] DSv4 ready after ${i} polls"
            break
        fi
    fi
    sleep 10
done
DSV4_BASE_URL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('base_url') or '')" "$DSV4_DISCOVERY" 2>/dev/null)
if [ -z "$DSV4_BASE_URL" ]; then
    echo "ERROR: DSv4 never became ready (waited 100 min) — discovery=$DSV4_DISCOVERY"
    cat "$DSV4_DISCOVERY" 2>/dev/null || true
    exit 3
fi
echo "  DSV4_BASE_URL=$DSV4_BASE_URL"

CACHE_BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/.cache/sft-cpu-${SLURM_JOB_ID}
mkdir -p "$CACHE_BASE"

INSTANCE_ARGS="--instance-id $INSTANCE_IDS"

exec srun --container-image="$IMAGE_SQSH" \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,/var/run/docker.sock:/var/run/docker.sock,/usr/bin/docker:/usr/bin/docker \
     --container-workdir=$RLER/slime \
     --container-writable \
     --no-container-mount-home \
     --export=ALL,RLER=$RLER,BASE=$BASE,CACHE_BASE=$CACHE_BASE,\
ARTIFACT_ROOT=$ARTIFACT_ROOT,DSV4_BASE_URL=$DSV4_BASE_URL,\
SUBSET=$SUBSET,SPLIT=$SPLIT,INSTANCE_WORKERS=$INSTANCE_WORKERS,\
SEARCH_M=$SEARCH_M,SEARCH_N=$SEARCH_N,LANES_MAX_MID_CPS=$LANES_MAX_MID_CPS,\
LANES_STEPS_PER_ROUND=$LANES_STEPS_PER_ROUND,SEARCH_STEP_LIMIT=$SEARCH_STEP_LIMIT,\
LANES_GT_EVAL_WORKERS=$LANES_GT_EVAL_WORKERS,RUBRIC_BANK_STRATEGY=$RUBRIC_BANK_STRATEGY,\
INSTANCE_ARGS="$INSTANCE_ARGS" \
     bash -lc '
set -euo pipefail
export PYTHONPATH=$RLER/slime:$RLER:$RLER/agent:$BASE/SWE-rebench-V2:${PYTHONPATH:-}
export HOME=$CACHE_BASE
export XDG_CACHE_HOME=$CACHE_BASE
export HF_HOME=$CACHE_BASE/hf
mkdir -p "$HF_HOME"

export MSWEA_DOCKER_EXECUTABLE=$BASE/bin/docker
export DOCKER_LUSTRE_TARDIR=$BASE/docker_tarballs
export DOCKER_LUSTRE_REAL=/usr/bin/docker
export MSWEA_COST_TRACKING=ignore_errors

export OPENAI_API_BASE=$DSV4_BASE_URL/v1
export OPENAI_API_KEY=EMPTY

python3 -u $RLER/slime/train_agent/scripts/run_teacher_data_collect.py \
    $INSTANCE_ARGS \
    --use-parallel-search \
    --teacher-model openai/deepseek-v4-pro \
    --teacher-backend openai \
    --teacher-base-url $DSV4_BASE_URL \
    --subset "$SUBSET" --split "$SPLIT" \
    --search-output-root "$ARTIFACT_ROOT/search_outputs" \
    --artifact-root "$ARTIFACT_ROOT" \
    --summary-path "$ARTIFACT_ROOT/summary.json" \
    --instance-workers "$INSTANCE_WORKERS" \
    --search-m "$SEARCH_M" \
    --search-n "$SEARCH_N" \
    --lanes-max-mid-cps "$LANES_MAX_MID_CPS" \
    --lanes-steps-per-round "$LANES_STEPS_PER_ROUND" \
    --search-step-limit "$SEARCH_STEP_LIMIT" \
    --lanes-gt-eval-workers "$LANES_GT_EVAL_WORKERS" \
    --rubric-bank-strategy "$RUBRIC_BANK_STRATEGY" \
    --skip-prepare-instance-images \
    --slime-keepalive-interval-seconds 0
'
