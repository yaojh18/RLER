#!/bin/bash
#SBATCH --job-name=sft_dyn
#SBATCH --account=ads_cpt
#SBATCH --qos=ads_cpt
#SBATCH --partition=a4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gpus-per-node=0
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/sft_dyn-%j.log

# Dynamic-scheduling collector. Reads instances from a SHARED queue file,
# processes one at a time (INSTANCE_WORKERS=1 effectively), pushes to a
# shared done.jsonl. Replaces static round-robin sharding.
#
# Required env (passed via --export):
#   ARTIFACT_ROOT      output dir (contains queue.txt + done.jsonl)
#   DSV4_BASE_URL      teacher endpoint
#   COLLECTOR_ID       per-collector tag for logs
# Optional env: SEARCH_M, SEARCH_N, LANES_MAX_MID_CPS, LANES_STEPS_PER_ROUND,
#               SEARCH_STEP_LIMIT, LANES_GT_EVAL_WORKERS, RUBRIC_BANK_STRATEGY

set -euo pipefail
: "${ARTIFACT_ROOT:?ARTIFACT_ROOT is required}"
: "${DSV4_BASE_URL:?DSV4_BASE_URL is required}"
: "${COLLECTOR_ID:=col-${SLURM_JOB_ID}}"

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

QUEUE_FILE=$ARTIFACT_ROOT/queue.txt
DONE_FILE=$ARTIFACT_ROOT/done.jsonl

echo "[$(date)] jobid=$SLURM_JOB_ID node=$SLURMD_NODENAME col=$COLLECTOR_ID"
echo "  ARTIFACT_ROOT=$ARTIFACT_ROOT  DSV4=$DSV4_BASE_URL"
echo "  queue=$QUEUE_FILE  done=$DONE_FILE"

CACHE_BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/.cache/sft-dyn-${SLURM_JOB_ID}
mkdir -p "$CACHE_BASE"

exec srun --container-image="$IMAGE_SQSH" \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,/var/run/docker.sock:/var/run/docker.sock,/usr/bin/docker:/usr/bin/docker \
     --container-workdir=$RLER/slime \
     --container-writable \
     --no-container-mount-home \
     --export=ALL,RLER=$RLER,BASE=$BASE,CACHE_BASE=$CACHE_BASE,\
ARTIFACT_ROOT=$ARTIFACT_ROOT,DSV4_BASE_URL=$DSV4_BASE_URL,COLLECTOR_ID=$COLLECTOR_ID,\
QUEUE_FILE=$QUEUE_FILE,DONE_FILE=$DONE_FILE,\
SEARCH_M=$SEARCH_M,SEARCH_N=$SEARCH_N,LANES_MAX_MID_CPS=$LANES_MAX_MID_CPS,\
LANES_STEPS_PER_ROUND=$LANES_STEPS_PER_ROUND,SEARCH_STEP_LIMIT=$SEARCH_STEP_LIMIT,\
LANES_GT_EVAL_WORKERS=$LANES_GT_EVAL_WORKERS,RUBRIC_BANK_STRATEGY=$RUBRIC_BANK_STRATEGY \
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

python3 -u $RLER/slime/train_agent/scripts/gcp/dynamic_collector_loop.py \
    --queue-file $QUEUE_FILE \
    --done-file $DONE_FILE \
    --teacher-base-url $DSV4_BASE_URL \
    --output-root $ARTIFACT_ROOT/search_outputs \
    --collector-id $COLLECTOR_ID \
    --rler-root $RLER \
    --search-m $SEARCH_M --search-n $SEARCH_N \
    --lanes-max-mid-cps $LANES_MAX_MID_CPS \
    --lanes-steps-per-round $LANES_STEPS_PER_ROUND \
    --search-step-limit $SEARCH_STEP_LIMIT \
    --gt-eval-workers $LANES_GT_EVAL_WORKERS \
    --rubric-bank-strategy $RUBRIC_BANK_STRATEGY
'
