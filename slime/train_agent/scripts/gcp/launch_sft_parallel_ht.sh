#!/bin/bash
# Launch the parallel-search SFT pipeline AGAINST the high-throughput DSv4
# recipe (DP=8 + DP-attention + mega-MoE + radix cache ON). Companion to
# `launch_sft_parallel.sh` which uses the low-latency recipe.
#
# Required env:
#   ARTIFACT_ROOT   output dir on Lustre
#   INSTANCE_IDS    space-separated list of swe-rebench instance IDs
# Optional env: same as launch_sft_parallel.sh (INSTANCE_WORKERS, SEARCH_M,
# SEARCH_N, LANES_MAX_MID_CPS, LANES_STEPS_PER_ROUND, SEARCH_STEP_LIMIT,
# LANES_GT_EVAL_WORKERS, RUBRIC_BANK_STRATEGY, RLER, BASE).

set -euo pipefail
: "${ARTIFACT_ROOT:?ARTIFACT_ROOT is required}"
: "${INSTANCE_IDS:?INSTANCE_IDS is required (space-separated)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLER=${RLER:-"$(cd "$SCRIPT_DIR/../../../.." && pwd)"}
BASE=${BASE:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench}
IMAGE_SQSH=${IMAGE_SQSH:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/sglang_dsv4_b200.sqsh}

mkdir -p "$ARTIFACT_ROOT"
DISCOVERY_DIR="$ARTIFACT_ROOT/discovery"
mkdir -p "$DISCOVERY_DIR"
DSV4_DISCOVERY="$DISCOVERY_DIR/dsv4.json"

# Submit HT-variant DSv4 teacher
echo "[launcher_ht] submitting DSv4 teacher (HIGH-THROUGHPUT recipe) sbatch..."
DSV4_JID=$(
    sbatch --parsable \
        --export=ALL,RLER=$RLER,DISCOVERY=$DSV4_DISCOVERY,IMAGE_SQSH=$IMAGE_SQSH \
        "$SCRIPT_DIR/sbatch_dsv4_teacher_high_throughput.sbatch"
)
echo "[launcher_ht] dsv4_ht jobid=$DSV4_JID  discovery=$DSV4_DISCOVERY"

# Collector — same as low_latency path
COLLECTOR_IMG=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh
echo "[launcher_ht] submitting SFT collector (depends after:$DSV4_JID)..."
COLLECTOR_JID=$(
    sbatch --parsable \
        --dependency=after:$DSV4_JID \
        --export=ALL,RLER=$RLER,BASE=$BASE,IMAGE_SQSH=$COLLECTOR_IMG,\
ARTIFACT_ROOT=$ARTIFACT_ROOT,DSV4_DISCOVERY=$DSV4_DISCOVERY,\
INSTANCE_IDS="$INSTANCE_IDS",\
SUBSET=${SUBSET:-rebench_v2},SPLIT=${SPLIT:-train},\
INSTANCE_WORKERS=${INSTANCE_WORKERS:-24},\
SEARCH_M=${SEARCH_M:-8},SEARCH_N=${SEARCH_N:-2},\
LANES_MAX_MID_CPS=${LANES_MAX_MID_CPS:-6},\
LANES_STEPS_PER_ROUND=${LANES_STEPS_PER_ROUND:-20},\
SEARCH_STEP_LIMIT=${SEARCH_STEP_LIMIT:-120},\
LANES_GT_EVAL_WORKERS=${LANES_GT_EVAL_WORKERS:-8},\
RUBRIC_BANK_STRATEGY=${RUBRIC_BANK_STRATEGY:-score} \
        "$SCRIPT_DIR/sbatch_collector_sft_parallel.sbatch"
)
echo "[launcher_ht] collector jobid=$COLLECTOR_JID"
echo "[launcher_ht] tail logs:"
echo "    tail -F /mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/dsv4_teacher_ht-${DSV4_JID}.log"
echo "    tail -F /mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/sft_parallel-${COLLECTOR_JID}.log"
echo "    summary.json: $ARTIFACT_ROOT/summary.json"
