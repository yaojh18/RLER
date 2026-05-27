#!/bin/bash
# Launch the parallel-search SFT collection pipeline:
#   1. DSv4 teacher sglang server (1 node, 8×B200)
#   2. SFT collector (1 cpu-only node, talks to DSv4 via lustre-shared
#      discovery file). Driven by run_teacher_data_collect.py
#      --use-parallel-search, fully bypasses the slime student.
#
# Required env:
#   ARTIFACT_ROOT   output dir on Lustre
#   INSTANCE_IDS    space-separated list of swe-rebench instance IDs
# Optional env (passed through to the collector sbatch):
#   SUBSET                default rebench_v2
#   SPLIT                 default train
#   INSTANCE_WORKERS      default 16
#   SEARCH_M, SEARCH_N
#   LANES_MAX_MID_CPS, LANES_STEPS_PER_ROUND, SEARCH_STEP_LIMIT
#   LANES_GT_EVAL_WORKERS
#   RUBRIC_BANK_STRATEGY
#   RLER, BASE, IMAGE_SQSH    paths into the lustre tree (defaults match
#                              the rest of the gcp/ sbatches)

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

# Submit DSv4 teacher
echo "[launcher] submitting DSv4 teacher sbatch..."
DSV4_JID=$(
    sbatch --parsable \
        --export=ALL,RLER=$RLER,DISCOVERY=$DSV4_DISCOVERY,IMAGE_SQSH=$IMAGE_SQSH \
        "$SCRIPT_DIR/sbatch_dsv4_teacher.sbatch"
)
echo "[launcher] dsv4 jobid=$DSV4_JID  discovery=$DSV4_DISCOVERY"

# Submit collector, gated to start AFTER dsv4 is in RUNNING state (afterany
# would wait for completion; the collector polls the discovery file itself
# so we can launch it as soon as dsv4 is queued — but it'd waste node
# allocation polling. afterok-on-start equivalent is `aftercorr` but slurm
# doesn't have it, so use --dependency=after (starts after dsv4 starts).)
COLLECTOR_RLER=${RLER}
COLLECTOR_BASE=${BASE}
COLLECTOR_IMG=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh
echo "[launcher] submitting SFT collector sbatch (depends after:$DSV4_JID)..."
COLLECTOR_JID=$(
    sbatch --parsable \
        --dependency=after:$DSV4_JID \
        --export=ALL,RLER=$COLLECTOR_RLER,BASE=$COLLECTOR_BASE,IMAGE_SQSH=$COLLECTOR_IMG,\
ARTIFACT_ROOT=$ARTIFACT_ROOT,DSV4_DISCOVERY=$DSV4_DISCOVERY,\
INSTANCE_IDS="$INSTANCE_IDS",\
SUBSET=${SUBSET:-rebench_v2},SPLIT=${SPLIT:-train},\
INSTANCE_WORKERS=${INSTANCE_WORKERS:-16},\
SEARCH_M=${SEARCH_M:-8},SEARCH_N=${SEARCH_N:-2},\
LANES_MAX_MID_CPS=${LANES_MAX_MID_CPS:-6},\
LANES_STEPS_PER_ROUND=${LANES_STEPS_PER_ROUND:-20},\
SEARCH_STEP_LIMIT=${SEARCH_STEP_LIMIT:-120},\
LANES_GT_EVAL_WORKERS=${LANES_GT_EVAL_WORKERS:-8},\
RUBRIC_BANK_STRATEGY=${RUBRIC_BANK_STRATEGY:-score} \
        "$SCRIPT_DIR/sbatch_collector_sft_parallel.sbatch"
)
echo "[launcher] collector jobid=$COLLECTOR_JID"
echo "[launcher] tail logs:"
echo "    tail -F /mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/dsv4_teacher-${DSV4_JID}.log"
echo "    tail -F /mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/sft_parallel-${COLLECTOR_JID}.log"
echo "    summary.json: $ARTIFACT_ROOT/summary.json"
