#!/bin/bash
# K-shard SFT collector launcher — Plan C: 1 collector process per server.
#
# Architecture:
#   K × DSv4 teacher (sbatch_dsv4_teacher_high_throughput.sbatch)
#   K × CPU-only collector (sbatch_collector_sft_parallel_cpuonly.sbatch)
#                                each pinned to ONE teacher endpoint
#                                each handling 1/K of the instance list
#
# Why: avoids single-process multi-endpoint complexity (no contextvar
# override, no global-route race). Each collector sees one endpoint, so
# the runner's global rubric routes work cleanly.
#
# Required env:
#   ARTIFACT_ROOT    output dir on Lustre
#   INSTANCE_IDS     space-separated list of swe-rebench instance IDs
#   N_SHARDS         number of (teacher, collector) pairs to launch
# Optional env (passed through to each collector):
#   INSTANCE_WORKERS_PER_SHARD  default 8 (per shard)
#   SEARCH_M, SEARCH_N, LANES_MAX_MID_CPS, LANES_STEPS_PER_ROUND,
#   SEARCH_STEP_LIMIT, LANES_GT_EVAL_WORKERS, RUBRIC_BANK_STRATEGY
#   REUSE_EXISTING_TEACHERS  CSV of existing dsv4 base_urls to reuse
#                            (1 url per shard). If set, skips teacher
#                            sbatch and uses these instead.

set -euo pipefail
: "${ARTIFACT_ROOT:?ARTIFACT_ROOT is required}"
: "${INSTANCE_IDS:?INSTANCE_IDS is required (space-separated)}"
: "${N_SHARDS:?N_SHARDS is required (e.g. 4)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLER=${RLER:-"$(cd "$SCRIPT_DIR/../../../.." && pwd)"}
BASE=${BASE:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench}
DSV4_IMAGE_SQSH=${DSV4_IMAGE_SQSH:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/sglang_dsv4_b200.sqsh}
COLLECTOR_IMAGE_SQSH=${COLLECTOR_IMAGE_SQSH:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh}

INSTANCE_WORKERS_PER_SHARD=${INSTANCE_WORKERS_PER_SHARD:-8}
SEARCH_M=${SEARCH_M:-8}
SEARCH_N=${SEARCH_N:-2}
LANES_MAX_MID_CPS=${LANES_MAX_MID_CPS:-6}
LANES_STEPS_PER_ROUND=${LANES_STEPS_PER_ROUND:-20}
SEARCH_STEP_LIMIT=${SEARCH_STEP_LIMIT:-120}
LANES_GT_EVAL_WORKERS=${LANES_GT_EVAL_WORKERS:-8}
RUBRIC_BANK_STRATEGY=${RUBRIC_BANK_STRATEGY:-score}

mkdir -p "$ARTIFACT_ROOT"

# Shard the instance list round-robin (NOT by hash) — round-robin gives
# perfect by-count balance. For walltime balance, sort instances by
# predicted duration (proxy: patch length) before passing in; this
# launcher just round-robins what it gets.
read -ra INST_ARR <<< "$INSTANCE_IDS"
N_INST=${#INST_ARR[@]}
echo "[launcher] $N_INST instances / $N_SHARDS shards = ~$((N_INST / N_SHARDS)) per shard"

declare -a SHARD_INSTANCES
for ((i=0; i<N_SHARDS; i++)); do SHARD_INSTANCES[$i]=""; done
for ((i=0; i<N_INST; i++)); do
    shard=$((i % N_SHARDS))
    SHARD_INSTANCES[$shard]="${SHARD_INSTANCES[$shard]} ${INST_ARR[$i]}"
done

# Parse REUSE_EXISTING_TEACHERS if provided
declare -a REUSE_URLS=()
if [ -n "${REUSE_EXISTING_TEACHERS:-}" ]; then
    IFS=',' read -ra REUSE_URLS <<< "$REUSE_EXISTING_TEACHERS"
    if [ ${#REUSE_URLS[@]} -ne $N_SHARDS ]; then
        echo "ERROR: REUSE_EXISTING_TEACHERS has ${#REUSE_URLS[@]} URLs but N_SHARDS=$N_SHARDS"
        exit 2
    fi
    echo "[launcher] reusing ${#REUSE_URLS[@]} existing teachers (skip teacher sbatch)"
fi

declare -a DSV4_JIDS=()
declare -a COLLECTOR_JIDS=()
declare -a DISCOVERY_PATHS=()

for ((shard=0; shard<N_SHARDS; shard++)); do
    SHARD_DIR="$ARTIFACT_ROOT/shard_$(printf '%02d' $shard)"
    mkdir -p "$SHARD_DIR/discovery"
    DISCOVERY="$SHARD_DIR/discovery/dsv4.json"
    DISCOVERY_PATHS[$shard]=$DISCOVERY

    SHARD_INST="${SHARD_INSTANCES[$shard]}"
    SHARD_INST_TRIMMED=$(echo "$SHARD_INST" | xargs)  # trim whitespace
    N_SHARD=$(echo "$SHARD_INST_TRIMMED" | wc -w)
    echo
    echo "[launcher] === SHARD $shard ($N_SHARD instances) ==="
    echo "  artifact: $SHARD_DIR"

    DSV4_DEP=""
    if [ ${#REUSE_URLS[@]} -gt 0 ]; then
        # Reuse existing teacher: write a discovery file pointing at it
        URL="${REUSE_URLS[$shard]}"
        NODE=$(echo "$URL" | sed -E 's,^https?://([^:/]+).*,\1,')
        PORT=$(echo "$URL" | sed -E 's,^https?://[^:]+:([0-9]+).*,\1,')
        cat > "$DISCOVERY" <<EOF
{
  "node": "$NODE", "port": $PORT, "base_url": "$URL",
  "model_name": "openai/deepseek-v4-pro",
  "served_model_name": "deepseek-v4-pro",
  "recipe": "reused", "ready": true
}
EOF
        echo "  reusing teacher endpoint: $URL"
    else
        # Submit dedicated teacher for this shard
        DSV4_JID=$(sbatch --parsable \
            --export=ALL,RLER=$RLER,DISCOVERY=$DISCOVERY,IMAGE_SQSH=$DSV4_IMAGE_SQSH \
            "$SCRIPT_DIR/sbatch_dsv4_teacher_high_throughput.sbatch")
        DSV4_JIDS[$shard]=$DSV4_JID
        DSV4_DEP="--dependency=after:$DSV4_JID"
        echo "  dsv4_ht jobid=$DSV4_JID  discovery=$DISCOVERY"
    fi

    # Collector waits for DSv4 readiness (sbatch polls discovery file)
    COLLECTOR_JID=$(sbatch --parsable $DSV4_DEP \
        --export=ALL,RLER=$RLER,IMAGE_SQSH=$COLLECTOR_IMAGE_SQSH,\
ARTIFACT_ROOT=$SHARD_DIR,DSV4_DISCOVERY=$DISCOVERY,\
INSTANCE_IDS="$SHARD_INST_TRIMMED",\
SUBSET=${SUBSET:-rebench_v2},SPLIT=${SPLIT:-train},\
INSTANCE_WORKERS=$INSTANCE_WORKERS_PER_SHARD,\
SEARCH_M=$SEARCH_M,SEARCH_N=$SEARCH_N,\
LANES_MAX_MID_CPS=$LANES_MAX_MID_CPS,\
LANES_STEPS_PER_ROUND=$LANES_STEPS_PER_ROUND,\
SEARCH_STEP_LIMIT=$SEARCH_STEP_LIMIT,\
LANES_GT_EVAL_WORKERS=$LANES_GT_EVAL_WORKERS,\
RUBRIC_BANK_STRATEGY=$RUBRIC_BANK_STRATEGY \
        "$SCRIPT_DIR/sbatch_collector_sft_parallel_cpuonly.sbatch")
    COLLECTOR_JIDS[$shard]=$COLLECTOR_JID
    echo "  collector jobid=$COLLECTOR_JID  workers=$INSTANCE_WORKERS_PER_SHARD"
done

echo
echo "[launcher] SUMMARY"
for ((shard=0; shard<N_SHARDS; shard++)); do
    echo "  shard $shard: dsv4=${DSV4_JIDS[$shard]:-reused} collector=${COLLECTOR_JIDS[$shard]} discovery=${DISCOVERY_PATHS[$shard]}"
done
echo
echo "  monitor totals: python3 $RLER/slime/train_agent/scripts/gcp/sft_full_breakdown.py"
echo "  per-shard:      ls $ARTIFACT_ROOT/shard_*/summary.json"
