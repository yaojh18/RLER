#!/bin/bash
#SBATCH --job-name=qwen35_student
#SBATCH --account=ads_cpt
#SBATCH --qos=ads_cpt
#SBATCH --partition=a4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/qwen35_student-%j.log

# Qwen3.5-9B student sglang server, single B200 node. Uses the same baked
# slime image that the collector/SFT job uses so we don't drag a second
# 41 GB sqsh onto the cache. Writes a discovery file at $DISCOVERY.
#
# Required env (passed via --export):
#   DISCOVERY        path to write {jobid, node, port, base_url}
# Optional env:
#   STUDENT_PORT     default 8032
#   STUDENT_TP       default 4
#   STUDENT_MODEL    served-model-name, default Qwen/Qwen3.5-9B
#   IMAGE_SQSH       default slime baked image
#   STUDENT_MODEL_DIR default /mnt/lustre/.../Qwen3.5-9B

set -euo pipefail

: "${DISCOVERY:?DISCOVERY path is required}"

STUDENT_PORT=${STUDENT_PORT:-8032}
STUDENT_TP=${STUDENT_TP:-4}
STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3.5-9B}
IMAGE_SQSH=${IMAGE_SQSH:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh}
STUDENT_MODEL_DIR=${STUDENT_MODEL_DIR:-/mnt/lustre/metavmds0lstre/teams/mrs_ranking_core_modeling/daili1/deepeyes/models/Qwen3.5-9B}

RUN_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/${SLURM_JOB_ID}-qwen35-student
mkdir -p "$RUN_DIR"
mkdir -p "$(dirname "$DISCOVERY")"

NODE=$SLURMD_NODENAME
echo "[$(date)] jobid=$SLURM_JOB_ID node=$NODE port=$STUDENT_PORT discovery=$DISCOVERY tp=$STUDENT_TP"

cat > "$DISCOVERY" <<EOF
{
  "jobid": $SLURM_JOB_ID,
  "node": "$NODE",
  "port": $STUDENT_PORT,
  "base_url": "http://${NODE}:${STUDENT_PORT}",
  "served_model_name": "$STUDENT_MODEL",
  "tensor_parallel_size": $STUDENT_TP,
  "ready": false
}
EOF

CACHE_BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/.cache/qwen35-${SLURM_JOB_ID}
mkdir -p "$CACHE_BASE"

exec srun --container-image="$IMAGE_SQSH" \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,$RUN_DIR:/workspace/run \
     --container-workdir=/workspace/run \
     --container-writable \
     --no-container-mount-home \
     --export=ALL,STUDENT_PORT=$STUDENT_PORT,STUDENT_TP=$STUDENT_TP,\
STUDENT_MODEL=$STUDENT_MODEL,STUDENT_MODEL_DIR=$STUDENT_MODEL_DIR,\
DISCOVERY=$DISCOVERY,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,\
FLASHINFER_USE_CUDA_NORM=1,\
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1,SGLANG_DISABLE_CUDNN_CHECK=1,\
HOME=$CACHE_BASE,XDG_CACHE_HOME=$CACHE_BASE,HF_HOME=$CACHE_BASE/hf,\
FLASHINFER_WORKSPACE_DIR=$CACHE_BASE/flashinfer,TRITON_CACHE_DIR=$CACHE_BASE/triton \
     bash -lc '
set -e
mkdir -p "$HF_HOME" "$FLASHINFER_WORKSPACE_DIR" "$TRITON_CACHE_DIR"
python3 -m sglang.launch_server \
    --model-path "$STUDENT_MODEL_DIR" \
    --host 0.0.0.0 --port "$STUDENT_PORT" \
    --tensor-parallel-size "$STUDENT_TP" \
    --context-length 128000 \
    --served-model-name "$STUDENT_MODEL" \
    --reasoning-parser qwen3 \
    --disable-radix-cache \
    --watchdog-timeout 3600 &
SERVER_PID=$!

for i in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${STUDENT_PORT}/v1/models" >/dev/null 2>&1; then
        echo "[qwen35-student] sglang ready after ${i} attempts"
        break
    fi
    sleep 5
done
python3 -c "
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d[\"ready\"] = True
open(p, \"w\").write(json.dumps(d, indent=2))
" "$DISCOVERY" || echo "[qwen35-student] WARN: failed to flip discovery ready=true (server is still serving)"
wait $SERVER_PID
'
