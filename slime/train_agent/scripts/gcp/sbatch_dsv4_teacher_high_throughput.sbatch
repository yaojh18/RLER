#!/bin/bash
#SBATCH --job-name=dsv4_teacher_ht
#SBATCH --account=ads_cpt
#SBATCH --qos=ads_cpt
#SBATCH --partition=a4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/dsv4_teacher_ht-%j.log

# High-throughput DSv4 variant: DP=8 + DP-attention + mega-MoE + RADIX CACHE
# ENABLED. The other (low-latency) sbatch disables radix cache and runs
# TP=8 only; this one is the throughput-maxing recipe from the original
# docs (≈600 tput/gpu @ CONC=64).
#
# CAVEAT: the pinned sglang_dsv4_b200.sqsh image hit a deepgemm CUDA_ERROR_ASSERT
# on this recipe under variable batch shapes earlier (see
# project_dsv4_blackwell_image_buggy memory). Trying again 2026-05-21
# with radix cache ON (which may also incidentally avoid the failing
# fp8 codepath via prefix-cache fast paths). If it crashes within
# ~30min of serving, fall back to sbatch_dsv4_teacher.sbatch.
#
# Required env (passed via --export):
#   DISCOVERY        path to write {jobid, node, port, base_url}
# Optional env:
#   DSV4_PORT        default 8888
#   IMAGE_SQSH       default /mnt/lustre/.../images/sglang_dsv4_b200.sqsh
#   SRC_MODEL        default /mnt/lustre/.../checkpoints/sihanzeng/models/DeepSeek-V4-Pro
#   LOCAL_MODEL_ROOT default /mnt/localssd/sihanzeng

set -euo pipefail

: "${DISCOVERY:?DISCOVERY path is required}"

DSV4_PORT=${DSV4_PORT:-8888}
IMAGE_SQSH=${IMAGE_SQSH:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/sglang_dsv4_b200.sqsh}
SRC_MODEL=${SRC_MODEL:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/DeepSeek-V4-Pro}
LOCAL_MODEL_ROOT=${LOCAL_MODEL_ROOT:-/mnt/localssd/sihanzeng}
LOCAL_MODEL=$LOCAL_MODEL_ROOT/DeepSeek-V4-Pro

RUN_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/${SLURM_JOB_ID}-dsv4-teacher-ht
mkdir -p "$RUN_DIR"
mkdir -p "$(dirname "$DISCOVERY")"

NODE=$SLURMD_NODENAME
echo "[$(date)] HT-variant jobid=$SLURM_JOB_ID node=$NODE port=$DSV4_PORT discovery=$DISCOVERY"

# --- Stage weights (parallel xargs -P64) ---
srun --jobid="$SLURM_JOB_ID" --overlap --ntasks=1 bash -c '
set -e
SRC='"$SRC_MODEL"'
DST='"$LOCAL_MODEL"'
mkdir -p '"$LOCAL_MODEL_ROOT"'
MARKER=$DST/.staged_ok
if [ -f "$MARKER" ] && [ "$(stat -c %s "$MARKER")" -gt 0 ]; then
  echo "weights cached, skipping"
else
  echo "[$(date)] staging weights $SRC -> $DST (parallel xargs -P64)"
  rm -rf "$DST"; mkdir -p "$DST"
  cd "$SRC"
  time (ls *.safetensors | xargs -P64 -I{} cp -L {} "$DST/")
  cp -rL --no-target-directory \
      config.json generation_config.json README.md LICENSE \
      assets encoding inference \
      "$DST/" 2>/dev/null || true
  for f in "$SRC"/*.json; do
      [ -f "$f" ] && cp -L "$f" "$DST/" 2>/dev/null || true
  done
  echo OK > "$MARKER"
fi
du -sh "$DST"
'

# --- Discovery file ---
cat > "$DISCOVERY" <<EOF
{
  "jobid": $SLURM_JOB_ID,
  "node": "$NODE",
  "port": $DSV4_PORT,
  "base_url": "http://${NODE}:${DSV4_PORT}",
  "model_name": "openai/deepseek-v4-pro",
  "served_model_name": "deepseek-v4-pro",
  "recipe": "dp_attention_megamoe_radix_on",
  "ready": false
}
EOF

# --- Launch DSv4 sglang server (DP-attention + mega-MoE + RADIX ON) ---
# COMMON_ENV is the same baseline as low_latency; MEGAMOE_ENV are the
# extra knobs that the dp_attention + deepep path needs to keep the
# deepgemm kernel inside its supported input shapes.
COMMON_ENV="PORT=$DSV4_PORT,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,\
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0,\
SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1,\
SGLANG_OPT_USE_JIT_NORM=1,\
SGLANG_OPT_USE_JIT_INDEXER_METADATA=1,\
SGLANG_OPT_USE_TOPK_V2=1,\
SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1"
MEGAMOE_ENV="SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1,\
SGLANG_OPT_FIX_HASH_MEGA_MOE=1,\
SGLANG_OPT_USE_FAST_MASK_EP=1,\
SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1,\
SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096,\
SGLANG_OPT_FIX_NEXTN_MEGA_MOE=1,\
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=0"
EXPORT_ENV="ALL,$COMMON_ENV,$MEGAMOE_ENV,DSV4_PORT=$DSV4_PORT"

exec srun --container-image="$IMAGE_SQSH" \
     --container-mounts="/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,$LOCAL_MODEL:/models/DeepSeek-V4-Pro,$RUN_DIR:/workspace/run" \
     --container-workdir=/workspace/run \
     --container-writable \
     --no-container-mount-home \
     --export="$EXPORT_ENV" \
     bash -lc '
set -e
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server \
    --model-path /models/DeepSeek-V4-Pro \
    --host 0.0.0.0 --port "$DSV4_PORT" \
    --trust-remote-code --tp 8 --dp-size 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-config "{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}" \
    --chunked-prefill-size 32768 \
    --max-running-requests 192 \
    --mem-fraction-static 0.90 \
    --swa-full-tokens-ratio 0.1 \
    --served-model-name deepseek-v4-pro &
SERVER_PID=$!

# Wait for /v1/models ready, then flip ready=true in discovery
for i in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:'"$DSV4_PORT"'/v1/models" >/dev/null 2>&1; then
        echo "[dsv4-teacher-ht] sglang ready after ${i} attempts"
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
" '"$DISCOVERY"' || echo "[dsv4-teacher-ht] WARN: failed to flip discovery ready=true"
wait $SERVER_PID
'
