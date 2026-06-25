#!/bin/bash
#SBATCH --job-name=dsv4_teacher
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/dsv4_teacher-%j.log

# DSv4-Pro teacher sglang server (low-latency recipe — the only one validated
# to be stable on our SWE-agent workload; see docs/deepseek_v4_pro_startup.md).
# Writes a discovery file at $DISCOVERY so the collector job can pick up the
# endpoint without a hand-edited socat config.
#
# Required env (passed via --export):
#   DISCOVERY        path to write {jobid, node, port, base_url}
# Optional env:
#   DSV4_PORT        default 8888
#   IMAGE_SQSH       default /mnt/lustre/.../images/sglang_dsv4_b200.sqsh
#                    Must be the pinned `:deepseek-v4-blackwell` image; the
#                    latest tag predates the sgl-project/sglang#25733 fix.
#   SRC_MODEL        default /mnt/lustre/.../checkpoints/sihanzeng/models/DeepSeek-V4-Pro
#   LOCAL_MODEL_ROOT default /mnt/localssd/sihanzeng

set -euo pipefail

: "${DISCOVERY:?DISCOVERY path is required}"

DSV4_PORT=${DSV4_PORT:-8888}
IMAGE_SQSH=${IMAGE_SQSH:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/sglang_dsv4_b200.sqsh}
SRC_MODEL=${SRC_MODEL:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/DeepSeek-V4-Pro}
LOCAL_MODEL_ROOT=${LOCAL_MODEL_ROOT:-/mnt/localssd/sihanzeng}
LOCAL_MODEL=$LOCAL_MODEL_ROOT/DeepSeek-V4-Pro

RUN_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/${SLURM_JOB_ID}-dsv4-teacher
mkdir -p "$RUN_DIR"
mkdir -p "$(dirname "$DISCOVERY")"

NODE=$SLURMD_NODENAME
echo "[$(date)] jobid=$SLURM_JOB_ID node=$NODE port=$DSV4_PORT discovery=$DISCOVERY"

# --- Stage weights Lustre -> local NVMe (parallel xargs -P64, marker cache) ---
# Per docs/deepseek_v4_pro_startup.md: 68 s cold @ 11.8 GB/s vs ~9 min serial.
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
  # Small metadata (configs, tokenizer, etc.) — fast enough serial
  cp -rL --no-target-directory \
      config.json generation_config.json README.md LICENSE \
      assets encoding inference \
      "$DST/" 2>/dev/null || true
  # Catch any *.json that the explicit list missed
  for f in "$SRC"/*.json; do
      [ -f "$f" ] && cp -L "$f" "$DST/" 2>/dev/null || true
  done
  echo OK > "$MARKER"
fi
du -sh "$DST"
'

# --- Write discovery file before server starts (collector polls with health check) ---
cat > "$DISCOVERY" <<EOF
{
  "jobid": $SLURM_JOB_ID,
  "node": "$NODE",
  "port": $DSV4_PORT,
  "base_url": "http://${NODE}:${DSV4_PORT}",
  "model_name": "openai/deepseek-v4-pro",
  "served_model_name": "deepseek-v4-pro",
  "ready": false
}
EOF

# --- Launch DSv4 sglang server (low-latency recipe) ---
EXPORT_ENV="ALL,PORT=$DSV4_PORT,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,\
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0,\
SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1,\
SGLANG_OPT_USE_JIT_NORM=1,\
SGLANG_OPT_USE_JIT_INDEXER_METADATA=1,\
SGLANG_OPT_USE_TOPK_V2=1,\
SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1,\
DSV4_PORT=$DSV4_PORT"

# Optional debug toggles appended to the recipe args via env. Useful for
# isolating the cudaErrorIllegalAddress at #full token=4096:
#   EXTRA_DSV4_ARGS="--page-size 512"        shifts KV-page boundary to 8192
#   EXTRA_DSV4_ARGS="--disable-cuda-graph"   skips cuda-graph replay codepath
EXPORT_ENV="$EXPORT_ENV,EXTRA_DSV4_ARGS=${EXTRA_DSV4_ARGS:-}"

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
    --trust-remote-code --tp 8 \
    "${RECIPE_ARGS[@]}" \
    ${EXTRA_DSV4_ARGS:-} \
    --served-model-name deepseek-v4-pro &
SERVER_PID=$!

# Wait for /v1/models ready, then flip ready=true in discovery
for i in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:'"$DSV4_PORT"'/v1/models" >/dev/null 2>&1; then
        echo "[dsv4-teacher] sglang ready after ${i} attempts"
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
" '"$DISCOVERY"' || echo "[dsv4-teacher] WARN: failed to flip discovery ready=true (server is still serving)"
wait $SERVER_PID
'
