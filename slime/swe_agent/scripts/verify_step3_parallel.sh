#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

: "${SEARCH_SWE_SLIME_API_BASE:=http://127.0.0.1:8032}"
: "${SEARCH_SWE_SLIME_API_KEY:=EMPTY}"
: "${SLIME_MODEL:=Qwen/Qwen3.5-9B}"
: "${INSTANCE_ID:=psf__requests-1766}"
: "${RUN_TAG:=$(date -u +%Y%m%d-%H%M%S)}"

ARTIFACT_ROOT="$ROOT_DIR/slime/swe_agent/artifacts/step3_parallel_verify/$RUN_TAG"
mkdir -p "$ARTIFACT_ROOT"

run_mode() {
  local mode="$1"
  local output_root="$ARTIFACT_ROOT/${mode}_outputs"
  local summary_path="$ARTIFACT_ROOT/${mode}_summary.json"
  echo "[stage3] running ${mode} for ${INSTANCE_ID}"
  SEARCH_SWE_SLIME_API_BASE="$SEARCH_SWE_SLIME_API_BASE" \
  SEARCH_SWE_SLIME_API_KEY="$SEARCH_SWE_SLIME_API_KEY" \
  agent/.venv/bin/python slime/swe_agent/scripts/run_stage3_timed_verify.py \
    --mode "$mode" \
    --instance-id "$INSTANCE_ID" \
    --output-root "$output_root" \
    --summary-path "$summary_path" \
    --slime-model "$SLIME_MODEL"
}

run_mode baseline
run_mode parallel_sync
run_mode parallel_async

echo "[stage3] artifacts: $ARTIFACT_ROOT"
