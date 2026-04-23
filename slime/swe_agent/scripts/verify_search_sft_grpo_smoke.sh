#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/agent/.venv/bin/python}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" slime/swe_agent/scripts/run_search_sft_grpo_smoke.py "$@"
