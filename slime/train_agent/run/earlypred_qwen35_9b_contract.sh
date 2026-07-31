# Canonical fold-0 Qwen3.5-9B early-prediction training contract.
#
# This file is sourced by the Slurm launcher.  Keep experiment semantics here
# so they are part of the auditable RLER git diff; the external launcher owns
# only cluster/container placement and lifecycle.

case "${METHOD:-}" in
  baseline|judge_depth1|judge_depth2_beam2)
    ;;
  *)
    echo "ERROR: METHOD must be baseline|judge_depth1|judge_depth2_beam2" >&2
    return 2
    ;;
esac

EVAL_INSTANCE_INTERVAL="${EVAL_INSTANCE_INTERVAL:-128}"
TRAIN_INSTANCE_BUDGET="${TRAIN_INSTANCE_BUDGET:-1250}"
EPOCH_INSTANCE_COUNT="${EPOCH_INSTANCE_COUNT:-250}"
VALIDATION_INSTANCE_COUNT="${VALIDATION_INSTANCE_COUNT:-50}"

# Preserve the previously successful 9B optimization batch and topology.
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MODEL_CONTEXT_LENGTH="${MODEL_CONTEXT_LENGTH:-65536}"
SMOKE_MAX_TOKENS_PER_GPU="${SMOKE_MAX_TOKENS_PER_GPU:-16384}"

# Validation shares the rollout endpoints and CPU node. Fifty policy sessions
# approximately match an 8-instance x 8-sibling baseline train batch, while a
# bounded process pool avoids fifty duplicate Python/runtime images.
VALIDATION_INSTANCE_WORKERS="${VALIDATION_INSTANCE_WORKERS:-50}"
VALIDATION_PROCESS_WORKERS="${VALIDATION_PROCESS_WORKERS:-8}"
VALIDATION_TEMPERATURE="${VALIDATION_TEMPERATURE:-0.2}"
VALIDATION_TOP_P="${VALIDATION_TOP_P:-0.95}"
RLER_VALIDATION_MAX_ATTEMPTS="${RLER_VALIDATION_MAX_ATTEMPTS:-2}"

# Checkpoints are recovery points, not changes to the optimizer recipe.
CHECKPOINT_SAVE_INTERVAL="${CHECKPOINT_SAVE_INTERVAL:-4}"
CHECKPOINT_RETAIN_LATEST="${CHECKPOINT_RETAIN_LATEST:-2}"

# Hosted Lane-C models are independent of the local 64K Qwen policy service.
RLER_HOSTED_MODEL_CONTEXT_LENGTH="${RLER_HOSTED_MODEL_CONTEXT_LENGTH:-128000}"
RLER_HOSTED_MAX_COMPLETION_TOKENS="${RLER_HOSTED_MAX_COMPLETION_TOKENS:-20480}"
RLER_LITELLM_RATE_LIMIT_FALLBACK_MODEL="${RLER_LITELLM_RATE_LIMIT_FALLBACK_MODEL:-nvidia/nvidia/nemotron-3-ultra}"

# Do not serialize or globally pace Lane-C calls.  A GLM 429 switches that
# logical judge call immediately to Ultra; it does not cool down the run.
LITELLM_MAX_CONCURRENT_CALLS="${LITELLM_MAX_CONCURRENT_CALLS:-128}"

export TRAIN_INSTANCE_BUDGET EPOCH_INSTANCE_COUNT
export VALIDATION_INSTANCE_COUNT EVAL_INSTANCE_INTERVAL
export ROLLOUT_BATCH_SIZE GLOBAL_BATCH_SIZE
export MODEL_CONTEXT_LENGTH SMOKE_MAX_TOKENS_PER_GPU
export VALIDATION_INSTANCE_WORKERS VALIDATION_PROCESS_WORKERS
export VALIDATION_TEMPERATURE VALIDATION_TOP_P
export RLER_VALIDATION_MAX_ATTEMPTS
export CHECKPOINT_SAVE_INTERVAL CHECKPOINT_RETAIN_LATEST
export RLER_HOSTED_MODEL_CONTEXT_LENGTH
export RLER_HOSTED_MAX_COMPLETION_TOKENS
export RLER_LITELLM_RATE_LIMIT_FALLBACK_MODEL
export LITELLM_MAX_CONCURRENT_CALLS
