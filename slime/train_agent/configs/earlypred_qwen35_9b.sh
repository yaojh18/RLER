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
RLER_LITELLM_EXPLICIT_RETRIES="${RLER_LITELLM_EXPLICIT_RETRIES:-8}"
SWE_AGENT_LANES_RUBRIC_MODEL="${SWE_AGENT_LANES_RUBRIC_MODEL:-openai/azure/openai/gpt-5.6-luna}"
SWE_AGENT_LANES_JUDGE_MODEL="${SWE_AGENT_LANES_JUDGE_MODEL:-$SWE_AGENT_LANES_RUBRIC_MODEL}"

# Do not serialize or globally pace Lane-C calls.  Rubric generation,
# retrieval, and judging use the same hosted model; exhausted calls invalidate the
# affected GRPO group instead of changing the reward model.
LITELLM_MAX_CONCURRENT_CALLS="${LITELLM_MAX_CONCURRENT_CALLS:-128}"

# Keep the online dashboard focused on learning signal and failure modes.
# Full metric dictionaries remain in the driver log/TensorBoard; this exact
# allowlist only reduces W&B history cardinality.  Method-only suffixes avoid
# constant-zero beam charts for depth1 and irrelevant format/GT charts for the
# judge methods.
if [ -z "${SLIME_WANDB_METRIC_ALLOWLIST:-}" ]; then
  SLIME_WANDB_METRIC_ALLOWLIST="train/step,train/pg_loss,train/entropy_loss,train/grad_norm,train/lr-pg_0,train/global_batch_size,train/pg_clipfrac,train/ppo_kl,train/tis,train/tis_abs,train/tis_clipfrac,train/train_rollout_logprob_abs_diff"
  SLIME_WANDB_METRIC_ALLOWLIST="$SLIME_WANDB_METRIC_ALLOWLIST,rollout/step,rollout/raw_reward,rollout/repetition_frac,rollout/truncated_ratio,perf/rollout_time,perf/actor_train_time,perf/update_weights_time,perf/train_wait_time,perf/wait_time_ratio"
  SLIME_WANDB_METRIC_ALLOWLIST="$SLIME_WANDB_METRIC_ALLOWLIST,eval/step,eval/train_instance_attempt,eval/attempted,eval/completed,eval/coverage,eval/errors,eval/incomplete,eval/resolved,eval/resolved_rate,eval/retries_exhausted,eval/stale_policy_aborted,eval/swebench_validation/truncated_ratio"
  SLIME_WANDB_METRIC_ALLOWLIST="$SLIME_WANDB_METRIC_ALLOWLIST,swe_agent/train_instances_attempted,swe_agent/train_epoch,swe_agent/last_validation_attempt,swe_agent/last_validation_scheduled_attempt,swe_agent/groups_attempted,swe_agent/groups_accepted_rate,swe_agent/groups_invalid_rate,swe_agent/groups_dynamic_filtered_rate,swe_agent/groups_excess_rate,swe_agent/stale_dropped_groups,swe_agent/failed_instances_total,swe_agent/truncated_oversized_samples_total"
  SLIME_WANDB_METRIC_ALLOWLIST="$SLIME_WANDB_METRIC_ALLOWLIST,swe_agent/reward_group_mean,swe_agent/reward_group_variance_mean,swe_agent/zero_variance_group_rate,swe_agent/sample_cont_steps_mean,swe_agent/sample_full_steps_mean,swe_agent/truncation_rate"
  SLIME_WANDB_METRIC_ALLOWLIST="$SLIME_WANDB_METRIC_ALLOWLIST,usage/event_step,usage/total_tokens_cumulative,usage/train_tokens_cumulative,usage/validation_tokens_cumulative,usage/qwen_tokens_cumulative"
  case "$METHOD" in
    baseline)
      SLIME_WANDB_METRIC_ALLOWLIST="$SLIME_WANDB_METRIC_ALLOWLIST,swe_agent/infra_drop_rate,swe_agent/sample_gt_mean,swe_agent/sample_pass_at_0.5,swe_agent/instance_pass_at_0.5,swe_agent/submit_rate,swe_agent/ratio_zero_patch,swe_agent/format_error_rate_mean,swe_agent/ratio_fe_gated_trajectories,swe_agent/ratio_format_error_killed"
      ;;
    judge_depth2_beam2)
      SLIME_WANDB_METRIC_ALLOWLIST="$SLIME_WANDB_METRIC_ALLOWLIST,swe_agent/sample_parent_steps_mean,swe_agent/root_groups,swe_agent/beam_groups,swe_agent/root_groups_invalid_rate,swe_agent/root_groups_dynamic_filtered_rate,swe_agent/beam_groups_invalid_rate,swe_agent/beam_groups_dynamic_filtered_rate,swe_agent/root_reward_group_mean,swe_agent/root_reward_group_variance_mean,swe_agent/root_zero_variance_group_rate,swe_agent/beam_reward_group_mean,swe_agent/beam_reward_group_variance_mean,swe_agent/beam_zero_variance_group_rate"
      ;;
  esac
fi

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
export RLER_LITELLM_EXPLICIT_RETRIES
export LITELLM_MAX_CONCURRENT_CALLS
export SWE_AGENT_LANES_RUBRIC_MODEL SWE_AGENT_LANES_JUDGE_MODEL
export SLIME_WANDB_METRIC_ALLOWLIST
