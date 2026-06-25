GRPO_COMMON_ARGS=(
  --save-interval 10
  --no-load-optim
  --no-load-rng
  --finetune
  --start-rollout-id 0
  --update-weights-interval 1
  --rollout-function-path train_agent.collect_grpo_rollout.generate_rollout
  --rollout-shuffle
  --input-key input
  --metadata-key metadata
  --n-samples-per-prompt 8
  --num-rollout 1
  --custom-convert-samples-to-train-data-path train_agent.run.grpo.convert_samples_to_train_data
  --loss-mask-type qwen3_5
  --advantage-estimator grpo
  --eps-clip 0.2
  --entropy-coef 0
  --use-kl-loss
  --kl-loss-coef 0.01
  --kl-loss-type k3
  --calculate-per-token-loss
  --log-probs-chunk-size 256
  --use-dynamic-global-batch-size
  --micro-batch-size 1
)

GRPO_ROLLOUT_ARGS=(
  --rollout-num-gpus-per-engine 1
)

GRPO_POLICY_ARGS=(
  --loss-type policy_loss
)

GRPO_RUBRIC_ARGS=(
  --loss-type custom_loss
  --custom-loss-function-path train_agent.run.grpo.compute_rubric_loss
  --swe-rtt-alpha 1.0
  --swe-rtt-beta 1.0
)

GRPO_PARALLEL_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 2
  --sequence-parallel
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
)

GRPO_RECOMPUTE_ARGS=(
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
)

GRPO_OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --min-lr 1e-6
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
)

GRPO_SGLANG_ARGS=(
  --sglang-context-length 80960
  --sglang-reasoning-parser qwen3
  # Radix cache enabled — slime calls engine.flush_cache.remote() inside
  # update_weight_from_distributed.py on every policy weight update, so
  # stale KV from old weights gets invalidated cleanly.
  # Without radix cache, agent multi-turn sessions on the same instance
  # re-prefill the entire history every turn (huge waste at ~80K context).
  --sglang-watchdog-timeout 3600
)

GRPO_MISC_ARGS=(
  --use-dynamic-batch-size
  # Per-sample cap = max_tokens_per_gpu * cp_size. We need this >= sglang's
  # context_length (80960) since the token-in/token-out path's full sequence
  # = last_assistant.prompt_token_ids + last_assistant.token_ids, bounded by
  # sglang's own input+output cap. 32768 (= 131072/sample) OOM'd on backward
  # under CP=4 packing (commit 47d170e); 16384 (= 65536/sample) was safe but
  # too small for healthy long rollouts. 22528 gives ~90k per-sample cap —
  # comfortably above sglang's 80960 ceiling, well below the OOM zone.
  --max-tokens-per-gpu 22528
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)
