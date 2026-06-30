# Two-node launchers intentionally override these for validation. Before a
# formal run, use: num_rollout=3000, actor_gpus=64, rollout_gpus=64,
# TP/PP/CP=4/2/4, and rollout_num_gpus_per_engine=4.
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
  --custom-convert-samples-to-train-data-path train_agent.run.grpo.convert_samples_to_train_data
  --loss-mask-type qwen3_5
  --advantage-estimator gspo
  --disable-grpo-std-normalization
  --eps-clip 3e-4
  --eps-clip-high 4e-4
  --entropy-coef 0
  --log-probs-chunk-size 256
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
  --lr 3e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

GRPO_SGLANG_ARGS=(
  --rollout-max-context-len 131072
  --rollout-max-response-len 16384
  --sglang-context-length 131072
  --sglang-mem-fraction-static 0.85
  --sglang-server-concurrency 128
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
  --balance-data
  --max-tokens-per-gpu 32768
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)
