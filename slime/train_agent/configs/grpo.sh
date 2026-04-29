GRPO_COMMON_ARGS=(
  --save-interval 1
  --no-load-optim
  --no-load-rng
  --finetune
  --start-rollout-id 0
  --update-weights-interval 1
  --rollout-function-path train_agent.collect_grpo_rollout.generate_rollout
  --input-key input
  --metadata-key metadata
  --n-samples-per-prompt 1
  --num-rollout 1
  --custom-convert-samples-to-train-data-path train_agent.run.grpo.convert_samples_to_train_data
  --loss-mask-type qwen3_5
  --advantage-estimator grpo
  --eps-clip 0.2
  --entropy-coef 0.0
  --calculate-per-token-loss
  --use-dynamic-global-batch-size
  --micro-batch-size 1
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
  --sglang-disable-radix-cache
  --sglang-watchdog-timeout 3600
)

GRPO_MISC_ARGS=(
  --use-dynamic-batch-size
  --max-tokens-per-gpu 32768
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)
