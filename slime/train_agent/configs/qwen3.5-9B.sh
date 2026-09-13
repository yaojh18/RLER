MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 16
   --num-query-groups 4
   --kv-channels 256
   --num-layers 32
   --hidden-size 4096
   --ffn-hidden-size 12288
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 248320

   --rotary-base 10000000

   # qwen3.5 specific
   --attention-output-gate
)

# Canonical Qwen3.5-9B DPPO training recipe.  Baseline and direct-judge
# experiments share these arrays; their only differences are collector and
# reward arguments supplied by the launcher.
GRPO_COMMON_ARGS=(
  --update-weights-interval 1
  --rollout-shuffle
  --input-key input
  --metadata-key metadata
  --custom-convert-samples-to-train-data-path train_agent.run.grpo.convert_samples_to_train_data
  --loss-mask-type qwen3_5
  --disable-grpo-std-normalization
  --entropy-coef 0
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --kl-coef 0.0
  --n-samples-per-prompt 8
  --advantage-estimator grpo
  --loss-type policy_loss
  --policy-loss-type dppo
  --dppo-divergence-type tv
  --dppo-divergence-threshold 0.1
  --use-rollout-logprobs
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --micro-batch-size 1
  --calculate-per-token-loss
  --clip-grad 1.0
  --lr-warmup-fraction 0.0
)

GRPO_FRESH_START_ARGS=(
  --no-load-optim
  --no-load-rng
  --finetune
  --start-rollout-id 0
)

GRPO_PARALLEL_ARGS=(
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
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.0
  --adam-beta1 0.9
  --adam-beta2 0.999
  --adam-eps 1e-8
)

GRPO_SGLANG_ARGS=(
  --sglang-server-concurrency 128
  --sglang-reasoning-parser qwen3
  --sglang-watchdog-timeout 3600
)

GRPO_MISC_ARGS=(
  --use-dynamic-batch-size
  --balance-data
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)
