SFT_COMMON_ARGS=(
  --save-interval "${SWE_SFT_SAVE_INTERVAL:-1}"
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --input-key messages
  --rollout-shuffle
  --num-epoch "${SWE_SFT_NUM_EPOCH:-1}"
  --micro-batch-size "${SWE_SFT_MICRO_BATCH_SIZE:-1}"
  --loss-type sft_loss
  --loss-mask-type qwen3_5
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)

SFT_POLICY_ARGS=(
  --tensor-model-parallel-size "${SWE_SFT_POLICY_TP:-1}"
  --pipeline-model-parallel-size "${SWE_SFT_POLICY_PP:-1}"
  --context-parallel-size "${SWE_SFT_POLICY_CP:-1}"
  --max-tokens-per-gpu "${SWE_SFT_POLICY_MAX_TOKENS_PER_GPU:-12288}"
)

SFT_RUBRIC_ARGS=(
  --tensor-model-parallel-size "${SWE_SFT_RUBRIC_TP:-1}"
  --pipeline-model-parallel-size "${SWE_SFT_RUBRIC_PP:-1}"
  --context-parallel-size "${SWE_SFT_RUBRIC_CP:-1}"
  --max-tokens-per-gpu "${SWE_SFT_RUBRIC_MAX_TOKENS_PER_GPU:-32768}"
)

SFT_PARALLEL_ARGS=(
  --sequence-parallel
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
)

SFT_RECOMPUTE_ARGS=(
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
)

SFT_OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${SWE_SFT_LR:-1e-5}"
  --lr-decay-style constant
  --min-lr 1e-6
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
)

SFT_MISC_ARGS=(
  --use-dynamic-batch-size
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)
