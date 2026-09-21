# Qwen3.5-9B early-prediction RL reproduction

This directory contains only the final three-fold Qwen RL configuration and
the launch/evaluation code needed to run it. It does not contain fold-search,
checkpoint-selection, data-order-search, or experiment-audit programs.

## Methods

| Method | rollout and terminal GT | actor loss horizon | reward | train cohort |
| --- | --- | --- | --- | --- |
| `baseline1` | full | full | terminal joint GT | all 250 train instances |
| `baseline2` | full | first 40 assistant turns | terminal joint GT | all 250 train instances |
| `direct` | full | first 40 assistant turns | golden-rubric Direct judge | eligible train instances |

All methods use `Qwen/Qwen3.5-9B`, a 65,536-token training context, 20,480
completion tokens, 16 rollout groups per optimizer batch, eight samples per
group, Binary-TV threshold 0.1, learning rate 1e-6, and stale lag 1. A stale
source is resampled without consuming the instance-attempt denominator.
Validation is deferred and uses four samples per task, 128K context, 10,240
completion tokens, temperature 0.7, top-p 0.95, fallback patch extraction, and
a 600-second wall-clock limit around the complete evaluator worker.

`final_fold_config.json` is the only split definition. It contains the final
ordered train/validation/test instance IDs for folds 0, 1, and 2. The
materializer reads the shared 500-instance source population and writes the
baseline and Direct-eligible datasets used by the launcher.

## Training

```bash
sbatch --export=ALL,FOLD=0,EXPERIMENT_METHOD=direct \
  slime/train_agent/scripts/qwen35_earlypred_train.slurm
```

Choose `FOLD=0|1|2` and
`EXPERIMENT_METHOD=baseline1|baseline2|direct`. A base-checkpoint run consumes
384 instance attempts and emits the 128/256/384 evaluation checkpoints. The
launcher resumes from its latest saved model/data state, exports evaluation
checkpoints, and requeues on the Slurm pre-timeout signal.

For a Direct 384-to-512 continuation, the final ordered 128-attempt inputs and
the rubric bank used by each fold are under
`frozen_inputs/direct_384_to_512/`. Point the start variables at the desired
attempt-384 HF checkpoint and its torch-distributed conversion:

```bash
sbatch --export=ALL,FOLD=0,EXPERIMENT_METHOD=direct,\
DIRECT_CONTINUATION_384_TO_512=1,\
CONTINUATION_START_HF=/workspace/path/to/attempt384/hf,\
CONTINUATION_START_LOAD_DIR=/workspace/path/to/attempt384/torch_dist_release \
  slime/train_agent/scripts/qwen35_earlypred_train.slurm
```

For the Fold0 rollout-cutoff ablation, the following variables make the
training rollout physically stop at the Direct loss horizon and omit the
training-only terminal GT diagnostic. `TRAIN_INSTANCE_BUDGET_OVERRIDE=512`
and `EVAL_INSTANCE_INTERVAL_OVERRIDE=512` train two epochs and mark only the
final checkpoint for deferred validation:

```bash
sbatch --export=ALL,FOLD=0,EXPERIMENT_METHOD=direct,\
DIRECT_ROLLOUT_CUTOFF=10,DIRECT_HARD_ROLLOUT_CUTOFF=1,\
DIRECT_SKIP_GT_EVALUATION=1,TRAIN_INSTANCE_BUDGET_OVERRIDE=512,\
EVAL_INSTANCE_INTERVAL_OVERRIDE=512 \
  slime/train_agent/scripts/qwen35_earlypred_train.slurm
```

Use `DIRECT_ROLLOUT_CUTOFF=10|20|30`. The Direct judge remains the sole
optimizer reward. These flags are rejected for non-Direct methods; without
them the three primary methods retain the contracts in the table above.

## Evaluation

Create a JSON file containing `{"tasks": [...]}`. Each task has `label`,
`checkpoint`, `fold`, and `split` (`validation` or `test`), then submit:

```bash
sbatch --export=ALL,EVAL_TASKS_CONFIG=path/relative/to/workspace/tasks.json \
  slime/train_agent/scripts/qwen35_earlypred_eval.slurm
```

Each allocated node evaluates one checkpoint at a time with four TP2 SGLang
replicas and no Megatron actor. Interrupted rollout artifacts are reused after
requeue; model outcomes are not retried as infrastructure failures.

External prerequisites are the Qwen3.5-9B HF and torch-distributed model
checkpoints, the training/evaluation container images, the per-instance
SWE-bench SIFs, the shared 500-instance source dataset, and the usual runtime
credentials.
