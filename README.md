<div align="center">

# Before the Rollout Ends

### Early Terminal Reward Prediction through Contextual Rubric for Long-horizon Coding Agents

</div>

Long-horizon coding agents normally receive verifiable feedback only after an expensive sequence of tool calls. **Contextual Rubric-guided Early Reward (CRER)** instead evaluates the behavioral evidence already visible in a trajectory prefix with task- and stage-specific rubrics. The same interface is used to guide test-time search and to provide dense, verifier-free rewards for reinforcement learning.

This repository contains the code and frozen artifacts for the CRER experiments on SWE-bench Verified. The test-time scaling (TTS) experiments cover Qwen and Nemotron policies. The reinforcement-learning experiments cover Qwen3.5-9B only and compare CRER with controlled TMax and TMax-40 baselines.

---

## Overview

This repository contains four main components:

- **[`agent/`](agent/)**: the mini-SWE-agent-based rollout, TTS algorithm, and shared RL runtime interfaces.

- **[`experience/`](experience/)**: the final Qwen and Nemotron TTS experience banks and the automatic experience generation, refinement, keyword extraction, jand filtering pipeline.

- **[`rubric/`](rubric/)**: the Qwen RL teacher-generated rubric bank and the automatic rubric generation, refinement, judge filter, robust weight optimization, and bank materialization pipeline.

- **[`slime/`](slime/)**: the RL training framework and the final Qwen3.5-9B three-fold training/evaluation launchers. The frozen split definition and reproduction scripts live in [`slime/train_agent/scripts/`](slime/train_agent/scripts/).

---

## Agent setup

Install the maintained SWE-agent package from `agent/` in a Python 3.10 or newer environment:

```bash
cd agent
uv pip install -e .
```

The primary rollout entry points are:

- `swe_agent/run/run_swe_agent.py` for ordinary rollouts and evaluation;
- `swe_agent/run/search_swe_agent.py` for trajectory search;
- `swe_agent/run/aggregate_swe_agent.py` for aggregation over saved
  trajectories.

See [`agent/README.md`](agent/README.md) for the maintained package boundary and runtime interfaces.

---

## Pipelines and training

### Test-Time Scaling (TTS)

CRER retrieves judging experience distilled from related historical tasks, synthesizes rubrics for the current task and trajectory stage, and uses the resulting scores to allocate the remaining rollout budget. The released Qwen and Nemotron experience banks are under [`experience/`](experience/).

The automatic bank-construction pipeline starts from collected rollouts with terminal ground-truth calculation enabled. It generates and refines experience for historical failure groups, extracts both model-summarized and query-overlap keywords, replays the provisional bank, and jointly filters experiences and keywords using the replay judgments:

```bash
PYTHONPATH=experience/pipeline python -m experience_gen.cli prepare-contexts \
  --plan /path/to/plan.json \
  --workspace-root /path/to/workspace \
  --output /path/to/contexts.json
```

See [`experience/pipeline/README.md`](experience/pipeline/) for the complete
stage contract, inputs, resume behavior, and tests.

### Reinforcement Learning (RL)

The RL experiments use the same Qwen3.5-9B policy, optimizer, sampling
configuration, and repository-disjoint folds for all methods. Only the reward,
loss horizon, and eligible training cohort differ:

| Method | Training rollout | Actor loss horizon | Reward | Training cohort |
| --- | --- | --- | --- | --- |
| TMax (`baseline1`) | full | full | terminal verifier | all training instances |
| TMax-40 (`baseline2`) | full | first 40 assistant turns | terminal verifier | all training instances |
| CRER (`direct`) | full | first 40 assistant turns | golden-rubric judge | eligible training instances |

The golden-reference rubric pipeline generates and re-judges task-specific
criteria from historical rollout groups, retains useful generations and
refinements, and performs robust joint weight optimization. The final bank
allows at most six active rubrics per task; zero-weight rubrics are removed
during materialization. See
[`rubric/pipeline/README.md`](rubric/pipeline/) for the frozen pipeline.

The only released split definition is
[`final_fold_config.json`](slime/train_agent/scripts/final_fold_config.json).
Each of its three folds contains 250 training, 50 validation, and 200 test
instances. To launch a two-node training run:

```bash
sbatch --export=ALL,FOLD=0,EXPERIMENT_METHOD=direct \
  slime/train_agent/scripts/qwen35_earlypred_train.slurm
```

Choose `FOLD=0|1|2` and
`EXPERIMENT_METHOD=baseline1|baseline2|direct`. Runs train continuously from
the base checkpoint. The default 384-attempt budget emits the 128, 256, and
384 checkpoints; setting `TRAIN_INSTANCE_BUDGET_OVERRIDE=512` also emits the
512 checkpoint. Training supports Slurm requeue and resume.

Validation and test jobs take a JSON task manifest and run rollout-only
evaluation, without allocating a Megatron actor:

```bash
sbatch --export=ALL,EVAL_TASKS_CONFIG=path/to/tasks.json \
  slime/train_agent/scripts/qwen35_earlypred_eval.slurm
```

See the
[`RL reproduction guide`](slime/train_agent/scripts/README_early_prediction.md)
for the exact training, continuation, evaluation, and external-artifact
requirements.

### Tests

The automatic artifact pipelines have model-free unit tests:

```bash
PYTHONPATH=experience/pipeline:agent \
  agent/.venv/bin/python -m pytest -q experience/pipeline/tests

PYTHONPATH=.. \
  agent/.venv/bin/python -m pytest -q rubric/pipeline/tests
```

---

## Acknowledgments

This project builds on the mini-SWE-agent scaffold and the vendored
[`slime`](slime/) RL framework, and evaluates coding agents on SWE-bench
Verified. We thank the maintainers and contributors of these projects. We would like to thank NVIDIA for GPU and Inference API support.

---

The repository is released under the [Apache License 2.0](LICENSE).
