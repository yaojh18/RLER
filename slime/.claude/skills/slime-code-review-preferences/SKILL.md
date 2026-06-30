---
name: slime-code-review-preferences
description: Use when reviewing or editing slime code, especially refactors around helper APIs, branch selection, argument validation, or recurring reviewer preferences about avoiding unnecessary wrappers and making control flow self-explanatory.
---

# Slime Code Review Preferences

Apply these lightweight review heuristics when changing slime code.

## Prefer Direct APIs Over Thin Wrappers

- Remove helper layers that only rename a call, format one path, or forward arguments without owning meaningful behavior.
- Prefer calling the concrete reusable API directly, for example a `*_to_path` helper when the caller already knows the destination path.
- Keep a wrapper only if it owns a real boundary: compatibility, validation, nontrivial error policy, lifecycle management, metrics/logging semantics, async/retry behavior, or cross-module ownership.
- Avoid moving a redundant wrapper's body into another file just to preserve the wrapper shape. Inline the simple call at the natural ownership site.
- When removing a wrapper, search for sibling wrappers and nearby helpers with `rg` and delete confirmed dead functions in the same pass.
- Treat single-use convenience functions as suspicious when their only job is path formatting plus forwarding. Prefer the caller owning that one line.

## Make Branches Explain Themselves

- Order conditionals by semantic precedence: special transport/lifecycle modes first, then explicit mode choices, then default paths.
- Prefer predicates that fully describe the branch, such as `mode == "full" and transport == "disk"`, over a broad predicate followed by an assert that explains what the branch really meant.
- Use asserts as invariants for impossible states after validation, not as a substitute for clear branch conditions.

## Keep Abstractions Honest

- Add an abstraction only when it removes real duplication, hides fragile mechanics, or clarifies ownership.
- When a review comment points out repeated indirection, look for a smaller public surface rather than adding another alias.
- Preserve existing behavior intentionally. If cleanup changes error handling, logging, or failure visibility, call that out in the final response.

## Fail Loudly At Real Boundaries

- Do not add fallback values, dummy samples, fake token IDs/logprobs, or exit-code-derived rewards to keep a run moving.
- Distinguish evaluator results from infrastructure exceptions. A normally returned evaluator failure is a valid zero-reward rollout; container startup, host execution, timeout, or evaluator invocation exceptions invalidate the trajectory and its complete sibling group.
- Fix missing official evaluator artifacts at their producer or path contract. Do not reconstruct `reward.txt`, reports, or passed tests from unrelated signals.
- Reuse the benchmark's official evaluator and keep `evaluate_swebench_instance_patches` as the external entry point.

## Preserve Exact Rollout Data

- Token-in/token-out is required for RL samples. Missing or malformed prompt IDs, output IDs, or logprobs invalidate the trajectory; one invalid sibling invalidates the complete GRPO group.
- Never re-tokenize text or synthesize zero logprobs as a training fallback.
- When a valid sample exceeds the training token budget, truncate tokens, masks, and rollout logprobs together while preserving alignment and a useful trainable prefix.
- Compute benchmark reward once in the shared evaluator payload. Persistence and bundle/train conversion must consume that value; training may only derive the configured group advantage from it.

## Load Datasets Lazily

- Load benchmark data from its Hugging Face dataset and select only requested instance IDs. Do not materialize a full dataset with `list(dataset)`.
- Keep benchmark-specific schema conversion at the dataset/evaluator boundary so rollout code continues to consume the SWE-bench-shaped instance contract.

## Run Cluster Experiments In One Allocation

- For one requested validation session, obtain one interactive Slurm allocation and run all related debugging inside it. Reallocate only when the allocation environment itself is unusable.
- Start from the known working `nemotron_rl_rm` / `normal` / `interactive` configuration in the standard naive and lane smoke scripts; request enough GPUs in that allocation instead of submitting several jobs.
- Run the outer Slime environment from the solidified Slime image and use Apptainer/Singularity SIF files under `singularity_images` for per-instance isolation. Do not reintroduce Enroot as the instance runtime.
- Validate prerequisites explicitly and stop on missing images, models, credentials, reports, or runtimes. Do not add a second runtime or network-download fallback during an experiment.
