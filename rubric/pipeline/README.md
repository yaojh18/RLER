# Automatic golden-reference rubric pipeline

This package retains only the model-driven pipeline used by the final bank. It
does not contain human refinement, eligibility fitting, or abstention-threshold
fitting.

1. Use `prepare_collection.py` with a JSONL manifest of calculate-GT rollout
   artifacts. It writes `instance_manifest.json` and one
   `current_ledgers/<instance>.json` per task, retaining only historical
   GT-variance groups and binding visible trajectory views, exact Luna scores,
   the current handbook, and strict GT-difference pairs.
2. Run `generate_refine.py initial-gen`, then `luna-eval`. Initial generation
   retains any rubric with positive pairwise accuracy against its zero baseline.
3. Run `sol-gen`, `luna-eval`, `sol-refine`, and `luna-eval`. Sol refinement is
   mandatory for every failed candidate. Every generated addition is compared
   with a zero baseline and retained when it has any positive strict-pair
   signal; a modification is retained only when it strictly improves its
   recorded parent rubric.
4. `build_weight_problems.py` combines every accepted candidate without a
   six-rubric prefilter. `optimize_weights.py` performs non-exhaustive
   successive-halving plus coordinate/swap search with the fixed score mapping
   `1,2,4,6,8`. Legal weights are zero or 0.5 through 5.0 in 0.1 increments;
   at most six are active.
5. `materialize_bank.py` copies the optimized rubric lists into the given bank;
   eligibility and score mapping remain unchanged. The given abstention detector
   remains a separate runtime configuration.

All hosted calls use the configured NVIDIA-internal route and read the API key
from `RLER_MODEL_API_KEY`. Completed per-instance outputs can be resumed. The
Luna judge prompt is stored directly in `prompts.py`.

Run the model-free pipeline tests from the repository root:

```bash
PYTHONPATH=rubric/pipeline \
  agent/.venv/bin/python -m pytest -q rubric/pipeline/tests
```
