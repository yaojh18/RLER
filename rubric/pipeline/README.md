# Teacher-rubric pipeline

1. Use `prepare_collection.py` with a JSONL manifest of calculate-GT rollout
   artifacts. The collection starts from an empty seed portfolio and retains only historical
   GT-variance groups and binds the cutoff-visible trajectories to exact
   terminal-reward difference pairs.
2. Run `generate_refine.py teacher-gen`, then `luna-eval`. Each generated
   rubric is evaluated independently and retained only when its strict-pair
   accuracy exceeds 50%.
3. Run `generate_refine.py teacher-refine` on every non-improving generated
   rubric, then run `luna-eval` again. Refinement must return one changed rubric
   per failed candidate, and it is retained only when its strict-pair accuracy
   exceeds that direct candidate before refinement.
4. `build_weight_problems.py` combines accepted generation and refinement
   candidates without a six-rubric prefilter. `optimize_weights.py` performs non-exhaustive
   successive-halving plus coordinate/swap search with the fixed score mapping
   `1,2,4,6,8`. Legal weights are zero or 0.5 through 5.0 in 0.1 increments;
   at most six are active.
5. `materialize_bank.py` copies the optimized rubric lists into the given bank;
   eligibility and score mapping remain unchanged. The given abstention detector
   remains a separate runtime configuration.

Run the model-free pipeline tests from the repository root:

```bash
PYTHONPATH=rubric/pipeline \
  agent/.venv/bin/python -m pytest -q rubric/pipeline/tests
```
