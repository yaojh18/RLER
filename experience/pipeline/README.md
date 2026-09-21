# Automatic TTS experience pipeline

This is the tested automatic pipeline used to build the final Qwen and
Nemotron TTS experience banks. It starts directly from trajectory-search
artifact directories produced with terminal GT calculation enabled; the
`prepare-contexts` stage freezes their model-visible branch views and terminal
GT into normalized contexts.

The frozen method is:

1. Retain every landed group containing both reward `2` and non-`2` branches.
2. For each group in that cohort whose historically selected branch has final
   reward other than `2`, generate a scope-specific card with
   `openai/azure/openai/gpt-5.6-sol`; if its
   Nemotron-Ultra replay does not strictly improve tie-aware top selection or
   pairwise accuracy, refine once with
   `openai/azure/openai/gpt-5.6-sol` and replay.
3. Extract two keyword sources for every locally accepted card: model-generated
   summary phrases and deterministic phrases that occur in both the visible
   query and the card's activation contract. Their safe union is used only for
   the provisional replay.
4. Replay the provisional bank over the frozen full population. Keep a card
   only when it is retrieved at least once and its mean replay pairwise
   accuracy is at least `0.5`. Select final keywords from the extracted union
   using the same joint-judge replay results. A keyword is retained when the
   mean pairwise accuracy of the replay events attributable to that phrase is
   at least `0.5`; when literal attribution is unavailable, the replayed union
   is preserved instead of making an unsupported deletion.

Generated cards and keywords are schema-checked before filtering. No retrieval-config fitting or retrieval
hyperparameter search is included. Rejected model attempts remain in artifacts
with status `filtered` but never enter the active bank.

Run the local pipeline tests from the repository root:

```bash
PYTHONPATH=experience/pipeline:agent \
  agent/.venv/bin/python -m pytest -q experience/pipeline/tests
```

Start with either a canonical calculate-GT plan or a search-artifact root:

```bash
python -m experience_gen.cli prepare-contexts \
  --plan /path/to/plan.json --workspace-root /path/to/workspace \
  --output /path/to/contexts.json
```
