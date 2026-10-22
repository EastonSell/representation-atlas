# Copying subspace source and tokenizer-only dataset freeze

## Checkpoint outcome

- The fixed 80-word pool passed tokenizer preflight on its first attempt: **40/20/20 distinct prompts**, **20/10/10 disjoint counterfactual groups**, zero word/prompt/pair overlap across splits, and zero overlap with discovery words. Every answer is one token; paired prompts have equal length and differ at exactly one source token. Metadata-only lexical priority scores exactly 50% in each split.
- **16 local tests, 6 prior-discovery tests and 62 aggregate core tests passed.** Optional aggregate tokenizer methods were explicitly not rerun in this checkpoint.
- Real-model hook verification used **two forwards on one training row only**. Untouched versus no-op hooked logits and block-state indexing matched exactly (maximum absolute error zero). The saved binary logits independently reproduce the recorded greedy token and candidate margin.
- **No training capability gate, PCA fitting, validation/test model inference or nonzero intervention has run.** Mock tests exercise all stages without pretrained inference. The source is ready for review of the training gate, not evidence that the future subspace experiment succeeded.

The prior arithmetic gates remain failed. This is an explicitly different copying task, grounded in the narrow [discovery](../copy-retrieval-discovery/README.md), not arithmetic recovery or evidence of semantic truth/brain alignment.

## Freeze and provenance

`words.json` is an investigator-authored pool, fixed before tokenizer checks. `DATA_FREEZE.json` records actual local freeze timing and hashes. `dataset.py` deterministically sorts SHA-256(`copy-subspace-v1|` + word), partitions 40/20/20 and pairs adjacent words in both orientations. The source/foil tokens are balanced within each split, not shared across lexical-disjoint splits.

`SOURCE_FREEZE.json` fixes the dataset, driver, tests, approved design and model manifest before the real hook check. It honestly records a local freeze, not prior public registration. Nothing was downloaded or installed. The tokenizer-only first attempt emitted a cache-location warning but loaded the local tokenizer successfully; no model weights were loaded during dataset construction. Real verification used the existing approved shared runtime and pinned safetensors model.

Read [PROTOCOL.md](PROTOCOL.md) for the precise stage boundaries, controls, numerical choices and resource limits. Training baseline states are cached for no-op verification/provenance before the capability decision, but no fit is made unless the frozen gate passes. This is explicit rather than claiming that zero states are ever cached before a passing gate.

## Staged execution and resource limits

| Command | Real inputs | Maximum forwards | Maximum cooperative seconds |
|---|---|---:|---:|
| verify-hook | one training row | 2 | 120 |
| train | 40 training rows | 80 | 300 |
| select | 20 validation rows | 1,300 | 900 |
| test, selected k > 0 | 20 test rows | 460 | 600 |
| test, selected k = 0 | 20 test rows | 40 | 600 |

Two CPU threads and offline CPU float32/eager evaluation are fixed. Budgets are checked before each forward, not hard process kill timers. A fresh output directory is required; rows are appended and fsynced incrementally, baseline arrays are compressed binary NPZ with hashes, and caught failures preserve partial evidence. No in-place resume or automatic train-to-test chain exists.

The training gate requires 36/40 strict margins, 36/40 greedy exact tokens and 18/20 entirely-correct pairs. It stops on failure. Selection and test each require the exact upstream artifact-manifest SHA-256, verify every archived file, and require a matching frozen source digest. Validation chooses only among fixed k = 0,1,2,4. Test cannot choose another k. Ten fixed random bases for each nonzero k are used for both rank-only and pointwise energy-matched controls.

Use an existing runtime/model; commands below do not authorize skipping result review:

```sh
python -m unittest discover -s milestones/copy-subspace -v
/path/to/model-runtime/bin/python milestones/copy-subspace/driver.py train --model-path /path/to/verified/model --out /tmp/copy-train-new
# After reviewing a passing train result:
/path/to/model-runtime/bin/python milestones/copy-subspace/driver.py select --model-path /path/to/verified/model --out /tmp/copy-selection-new --upstream /tmp/copy-train-new --upstream-digest EXACT_TRAIN_MANIFEST_SHA256
# Only after reviewing frozen selection:
/path/to/model-runtime/bin/python milestones/copy-subspace/driver.py test --model-path /path/to/verified/model --out /tmp/copy-test-new --upstream /tmp/copy-selection-new --upstream-digest EXACT_SELECTION_MANIFEST_SHA256
```

The root aggregate does not yet discover these milestone-local tests; run the explicit command as well. `hook-verification/arrays/train-00-0.npz` contains the actual full float32 logits and residual, not rounded decimal JSON. `validation.json` reports the independent binary check, source verification and exact execution boundaries. Hook output arrays must accompany its manifest for a complete archive; publishing metadata alone must be labeled as such.

## Interpretation limits

A convenience-word, fixed-template study tests lexical transfer within that format, not arbitrary copying or pretraining independence. Forty training points estimate a small subspace of 576-dimensional states; PCA variance is not intrinsically semantic importance. Accuracy-preservation tolerances are operational, not formal equivalence tests. A perturbation difference against controls supports sensitivity at this site/task, not an exclusive necessary mechanism. Full untouched logits, all conditions and paired group-level values are retained to prevent selective reporting.
