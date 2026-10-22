# Frozen supplied-answer evaluation protocol v1

Source-only checkpoint dated 2026-10-22, after benchmark commit `94b384dcd79194fdbe93b07db433432cd8313682` and before any model evaluation of this benchmark. Prior exploratory model results informed the task and controls; this is not independent preregistration.

## Immutable inputs

- Dataset: `milestones/unordered-arithmetic-benchmark/dataset.json`, SHA-256 `396381dcd0472c6f27a45d97732a758da0a6433c7e9100c42b5fb89ec36fe4b6`.
- Model: HuggingFaceTB/SmolLM2-135M-Instruct, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, Apache-2.0. Verify every size and SHA-256 in the existing eight-file model manifest before loading. Manifest SHA-256: `34f007987ac97042a0f1f8278050af521edb868cec1ac083fe4c77ea93bf3ce2`.
- Runtime: torch 2.5.1 CPU, transformers 4.46.3, CPU float32, eager attention, two torch threads, evaluation/inference modes. NumPy performs float64 geometry and its actual version is recorded. No remote code, automatic download, adapter, fine-tuning, GPU or network fallback.
- Existing frozen 60/20/20 split and exact prompts are used without filtering, reordering split assignments, chat formatting or prompt repair.

## Stage 1: tokenizer preflight, no model outcomes

Encode each fixed plain prompt without special tokens. Separately encode prompt + one space + each answer. Both candidate encodings must preserve the complete prompt prefix, have equal total lengths, have a nonempty answer suffix, and have at most 256 tokens. Failure aborts the run rather than silently dropping rows. This check reads all split strings but observes no model outputs.

## Stage 2: training-only capability gate

Score both supplied candidate sequences for only the 60 training rows, without hidden-state extraction. Candidate log likelihood is the sum of teacher-forced suffix-token log probabilities, using token j's distribution at position j−1. The prompt is not scored, and candidate lengths are matched within each row. Archive prefix/suffix token IDs and both log likelihoods for every row.

Advance only if both conditions hold:
- At least 45/60 rows strictly prefer the correct candidate.
- At least 24/33 unordered training groups have every constituent row strictly correct.

Ties fail the gate. These operational thresholds are frozen design choices, not calibrated significance tests. This gate concerns **supplied-answer discrimination**, not independent generation, general reasoning, pre-answer knowledge or causal steerability. A failed gate writes its raw evidence and `gate_failed`, then stops: no validation/test model calls, no activation extraction, no fit and no projection selection.

## Stage 3: training fit and validation selection, conditional on gate pass

Re-evaluate training candidates and evaluate validation candidates, extracting the final returned hidden-state vector at the final candidate token. This is answer-conditioned, post-final-normalization Llama state, not a pre-answer residual site. Training likelihoods must agree with the gate within absolute 1e-4 or the run aborts.

Candidate projections are fixed to removal counts k=0,1,2. For each candidate, fit the mean and PCA directions using only the 120 training answer vectors, and fit the mean correct-minus-incorrect direction using only transformed training pairs. Apply those frozen transforms and direction to 20 validation pairs. Rank with positive margin = success and exact zero margin = half credit. Select maximum validation accuracy, breaking ties toward the smallest k.

Persist `selection.json` before any test inference: all validation candidate margins, selected k, train-fitted mean/PCs/direction, train IDs and validation likelihood evidence. Save train/validation activation arrays. No projection is fitted to validation or test. No additional layer, model, direction, threshold or prompt is selected.

## Stage 4: selected test once

Only after selection is persisted, evaluate the 20 test pairs once and report the selected projection's margins, half-tie ranking and candidate likelihood preference. Do not compute a test leaderboard across k. Archive every selected test margin, every likelihood pair and tokenization evidence, training margins and raw activation arrays. Save a file-hash manifest plus runtime/source provenance. Test outcomes cannot influence selection.

The code enforces sequencing and refusal to overwrite an output directory, not a global lock against an operator rerunning in a different directory. Results must not be used to tune this frozen test protocol. Such changes require a new dataset/protocol and explicit exploratory labeling.

## Failure, resources and interpretation

An orchestration exception preserves prior stage files and writes an incomplete-stage marker, never a successful evaluation. A tokenizer/model-load failure occurs before orchestration and may leave no run directory. One-hour checks occur between forward calls; these are cooperative, not an external watchdog. Batches contain two candidate sequences, maximum 256 tokens each. No hard OS memory limit is installed. Existing directories are refused; partial runs are retained rather than resumed or overwritten.

Pretrained arrays and likelihoods are not supplied by this source checkpoint. Tiny fake-backend unit tests validate orchestration and train/validation/test separation; they do not validate model accuracy or all runtime behaviors. No p-value, confidence interval, causal intervention, hidden-state steering, brain mapping or biological correspondence is claimed. Reversed-operand rows are clustered and only 11 unordered test problems exist. This is a deliberately small, public, narrow arithmetic task whose negative policy and operand range differ from historical experiments.

## Official API references consulted

- https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/model (local-only model loading, evaluation mode defaults, dtype and safetensors options).
- https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/output (hidden-state/logit output structures).
- https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/llama (causal LM output and logits).

Model card: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md . Original runner/test code is MIT under the repository license. Weights and third-party source are not bundled.
