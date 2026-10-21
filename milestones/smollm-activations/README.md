# Pretrained language-model geometry milestone

This experiment extracts real final-layer activations from the Apache-2.0 SmolLM2-135M-Instruct checkpoint for 100 matched, exactly verifiable arithmetic-answer pairs. It tests whether a training mean-difference direction ranks supplied correct answers above incorrect ones on held-out prompts.

## Findings
| Split | Center only | Remove 1 PC | Remove 2 PCs | Model likelihood prefers correct |
|---|---:|---:|---:|---:|
| Random operand pairs | 40.83% | 36.67% | 37.50% | 68.33% |
| Answer-balanced first-operand groups | 53.33% | 54.17% | 61.67% | 70.83% |

Entries average three specified seed splits. Twenty shuffled-label directions per split/configuration average approximately 49–51%. Centering-only group-split scores are 52.5%, 55%, 52.5%. The group split was added after inspecting the initial below-chance result to diagnose answer-token composition imbalance; it is exploratory. The two-PC result is not a validation-selected winner or a claimed general improvement. Overlapping splits and 100 prompts do not support broad statistical conclusions.

The initial and follow-on passes produced byte-identical extracted-activation hashes. CPU extraction plus analysis took 14.3 and 18.5 seconds on this shared host with two PyTorch threads. Timings are not hardware-independent benchmarks.

## Reproduce
Python 3.12 was used. Create a separate environment, install the documented pinned CPU runtime, then explicitly fetch the 271,177,292-byte checkpoint:

```sh
python -m venv .venv
.venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers==4.46.3 numpy==2.3.5
.venv/bin/python prepare_model.py --destination ./model-cache
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python run.py --model-path ./model-cache --threads 2
.venv/bin/python -m unittest -v
```

`MODEL_MANIFEST.json` pins the model revision and every input-file hash. The downloader verifies existing cache entries and all fetched files. `run.py` requires local files and disables remote model code. Raw pretrained weights are not bundled. Run from this milestone directory within the repository, because it imports the shared geometry implementation two directories above.

## What the measurement establishes
The model sees the candidate answer as input. Hidden-state pair ranking and conditional likelihood preference answer different questions. A score does not establish the model independently generated that answer, understood the task robustly, or can be causally steered. The prompt uses a fixed plain question/answer format rather than the model's chat template, a separate protocol variable to test later. Both labels have identical global answer-token distributions; tokenizer prefix and answer-length checks accepted all 100 pairs.

Centering cancels in pair differences. Removing top components changes the space and may help or hurt. Matching and control selection need care even with exact task labels. The next confirmatory experiment must use new prompt families and a frozen split/projection protocol selected without those test results.

## Sources and license
Model card at immutable revision: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md . Model license: Apache-2.0. Original code, generated arithmetic data and result analysis fall under this repository's MIT license. Framework dependencies retain their own licenses. No biological data or human-comprehension evaluation is involved.
