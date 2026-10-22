# Copy-target activation patching: pre-inference source snapshot

A reproducible, prospective fixed-panel experiment for SmolLM2-135M-Instruct.
This package has source, synthetic tests and tokenizer-only data, not model
results. No pretrained forward, training, evaluation or publication was run.

The frozen design contains 64 words in 32 disjoint lexical pair groups, both
orientations, seven raw decoder-output sites, six prespecified conditions and a
maximum 2,816 attempted forwards. Pair groups are the resampling units; they are
not independently sampled English words. See PROTOCOL.md for hypotheses,
controls, gates and limitations, and RESOURCES.md for fixed resource caps.

## Reproduction without inference

Use an already installed offline environment with Python 3.12, torch 2.5.1+cpu,
transformers 4.46.3 and NumPy 2.3.5. No install/download is part of this package.
Run `python -m unittest discover -s tests -v` from this directory. Tests use
synthetic tensors or architecture-shaped toy networks, never pretrained weights.

The recorded tokenizer audit used the pinned model revision in PROTOCOL.md.
To repeat it, copy source into a new directory with an empty data directory and
run `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m patchstudy.preflight
--model-dir PATH_TO_LOCAL_SNAPSHOT --root NEW_DIRECTORY`. Existing artifacts
are never overwritten. The audit imports a tokenizer only, preserves every
rejection and stops if fewer than 64 candidates qualify.

## Inference boundary

The legacy runner entry point remains disabled. V2 adds the portable supervised
launcher `python -m patchstudy.execution launch baseline` with explicit source,
model, output and external clearance paths. See BASELINE_LAUNCH_PLAN.md. This is
source-only integration, not authorization to execute. Source/data review and a
reviewed lightweight canary remain outstanding; no model launch has occurred.
There is no automatic baseline-to-patch transition or same-run retry.

## Package contents

- `provenance/`: untouched proposal, candidate pool and historical exclusions.
- `data/`: all candidate eligibility, selected prefixes/pairs and tokenizer hashes.
- `patchstudy/`: selection, raw hooks, phase engines, full-logit storage, scores,
  independent artifact audit and deterministic report generation.
- `tests/`: synthetic and adversarial correctness tests.
- Source/data/package manifests bind exact reviewed bytes without self-hashing.

No model weights, private messages, execution-clearance records or machine-specific
runtime paths are distributed. This is a local freeze, not public preregistration.
