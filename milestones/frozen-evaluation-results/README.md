# Frozen arithmetic evaluation: failed training capability gate

SmolLM2-135M-Instruct failed the [published, frozen runner protocol](../frozen-evaluation-runner/PROTOCOL.md) on the new unordered-arithmetic benchmark. The runner and thresholds were published at commit `42b78dee798ddb4faf50f7aa9ba877e88701302a` before these model outcomes were collected.

## Result

- Strict supplied-answer preference: **30/60 training rows (50%)**, required at least 45/60.
- Every member strictly correct: **16/33 unordered training problems**, required at least 24/33.
- There were 30 positive margins, 30 negative margins and no ties. One reversed-operand group disagreed between its two orders.
- All 60 evaluated answer pairs had matching two-token suffix lengths and preserved their prompt prefixes.
- A second invocation reproduced `gate.json` and `results.json` byte-for-byte, including every likelihood, token ID and failure decision.

Both runs stopped at the training gate. **No validation/test model inference, activation extraction, geometric fitting, projection selection or causal intervention was performed.** Thresholds, tokenization, scoring and prompts were not relaxed or repaired after failure.

This is a real pretrained forward evaluation, not a fake-backend or random-weight demonstration. The checkpoint revision was `12fd25f77366fa6b3b4b768ec3050bf629380bac`; all eight pinned input files were checked before loading. Runtime was torch 2.5.1+cpu, transformers 4.46.3, NumPy 2.3.5, Python 3.12.14, float32, eager attention and two CPU threads. The first forward-evaluation phase took about 9.65 seconds; both precise timings are recorded in the run folders. Other experiments shared the host, and initialization/loading is excluded, so these are not portable speed benchmarks.

## Raw evidence and verification

Both `run/` and `repeat/` contain:
- `gate.json`: all 60 exact training IDs, candidate log likelihoods, their differences, prefix/suffix token IDs and gate decision.
- `results.json`: explicit failure/stop status.
- `runtime.json`: versions, timing and runner/geometry/model-manifest hashes.
- `artifact_manifest.json`: byte counts and SHA-256 hashes of those three files.

From the repository root:

```sh
python -m unittest discover -s tests -p test_frozen_gate_results.py -v
python -m unittest discover -s tests -v
```

The independent standard-library evidence tests do not import the model runner. They recompute row/group outcomes, check exact train-only membership, verify finite log likelihoods and margins, validate hashes, check failure status/no downstream artifacts, and compare raw repeats. They check internal consistency, not logits from a separate model implementation. The full root suite passes 39 tests; the five existing pretrained-milestone tests and all 18 old preanswer tokenizer/evidence tests also pass with the documented local runtime.

To perform a new frozen reproduction, use the runner command in the source checkpoint with a new output directory and the verified local model. Do not overwrite either archive. Weights are not redistributed here. Code, generated arithmetic data and results are MIT under the repository license; model license remains Apache-2.0.

## Interpretation and next boundary

This failed the operational threshold for supplied-answer discrimination on one narrow task and fixed checkpoint/prompt format. It does not establish an inability to learn arithmetic, inability to steer a different model, a universal factuality result, or biological correspondence. Candidate answers are provided to the model; independent generation and pre-answer capability are not measured. Only 33 training problem groups were involved, reversed rows are dependent, and no significance test or confidence interval is claimed.

A future remedy must be a separate model/prompt choice and newly frozen protocol, not threshold loosening or reinterpretation of the current failure. Held-out model outcomes remain unobserved under this runner. Historical experiments, the earlier source-grounded failed gate, the benchmark and this source freeze remain unchanged.
