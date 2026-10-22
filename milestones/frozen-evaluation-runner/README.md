# Frozen evaluation runner: source checkpoint

This checkpoint supplies an offline runner for the [unordered arithmetic benchmark](../unordered-arithmetic-benchmark/README.md), plus fake-backend tests and a [frozen protocol](PROTOCOL.md). **No model result on that benchmark is included.**

The runner first gates supplied-answer discrimination on training data only. Failure stops before held-out inference or projection fitting. On a pass, centering/PCA and directions are fitted on training, k=0/1/2 is selected on validation, selection is saved, and only the selected configuration is evaluated on test. Every candidate likelihood and selected per-pair geometry margin is archived, with tokenizer IDs, raw activation arrays and provenance hashes.

## Run after obtaining the separately verified local model/runtime

From the repository root, with a local model directory matching all eight pinned files and a Python environment containing torch 2.5.1 CPU, transformers 4.46.3 and NumPy:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false python benchmark_runner.py --model-path /path/to/verified/model --output-dir /tmp/representation-evaluation-new
```

This command does not install or download anything. It refuses existing output directories, mismatched dataset/model hashes, runtime version mismatches, and tokenizer prefix/length failures. It does not change historical results. A passing run fits no new language model and performs no intervention.

Expected artifacts:
- Always after a completed gate: `gate.json`, `results.json`, `runtime.json`, `artifact_manifest.json`.
- Only after a passing gate: `selection.json`, `training_validation_activations.npz`, and on test completion `activations.npz`.
- On an orchestration exception: `failure.json`, alongside any prior stage evidence. Load/preflight exceptions can occur before an output directory exists.

## Source verification

```sh
python -m unittest discover -s tests -v
sha256sum -c milestones/frozen-evaluation-runner/SOURCE_FREEZE.sha256
```

The source-only root suite passes 34 tests. Ten runner tests cover gate fail/stop, train-only fitting, persisted validation selection before test, test-value independence, malformed evidence, overwrite prevention, incomplete failure artifacts, tokenizer preflight and frozen input checks. Test backends are synthetic fixtures, not measurements of SmolLM2. Existing pretrained-milestone tests also remain separate from real inference.

The gate's 45/60 strict rows and 24/33 entirely-correct groups are operational advancement thresholds, not significance tests. This runner evaluates supplied candidate answers; it cannot establish independent generation or pre-answer causal capability. Only 11 unordered test problems exist, and dependent reversed rows must not be treated as independent evidence. No brain data, brain mapping, universal truthfulness, causal steering or biological claim is involved.

Using the verified shared tokenizer/runtime, all 18 existing preanswer tests now also pass, including the previously blocked independent evidence verifier. This checks archived evidence and tokenizer behavior; it does not rerun model generation or reverse that historical failed gate. An initial invocation from the repository root hit that old suite's working-directory assumption; rerunning from its documented milestone directory passed. The dependency stack emits a NumPy deprecation warning from Accelerate.
