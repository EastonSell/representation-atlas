# Pre-answer causal geometry: a failed capability gate

A small, original source-grounded development experiment with SmolLM2-135M-Instruct failed its frozen advancement criteria. No causal steering, train-fitted directions, heldout test, transfer evaluation or intervention confidence intervals were run.

## Results

- Constrained next-token choice: 14/24 (58.3%), required at least 20/24.
- Unrestricted one-word exactness: 0/24, required at least 15/24.
- Both counterfactual members constrained-correct: 2/12 pairs, required at least 6/12.
- All six answer families passed single-token and prefix-preservation checks.
- Original runtime: 28.32 seconds, peak RSS 1.057 GiB, four CPU threads.

Zero exactness is not zero knowledge: every output was a complete sentence, with EOS within 10–11 generated tokens. None hit the 12-token cap. A labeled post-hoc audit found 12/24 correct target propositions, 15/24 matching value words ignoring entity, and five wrong-entity sentences. Constrained selection independently failed its threshold. The full raw prompt/output/entity/value comparison is in posthoc-confusion.csv.

An additional post-hoc diagnostic found 22/24 constrained choices matched the first option listed in the prompt. Template and fact order are confounded with option order; this does not isolate a causal position bias.

These findings apply only to this checkpoint/prompt/parser combination. They do not show that steering is impossible, that another prompt would fail, or that the model lacks all source-grounded knowledge. No prompt repairs, model upgrades or final-test selections followed the gate failure.

## Portable reproduction

Supported reproduction platform: Linux CPU, Python 3.12. The RSS measurement uses Linux ru_maxrss units; macOS and Windows are not supported by this runner. This bundle is portable between Linux workspaces, not cross-platform. Python 3.12 was used. Create a virtual environment of your choice. CPU dependencies:

    python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install transformers==4.46.3 numpy==2.3.5

requirements.txt records the top-level version pins. Transitive dependency versions are in validation-environment.txt. No packages or model weights are bundled.

To obtain the pinned Apache-2.0 checkpoint explicitly (approximately 271 MB):

    python prepare_model.py --destination ./model-cache

This downloader is reused unchanged from the prior activation milestone. It verifies every input against MODEL_MANIFEST.json, including SHA-256 and size. It is an explicit network action, never invoked by inference or tests. If you already have the verified model files, reuse their directory; no download is necessary. Inference itself always uses local_files_only=True and trust_remote_code=False.

Run from this release directory:

    export SMOLLM_MODEL_PATH="$PWD/model-cache"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
    python -m unittest discover -v
    python verify_gate_independent.py --model-path "$SMOLLM_MODEL_PATH" --results results.json
    python gate.py --model-path "$SMOLLM_MODEL_PATH" --output-dir output

The runner also accepts SMOLLM_MODEL_PATH instead of --model-path. The verifier loads only the tokenizer, checks all model-file hashes, and recomputes evidence consistency; it does not regenerate logits. The runner performs all 24 model generations and writes output/results.json, leaving the archived results.json untouched. Use a fresh output directory for each reproduction. Do not use this gate to select favorable final-test cases.

## Provenance boundary

results.json and PROTOCOL.md are byte-identical to the internal historical experiment artifacts. HISTORICAL_FREEZE.sha256 records the original protocol/runner/test hashes frozen before first inference. It is historical evidence, **not a checksum file for the current runner**. The original runner depended on a local cache path and had two bounded implementation defects; it is not bundled. Current gate.py is explicitly post-result portable packaging of a repaired version. No modified source bytes are called preregistered.

The repaired runner rejects incomplete or duplicate-answer pairs and checks output-directory artifact bytes before loading, per row, and before/after writing results, reserving serialized output bytes. The original pair-count defect did not affect the complete 24-row data; the original output size was far below the cap. Tests exercise both repairs. Original archive logs are not included because they contained environment-specific paths. Public validation logs are sanitized to remove local filesystem locations without changing pass/fail counts. A fresh isolated-directory test run and full model replay validate this packaging; see validation.json.

Resource limits are application-level checks: four torch/BLAS threads, RSS and 60-minute checks after each completed row, and a 1 GiB output-artifact cap. There is no hard OS memory quota or external watchdog, and concurrent writes are outside snapshot checks. On a tokenizer failure, incomplete-pair rejection may raise before saving partial results. The current complete data do not hit this path.

## Architecture and limitations

The checkpoint has 30 Llama decoder blocks, width 576. Proposed layers 10/20/30 mean one-based outputs at Python indices 9/19/29, before final RMSNorm. Index 30 is invalid. Extraction is at the final chat generation-prefix token, with no candidate answer appended. Captured residual norms validate the extraction sites, not causality.

Only 12 development pairs across six properties and two templates were tested. Rows are not 24 independent tasks. Distractor values are fixed to each family's second value; template/position effects are not fully crossed. Synthetic numbered entities and partly artificial property phrasing limit transfer. The protocol's wider 120/60/180-pair causal outline was contingent, never implemented or tested; it would require a new implementation/data/analysis freeze. No biological, universal truthfulness or mechanistic-discovery claims are made.

## Sources and licensing

Original code and synthetic data: MIT, see LICENSE. Model: Apache-2.0, immutable revision 12fd25f77366fa6b3b4b768ec3050bf629380bac. Frameworks retain their licenses. Neither model weights nor paper text are bundled.

- Model card: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md
- Zhang and Nanda, activation-patching metrics/methods: https://arxiv.org/abs/2309.16042v2
- Belrose et al., linear-erasure scope: https://arxiv.org/abs/2306.03819v4

Primary-source abstracts motivated the proposed future controls, not the original gate thresholds. This study does not reproduce those papers.
