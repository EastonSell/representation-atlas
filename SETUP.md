# Reproducible local setup and aggregate checks

## Tested profiles, not universal numerical guarantees

The release-safety checkpoint was checked on Linux CPU with Python 3.12.14. `requirements.lock` pins the exact five core packages observed in the passing tests: NumPy, SciPy, scikit-learn, joblib and threadpoolctl. `requirements.txt` now delegates to that lock instead of accepting broad version ranges.

`requirements-model.lock` separately records the exact installed dependency closure for torch 2.5.1+cpu and transformers 4.46.3, including their non-extra dependencies. It excludes unrelated PEFT/training extras. The optional runtime and core runtime were tested in separate existing environments, using the aggregate checker's `--model-python` support. No fresh environment installation or package download was performed for this release-safety checkpoint. Combining the locks in one environment is a setup recipe, not a separately tested clean-install result.

These are exact **version locks**, not supply-chain wheel-hash locks. Wheel availability, Python patch versions, OS, CPU and BLAS differences can still affect numerical behavior. They document the current verification environment and do not retroactively change versions recorded in historical experiment JSON. NumPy/scikit-learn versions used for a new scientific run must still be archived.

## Core setup

Run these commands yourself in a new environment when installation is desired:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python run_checks.py
```

The default aggregate check runs:
1. The full root test suite.
2. The pretrained arithmetic artifact/task tests in their milestone working directory.
3. The seven non-tokenizer preanswer tests in their required milestone working directory.

The remaining 11 tokenizer-dependent tests are explicitly marked **NOT RUN**. A zero exit code means the requested core checks passed, not that optional model verification ran. The summary status is `core_passed_optional_not_run`. Use `--report new-summary.json` for a machine-readable report; existing reports are refused. Use `--plan` to inspect exact commands without running tests.

## Optional pinned-model verification

Use a separate Python 3.12 environment if preferred. The exact CPU torch wheel comes from its official index; the remainder uses the normal package registry:

```sh
python3.12 -m venv .venv-model
.venv-model/bin/python -m pip install 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv-model/bin/python -m pip install -r requirements-model.lock
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python run_checks.py --require-model --model-path /path/to/verified/model --model-python .venv-model/bin/python --report optional-checks-new.json
```

The model directory must already contain the eight files pinned by `milestones/smollm-activations/MODEL_MANIFEST.json`. This aggregate command does not fetch weights, run model generation, fit projections or evaluate held-out benchmark prompts. It verifies all pinned file hashes, then runs the complete 18-test historical preanswer suite using its tokenizer. If model verification is requested but unavailable or fails, the command fails and marks remaining checks not run. `--require-model` without `--model-path` is an argument error. A missing `--model-python` means use the current interpreter.

The two profiles share NumPy 2.3.5. Runtime versions are pinned because the frozen model runner enforces its tested torch/transformers versions. Accelerate, if present as an unrelated extra, may emit its known NumPy namespace deprecation warning; it is not required by these tests.

## Fresh outputs for legacy experiments

The synthetic and digit runners now require an explicit new output file:

```sh
OPENBLAS_NUM_THREADS=2 python run_experiment.py --out /tmp/representation-synthetic-new.json
OPENBLAS_NUM_THREADS=2 python run_digits.py --out /tmp/representation-digits-new.json
```

Configuration files resolve relative to their scripts, so invocation need not occur at the repository root. Existing targets, including dangling symlinks, are rejected before computation. Final writes also use exclusive creation to prevent replacement if a file appears during computation. Parent directories can be created. A disk/write failure can leave an incomplete new file; it does not authorize overwriting it on retry. Choose another new filename after inspection.

The Python API likewise requires `run(output_path, config_path=None)`. Numerical kernels and default configs remain unchanged; only output safety and config-path resolution changed. Tracked historical results are not regenerated. Original runner/requirements bytes are preserved in `milestones/release-safety/historical/` for provenance, not execution. See the release note for historical manifest boundaries.
