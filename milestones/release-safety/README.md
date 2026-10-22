# Release safety and reproducible check entrypoint

## Outcome

- Core aggregate: **62 tests passed** (50 root, 5 pretrained-artifact, 7 preanswer-core); 11 optional tokenizer methods explicitly NOT RUN.
- Explicit optional aggregate: **73 tests passed** (50 root, 5 pretrained-artifact, 18 preanswer), plus all eight pinned model-file hashes verified.
- Exact core and optional-runtime dependency versions match the two existing tested Python 3.12.14 environments. No installation or download was performed. See [SETUP.md](../../SETUP.md) for setup recipes and lock limitations.
- Both legacy runners require a fresh `--out`; existing paths and dangling symlinks are rejected before computation, and exclusive creation prevents a final-write replacement race. Tiny synthetic/digit tests exercise actual fresh writes without changing tracked results.

These are software/provenance checks, not a new scientific result. The frozen arithmetic model gate still failed twice at 30/60 rows and 16/33 unordered groups. There was no new model generation, frozen held-out evaluation, representation extraction, projection fit, causal experiment or brain mapping.

## Evidence and limitations

`final-core-checks.json` and `final-full-checks.json` are the final aggregate reports. The matching test-output files include all method outcomes and counts; only the optional environment's absolute prefix was replaced by `<MODEL_RUNTIME>` for portable presentation. Timings are incidental shared-CPU measurements, not performance claims.

Initial diagnostic reports are retained under `diagnostics/`. The first optional aggregate failed because resolving the chosen interpreter symlink bypassed its virtual environment. This was a checker defect, not a failed capability result. Lexical absolute-path normalization fixes it without dereferencing the interpreter; a dedicated regression reproduced the failure before the fix and passed afterward. No missing dependency was installed to conceal it. The final optional aggregate passed all requested checks.

Default aggregate success deliberately has status `core_passed_optional_not_run`, not full verification. Requested optional verification fails on missing/bad assets or unavailable runtime and does not silently skip. `--require-model` without a path fails argument validation. Model-file hashes are checked before tokenizer-dependent tests, all subprocesses are offline, and neither mode generates model answers.

Version locks pin observed versions, not wheel hashes or universal cross-platform floating-point identity. They are not claimed to be fresh-install tested. The optional lock excludes unrelated installed training extras. See setup notes for the exact division of runtimes.

## Historical preservation

Original runner and requirements bytes are preserved in `historical/` and checked against the unchanged original root `MANIFEST.json`. Historical manifests describe their source checkpoints; root runner hashes intentionally change for this release. Frozen benchmark and runner source hashes, configurations, geometry implementation, datasets and all tracked historical scientific outputs remain unchanged. This milestone's publication manifest records the new allowlisted bytes instead of rewriting old evidence.

Use the current root entrypoints, not the historical copies. Failed new writes may leave an incomplete new file; retries must use another fresh filename after inspection. Output safety is not a transactional multi-file artifact store.
