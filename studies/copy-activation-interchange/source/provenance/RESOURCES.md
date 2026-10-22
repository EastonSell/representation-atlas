# Prospective resource forecast and hard caps

No model was loaded or benchmarked for this proposal. Forecasts are planning estimates, not measured performance.

## Forward counts

- Tokenizer/schema/unit-test phase: zero pretrained forwards.
- Baseline: 64 untouched + 64 read-only-capture = 128.
- Patch: 64 directed prompts × 7 layers × 6 conditions = 2,688.
- Total maximum: 2,816 attempted full-prefix forwards; zero training/backward/generation calls. An interrupted/failed attempt counts. No separately uncounted warmups or benchmarks.

Assume at most 32 prompt tokens, verified in tokenizer preflight. A dense rough upper-envelope calculation 2×135e6×32×2816 is 24.33e12 FLOPs, excluding some attention/bookkeeping and including an intentionally coarse parameter-based accounting. Actual prompt lengths will be reported and the estimate recomputed BEFORE outputs without changing any cap. This is not a speed benchmark.

At assumed 0.1–0.5 seconds per forward, compute alone is approximately 4.7–23.5 minutes, plus model loading, state capture, hashes and output. Shared CPU contention may exceed this. Total execution deadline 2,400 wall seconds, including model load and both phases; baseline cap 300 seconds and patch cap 2,100 seconds, fixed before inference. Phase review idle time is not execution time. Timer values are summed from phase start/end and count startup; a supervisor enforces each phase deadline externally. Cooperative internal checks alone are insufficient for a hard cap. One 15-second termination grace can exceed the nominal deadline and must be reported; no claim of literal zero-overrun enforcement. If supervisors/resource limits cannot be set with available permissions, do not begin the run under a false hard-limit claim.

## CPU, memory, disk, financial cap

- Existing CPU only, two torch intra-op/BLAS threads and one inter-op thread. No GPU reservation, installation, download, paid compute or API inference. Incremental paid compute budget: zero.
- Working memory forecast 1–2 GiB, 3 GiB resident-memory watchdog cap. Float32 model parameters alone roughly 540 MB, plus runtime, temporary tensors and caches. Polling RSS can overshoot between checks; report enforcement method and actual peak.
- Full final-logit archive: 2,816×49,152×4 bytes = 553,648,128 bytes (~528 MiB), includes untouched, capture-check and every patch result. Retaining all intervention logits closes a previous independent-audit limitation.
- Seven-layer baseline-state cache upper bound 64×7×32×576×4 = 33,030,144 bytes (~31.5 MiB). Actual short sequences smaller. Avoid all-layer/all-forward state archival.
- Scalars, source, hashes, tests and report reserve <100 MiB. Expected archive <700 MiB uncompressed; new-output disk cap 1 GiB with checks before each save. No deleting evidence to remain under cap.
- Bootstrap: 10,000 resamples of 32 groups × 12 substantive cells, about 3.84 million scalar samples; batch or stream without retaining giant replicated tensors. No model calls.

Freeze all caps in RUN_BUDGET.json before any inference. On resource/identity/gate failure, retain evidence and report the failure; no extra rounds, added candidates or threshold changes. The test remains valid as an incomplete/failed study, not a successful full map.
