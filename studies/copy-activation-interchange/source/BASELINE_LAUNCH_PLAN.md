# Exact baseline launch plan for pre-inference review

Do not execute this plan until the V2 source/data freeze and integration canary
have been reviewed and the baseline phase explicitly cleared.

1. Use the existing pinned Python environment and local SmolLM2 snapshot. No
   installation, download, CUDA, remote API or additional model is permitted.
2. Verify PACKAGE_MANIFEST.json, SOURCE_MANIFEST.json, DATA_FREEZE.json, model
   receipt hashes and the two vendored dependency hashes. Confirm data rows and
   tokenizer records match V1 exactly.
3. Place the exact-stage execution-clearance file outside the source and evidence
   directories. It must identify stage baseline, approved true, the frozen source
   manifest SHA-256 and DATA_FREEZE SHA-256. Its presence does not replace review.
4. Choose a nonexistent phase output directory and run from the frozen V2 root:

   PINNED_PYTHON -m patchstudy.execution launch baseline --root FROZEN_V2_ROOT
   --model-dir LOCAL_PINNED_MODEL --out NEW_BASELINE_PHASE --clearance EXTERNAL_CLEARANCE

   Treat the displayed wrapped command as one shell command. All paths are caller
   arguments, not personal paths embedded in public source.
5. Launcher verifies frozen inputs and clearance, reserves only status.json, and
   starts the trusted worker under the reviewed supervisor. No warmup forward.
   Supervisor budget: wall_seconds 300, cpu_seconds 605 (effective child CPU
   threshold 600), rss_bytes 3221225472, sampling_seconds 0.1. Environment pins
   offline mode and two BLAS threads; worker sets torch intra-op 2/inter-op 1.
6. Worker loads the local model, then executes exactly untouched and read-only
   capture forwards for row IDs 0 through 63 in order, at most 128 attempts.
   Every attempt is durably planned first. Any technical/control/nonfinite error
   stops the phase; all baseline competence failures remain without replacements.
7. Require study-level acceptance with accepted_phase(NEW_BASELINE_PHASE), not
   worker status alone. Inspect the complete gate, all 128 vectors and cache
   hashes, full identity errors, attempts, resource summary and manifests. Run
   audit_phase(NEW_BASELINE_PHASE, source_path=FROZEN_V2_ROOT,
   model_path=LOCAL_PINNED_MODEL) without inference.
8. Stop after baseline. Even a passing gate does not authorize patching. If the
   gate fails or artifacts/resources are incomplete, retain all evidence and
   report failure. No same-run retry or automatic continuation is permitted.

Expected baseline vector bytes: 25,165,824. Raw cache bytes: 13,418,496. Metadata,
logs and manifests add modest overhead. Whole-study archive cap remains 1 GiB.
