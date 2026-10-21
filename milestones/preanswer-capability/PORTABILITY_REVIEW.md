# Portability and provenance review

Reviewed 2026-10-21. The release is ready for publication within its explicitly stated **Linux CPU / Python 3.12** scope, subject to retaining the limitations below. No private absolute filesystem paths were found in release source, data, documentation or validation logs. This review performed no model inference or downloads.

## Verified

- Both runner and verifier accept `--model-path` or `SMOLLM_MODEL_PATH`. The runner accepts `--output-dir`; the documented command preserves the archived result file. Help commands work without loading a model.
- The runner and verifier bind model-file checks to the release-adjacent pinned `MODEL_MANIFEST.json`, rather than trusting a replacement manifest in the supplied model directory. Inference remains local-only with remote code disabled. The separately invoked downloader checks the manifest's sizes and hashes.
- Repaired complete-pair validation and output-artifact checks are called by the actual runner, not merely exposed as unused test helpers. Model/prompt/scoring code is not retuned for the observed failure.
- All **18 release tests pass**, with zero expected failures. The tokenizer-only verifier independently recomputes 14/24 constrained, 0/24 exact, 2/12 both-correct, and a failed gate.
- The archived protocol and result bytes match the historical artifacts. Documentation clearly separates those artifacts and historical hashes from the post-result repaired, portable source. It does not call modified source preregistered or claim that causal experiments ran.
- All 24 saved final-validation rows exactly match the archived rows, including logits, tokens and residual norms. The final validation record binds the current runner SHA-256 `8951b5d093d436a77e530855f0aadabf453d571a35c821e3617ef1c8d9ecab99`. This establishes same-input repeatability, not independent statistical confirmation.

## Retained limitations

- Portability means relocation between Linux workspaces, not Windows or macOS support. Linux RSS units are assumed.
- Run normal Python, not optimized `python -O`: some model, memory and timing safeguards use assertions, which optimization disables. Resource checks are application-level snapshots, not an operating-system quota or continuous watchdog. BLAS thread limits rely on the documented environment settings.
- Artifact accounting covers the selected output directory, not separately stored model weights or files elsewhere. Output-size reservation is conservative when replacing an existing result. Use a fresh output directory as documented.
- A partial tokenizer failure raises before saving partial results. The archived and validation datasets have no such failure.
- The historical runner itself is not bundled. Its hash is a provenance record, not a checkable hash of the current portable runner. Current source and validation checksums should be kept together.
- This is a small failed development gate. Formatting failure is distinct from semantic correctness, post-hoc diagnostics do not replace frozen endpoints, and no causal conclusion is supported.
