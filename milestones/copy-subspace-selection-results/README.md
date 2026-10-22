# Validation selection: remove k=4, test still locked

The fixed validation rule selected **k=4** from `{0,1,2,4}`. All three nonzero candidates were eligible. This is the rank of the **removed** train-PCA subspace at block14's last prompt position, not a four-dimensional retained representation or compression result.

| Condition | Greedy exact | Mean target-minus-foil margin |
| --- | ---: | ---: |
| Untouched | 20/20 | 7.956754231452942 |
| Remove PC1 | 20/20 | 7.953857517242431 |
| Remove PCs1–2 | 20/20 | 7.946023750305176 |
| Remove PCs1–4 | 20/20 | 7.915532898902893 |

All these conditions had 10/10 both-orientation positive-margin groups and, separately, 10/10 both-orientation greedy-exact groups; zero pairwise ties. The unchanged selection thresholds were greedy accuracy ≥0.95 and mean margin ≥7.161078808307648 (90% of untouched). The largest eligible k was selected without outcome-dependent changes.

Exactly **1,300 forwards** completed: 20 untouched baselines, 20 no-op checks and 1,260 interventions, covering every fixed PCA, ten rank-matched random seeds and ten energy-matched random seeds for each of k1,2,4. All requested conditions, including failures if any, are preserved in incremental scalar rows. All untouched full-vocabulary logits and pre-answer states are in 20 binary NPZ files. The immutable training fit NPZ was copied byte-for-byte, not refitted.

No-op logit/state errors were zero. `audit_selection.py` independently verifies all archive/source hashes, exact validation IDs, all untouched predictions/probabilities/margins, selection arithmetic, and all requested/realized perturbation norms reconstructed from baseline states and the fixed fit. Maximum requested-norm reconstruction error was 1.16e-14; realized-norm error was zero. Run it with the tested Python runtime; default output goes to stdout, or `--out` requires a fresh path. Re-running reproduced `audit.json` byte-for-byte.

**Intervention full logits were not archived.** Their scores are recorded directly by the frozen driver and grouped by the audit; unlike baseline scores they cannot be independently reconstructed from archived full logits. Norm reconstruction does not independently verify intervention predictions. Raw scalar intervention evidence and all random-control summaries remain available in `run/rows.jsonl` and `audit.json`.

## Frozen boundary

- Selection archive SHA256: `a32edd039c25bf3d01086730e20e3f92fb68cfc4e8ed37d673806bb802624c49`.
- Source freeze v2: `594a700c0fc1eb19b7e269a60531a3fd9fbfd6ca3a55d7dce3d90a431e01efd6`.
- Prior train archive: `d5d09ed7245bbbf82c8baeaee8e91a17407044c745862c41438b5cb466adf640`.
- Stage plan recorded before model loading: 1,300 forwards, two CPU threads, 900 cooperative seconds. Execution was on shared CPU, not a timing benchmark.
- Local source freeze and independent review preceded this inference. Publication may follow; this was not prior public registration.

This is validation-only preservation, not a held-out result or proof of causal necessity. No test model inference has occurred at this checkpoint. Test execution must use this exact selected archive digest and unchanged v2 source, only after independent review and separate clearance. A successful result does not rehabilitate the failed arithmetic task or establish general knowledge/brain correspondence. Complete binary distribution is necessary for the full baseline evidence archive.
