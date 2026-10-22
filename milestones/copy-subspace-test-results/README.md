# Held-out copying: small train-PCA removal preserved exact predictions

**Rank4 was selected for preservation on validation, and every test condition hit the accuracy ceiling.** The locked k=4 intervention preserved all20/20 held-out exact next-token predictions. Untouched baseline, PCA removal and all twenty random-control conditions also retained 20/20 positive correct-versus-foil margins, zero ties, and 10/10 complete two-orientation groups under both the separate margin and greedy definitions.

| Condition | Greedy exact | Mean margin | Mean realized residual perturbation norm |
| --- | ---: | ---: | ---: |
| Untouched | 20/20 | 7.258719801902771 | 0 |
| Remove top four train PCs | 20/20 | 7.244289445877075 | 5.190251842353183 |
| Rank-matched random, ten seed range | 20/20 each | 7.254780650138855–7.262434291839599 | 0.6012865786377513–0.7960906610444292 |
| Energy-matched random, ten seed range | 20/20 each | 7.235829687118530–7.293977069854736 | 5.190251749535366–5.190251799216999 |

The PCA mean margin change was −0.0144303560, about −0.20% of baseline. Across the ten paired groups, six changes were negative and four positive, with range −0.1414063 to +0.0805969. The PCA mean falls inside the range of the energy-matched random-seed means. Averaging each random-control type over seeds and then groups gave mean changes +0.0003452 (rank) and +0.0001568 (energy). These are descriptive summaries, not a statistical significance test or proof of equivalence.

## What this does and does not establish

For these twenty lexically held-out, explicitly instructed word-copying prompts, removing the centered training top-four-PC component at the last prompt position of decoder block14 did not change exact next-token accuracy. This is **removal of four directions from 576**, not successful compression to four dimensions. No loss of exact performance was observed under this specific intervention/site/task; this is not a general conclusion about PCA necessity or unimportance. The result does not establish that those directions carry no information, that every redundant pathway is unnecessary, or that PCA isolates a semantic truth representation.

Because k was explicitly selected for validation preservation and every test condition was at the accuracy ceiling, this is not a stringent search for a necessary subspace and cannot establish equivalence between PCA and random controls.

Energy-matched controls matter because rank-matched random removal changed substantially less residual energy. Their descriptive range provides no uniquely damaging PCA effect here. We do not generalize from ten paired test groups to unrestricted copying, arithmetic, general knowledge, other layers/models, or biological brain representations. The historical arithmetic failures remain failures. No full-answer generation or termination criterion was tested.

## Locked protocol and coverage

- This test used the independently reviewed selected k=4 archive `a32edd039c25bf3d01086730e20e3f92fb68cfc4e8ed37d673806bb802624c49` and unchanged v2 source freeze `594a700c0fc1eb19b7e269a60531a3fd9fbfd6ca3a55d7dce3d90a431e01efd6`.
- PCA and mean were fitted exclusively on training states after the training gate passed; random bases were generated independently from fixed seeds. Validation chose k by the unchanged predeclared accuracy/margin rule. Test words/groups were disjoint from train, validation and discovery.
- No thresholds, k, prompt, fit, layer, scoring rules or source were changed after selection. Test was run only after independent selection review and separate execution clearance.
- Exactly **460 forwards** completed:20 untouched,20 no-op,420 interventions (20 rows ×21 fixed conditions). The plan set two CPU threads and600 cooperative seconds before loading the model. Timing on shared CPU is not a performance benchmark.
- All20 no-op checks had exactly zero logit/state error. Each condition has20 distinct rows but only ten paired groups; 420 interventions are not420 independent lexical test cases.
- All210 condition/seed/group paired changes and20 random-seed-averaged group changes are archived, without filtering favorable cases.

## Evidence and reproducibility

Test archive manifest SHA256: `e2fee54c17626190b0c5b86d36c29ba099e650f1b4dfd95b0fbae30fa3c9c313`.

`run/` contains metadata, pre-run budget, incremental raw scalar rows, result,20 full untouched-logit/state NPZ arrays, the unchanged training fit NPZ, and per-file SHA256. `audit_test.py` is a separate no-inference audit. With the tested runtime, it verifies source/artifact hashes, exact IDs and coverage, upstream selection, unchanged fit bytes,20 untouched predictions/probabilities/margins, all420 perturbation norms and all grouped summaries. Requested-norm reconstruction error was below5.4e-15; realized float32-norm reconstruction error was zero. Default audit output is stdout; optional `--out` refuses existing files.

**Full intervention logits are not archived.** Intervention scores are recorded directly by the frozen driver; their grouped results are independently recomputed from scalar rows, not independently rescored from full logits. The untouched baseline scores and requested/realized norms do have independent numerical reconstruction. This limitation must accompany the result. Complete baseline reproduction requires distributing the binary NPZ files, not just their hashes.

Source freezes and independent reviews preceded their corresponding model stages locally. Public publication may follow; no earlier public registration is claimed. This is a bounded preservation result under small, validation-selected perturbations, plus a working reproducible intervention pipeline, not evidence that the original broad geometry goal is solved.
