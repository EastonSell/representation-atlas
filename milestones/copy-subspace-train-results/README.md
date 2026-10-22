# New copying training gate passed; no held-out or intervention result

Using the locally frozen, independently reviewed v1 source, the model passed all fixed training criteria:

- **40/40** correct-versus-foil logit margins strictly positive; zero ties.
- **40/40** unrestricted greedy exact next-token predictions correct.
- **20/20** counterfactual groups had positive margins in both orientations.

The group criterion was margin-only. A separate post-hoc audit also found 20/20 greedy-exact groups, but that was not an added gate requirement. These are 40 distinct new word-copying prompts and 20 disjoint paired groups, unlike the overlapping 12-word discovery. The frozen validation/test words remain unqueried.

Exactly **80 forwards** were completed, including untouched/no-op verification for every training row. All no-op maximum logit/state errors were zero. The 40 unmodified float32 full-vocabulary logit vectors and 576-dimensional pre-answer residuals are stored as binary NPZ. Every recorded greedy token, probability and margin was independently reconstructed from those arrays. Margin minimum/mean/maximum were 4.7932 / 7.6580 / 11.4346.

PCA and mean were fitted **only after** the gate passed, using all training states. The predeclared random bases were generated independently from fixed seeds, not fitted to training data. Independent reconstruction matched the mean exactly and the top-four PCA projector to maximum error below 4e-18; random bases matched exactly. The first four training variance fractions were 27.72%, 10.73%, 10.08% and 8.28%. These describe the training covariance, not causal importance or preservation.

There was **no validation/test inference, selected k, nonzero intervention or causal result**. No weights were trained. This confirms a narrow training copying capability, not generalization and not recovery of the failed arithmetic tasks. An exact next token does not establish full-response termination or arbitrary-format copying.

## Integrity and chronology

- [GATE_CRITERIA.md](GATE_CRITERIA.md) was written before execution to clarify the unchanged margin-only group definition.
- Training source freeze: `63f1cac069d8a319e9603517ddfe0dbc3663843205789ed42edbcc86caf10b7a`.
- Training archive manifest: `d5d09ed7245bbbf82c8baeaee8e91a17407044c745862c41438b5cb466adf640`.
- `run/` contains incremental rows, binary baseline arrays, fit NPZ, runtime metadata, budget plan, result and per-file hashes.
- `audit.json` contains independent reconstruction and explicit stage boundaries.

The freeze and review occurred locally; publication may follow. No claim of earlier public registration is made. Full binary files are required for a complete archive and independent reconstruction; metadata-only publication must be labeled as incomplete binary distribution.

The prospective [v2 amendment](../copy-subspace/AMENDMENT_V2.md) addresses later-stage review findings without changing this run's source, data or fit. Validation selection needs a fresh review of that source. Held-out test execution remains conditional on a separately reviewed frozen selection digest.
