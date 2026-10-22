# Training-only gate clarification, recorded before execution

Use the independently reviewed source freeze with SHA-256 `63f1cac069d8a319e9603517ddfe0dbc3663843205789ed42edbcc86caf10b7a` without modification.

All three frozen requirements must hold:

1. At least 36/40 training rows have strictly positive correct-minus-foil logit margin. Ties fail.
2. At least 36/40 training rows have the correct token as the unrestricted vocabulary argmax.
3. At least 18/20 unordered counterfactual groups have strictly positive margins in **both** orientations.

Requirement 3 is margin-only. It does not require both greedy predictions in each group to be correct. A report must not describe these as fully greedy-correct groups. This clarification restates the approved proposal and current source implementation, not a change to thresholds or scoring.

This run executes only the train command: 80 forwards maximum, two CPU threads, cooperative 300-second wall budget. It caches baseline logits/states and no-op checks incrementally. Fitting is conditional on passing the gate. No validation/test inference, nonzero intervention or selection is authorized in this run.

Known later-stage issues must be addressed in a prospective, separately frozen source version before those stages: checking validation baseline positivity before interventions, adding complete-group summaries, and recording realized float32 perturbation norm alongside requested float64 norm. The reviewed training source remains immutable.
