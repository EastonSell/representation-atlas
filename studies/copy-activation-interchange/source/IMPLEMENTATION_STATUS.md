# Source/data review status

This snapshot is intended for source/data review before any pretrained output.
Tokenizer eligibility: 219 examined, 219 eligible, 0 rejected, 64 selected,
32 disjoint lexical pair groups, 13 tokens per prefix. The candidate pool and
92 historical exclusions are retained unchanged under provenance. The earlier
lexical removal of “coat” is recorded in that original pool's provenance.

Implemented components include strict deterministic data validation, offline
snapshot receipt validation, raw decoder-output hook capture and replacement,
full-vocabulary float32 evidence, append-only attempt accounting, distinct
baseline and patch engines, numerical identities, fixed competence gate,
paired descriptive bootstrap, independent artifact/summary audit, and a
scientific report renderer. Synthetic tests use no pretrained weights.

Known review boundaries:

- V2 integrates reviewed single-process supervision. Model execution still requires
  source/data review, a reviewed lightweight canary and exact-stage clearance.
  Resource guarantees and limitations are specified in SUPERVISION.md.
- Component and synthetic-panel tests are not a pretrained end-to-end run.
  Numerical reproducibility at tolerance 1e-6 remains an empirical gate, not a
  guaranteed model property. Failed controls must retain evidence and stop.
- The current report renderer supplies the full cell table and machine-readable
  detailed summaries, not plotted figures. No result narrative is prewritten.
- The independent archive auditor requires explicit accessible source and model
  paths to verify their actual bytes; digest cross-binding alone is insufficient.
- V2 has a new source manifest. Keep both V1 and V2 unchanged as review evidence.

No execution-clearance records, private communications or operational handoffs
are part of this scientific source package. Nothing has been published.
