# Prospective activation-patching design

This is a review package, not an implemented or executed experiment.

- PROTOCOL.md: hypotheses, frozen-design proposal, exact controls, competence gate, pair-level statistics, claims and primary sources.
- ARCHITECTURE.md: proposed module boundaries, numerical hook contract and required tests.
- RESOURCES.md: 2,816-forward maximum, resource forecast and stop policy.
- SCHEMAS.md: raw evidence and independent-audit contracts.
- candidate_words.json: authored, untested candidate pool; not yet frozen or tokenizer-filtered.
- historical_exclusions.json: all 92 previously inspected words.

The main scientific correction is to separate causal transfer at earlier positions/layers from architectural replay identities. Full-sequence replacement and final-position replacement at the final layer cannot by themselves be claimed as a discovered copying mechanism. Retaining complete intervention logits enables a stronger independent audit than scalar-only output.

The next permitted checkpoint is design review. No inference result exists in this package. Candidate selection and code implementation remain pending; no model output was consulted to construct the pool.
