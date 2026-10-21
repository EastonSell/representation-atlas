# Implementation plan
1. Write unit tests for train-only centering, translation distance invariance, pair permutation invariance, PCA orthogonality, CKA rotation invariance, invalid inputs.
2. Implement NumPy transformations, synthetic paired generator, and held-out direction evaluation.
3. Run three seeds at dimensions 8, 32, 128 under low/high task-signal variance. Save raw per-run metrics and provenance.
4. Summarize measured results without generalizing beyond this generator. Document real-activation and human-brain-dataset validation requirements.
5. Implement a no-download real-data MLP experiment, with explicit split, model configuration, dataset hash, prediction agreement controls and optional sklearn dependency. Save metrics and train/test indices for exact reproduction. Add tests for hidden activation extraction and centroid translation invariance.
