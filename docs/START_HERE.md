# Start here: what a representation picture can and cannot tell you

**For a curious Python/ML learner.** A representation is a list of numbers used to describe an item, such as an image or a word in context. This repository asks when manipulating those numbers changes geometry, preserves a task, or changes a model's answer. Those are different questions.

The short demo below needs only Python's standard library. It illustrates mathematical possibilities; it is not new benchmark evidence or a pretrained-model experiment.

![Three illustrative geometry panels: translating points keeps their separation while changing their angle from the origin; projection can merge differently labelled points; a translated state changes a fixed downstream score unless the readout is compensated.](images/geometry-learning.svg)

## 1. Work through a translation

Take A = (1, 0) and B = (0, 1). Their straight-line, or **Euclidean**, distance is √2. Their **cosine similarity**, which compares the directions of arrows from the origin, is 0.

Add (2, 2) to both. A becomes (3, 2), and B becomes (2, 3). Their distance is still √2, but cosine similarity is now **12/13, approximately 0.923**. The pair got no closer. Both arrows now point in similar directions from the same origin.

Why does distance stay fixed? In the difference (A + t) − (B + t), the shared translation t cancels. The same holds for every pair. **Centering** subtracts one shared mean vector; it is also a translation. For held-out comparisons, this repository estimates the mean on training data, not the test set.

Run from the repository root:

```sh
python -B examples/learning/geometry_demo.py
python -B -m unittest discover -s examples/learning -v
```

Read [the script](../examples/learning/geometry_demo.py) and [tests](../examples/learning/test_geometry_demo.py). They check all pairwise distances for a small fixture, the changed cosine, a fixed readout, and projection loss. Cosine is undefined for a zero vector; the demo rejects that case.

### A coordinate description is not a free intervention inside a fixed model

Suppose a downstream unit reads a state using score = x1 − 0.5. The original A/B scores are +0.5 and −0.5. Translating their states by (2, 2), **without changing this unit**, gives +2.5 and +1.5. A zero-threshold decision changes for B even though pairwise distances are preserved.

To express the same original computation in shifted coordinates, we would also change the bias from −0.5 to −2.5. The original scores return. That compensation is a second operation. Moving internal activations while holding the rest of a trained model fixed is an intervention, not merely relabelling coordinates. Translation invariance of distances does not promise invariant model behavior.

## 2. Projection can delete the answer

A **projection** keeps some directions and discards others. Project (1, 1) and (1, −1) onto the horizontal axis: both become (1, 0). If the label was “is y positive?”, the answer is gone. No classifier using only the projected point can perfectly distinguish these two labels.

**Principal component analysis (PCA)** finds directions of large variation in training data. Large variation is not synonymous with nuisance. Discarding a leading principal component can discard task information. The demo's projection is chosen by hand, not fitted PCA; it illustrates loss, not the outcome of a PCA benchmark.

The [original synthetic results](../README.md) deliberately include a high-signal-variance control: mean held-out pair-ranking accuracy (correct score ordering of positive/negative pairs, with half credit for ties) falls from 1.0000 without removal to 0.5318 after removing one leading PC. That constructed regime diagnoses a failure mode, not expected performance on all embeddings.

The [learned-digit experiment](real-data-results.md) supplies a measured counterpart: small feedforward networks (multilayer perceptrons, or MLPs) averaged 97.22% test accuracy. A separate nearest-centroid probe, which picks the class with the closest average hidden vector, scored 95.49% with centering only, 91.05% after removing one leading PC, and 38.02% after eight. Centering alone left this probe's predictions unchanged because both points and centroids translate together. The MLP's accuracy and the probe's accuracy are different measurements; the capped training emitted recorded convergence warnings.

Conversely, the [held-out copy-subspace test](../milestones/copy-subspace-test-results/README.md) preserved 20/20 exact predictions after removing four training PCs, as did every random control. Four directions were removed from 576, not retained as a four-dimensional representation. The removal count was selected for preservation on validation. This is a bounded preservation result, not proof that PCA directions are useless or that all conditions are equivalent. Full intervention logits were not archived; the report explains what can and cannot be independently reconstructed.

## 3. From looking at vectors to swapping a model's state

Geometry describes relationships. A **causal intervention** changes a model's internal state and observes its later output. The [copy-target interchange study](../studies/copy-activation-interchange/report/REPORT.md) tests one such intervention in the pinned SmolLM2-135M-Instruct checkpoint.

Here is the donor/recipient story, using an actual paired word group:

1. The recipient is asked to copy “shadow”; its paired donor is asked to copy “office”. Both runs have the same prompt form.
2. At a selected decoder block, take the donor's **residual stream**, the vector passed between processing blocks, at one chosen token position.
3. Substitute it at the same block and position in the recipient. Keep the recipient's other states, then continue computation.
4. Ask whether the next predicted token genuinely changes from the recipient word to the donor word. A change in a continuous token score alone is not a genuine switch.

The **source-word position** is where the word to copy appears. The **final prompt position** is where the model forms the next-token prediction. Seven block outputs were sampled, numbered from zero; the experiment did not inspect every layer transition.

### What distinguishes explanations?

- Identity replacement puts the recipient's own state back. It checks that the intervention machinery itself does not alter logits, the model's token scores.
- Donor replacement at a shared causal-prefix position before the differing word is a negative control: that prefix has not seen which word differs.
- Whole-state donor replacement checks that continuing the donor state reproduces donor output.
- Comparing source-word with final-position replacement asks where a donor substitution is effective at the sampled stages. It does not by itself prove necessity, exclusive storage or one unique circuit.

### What actually happened?

Untouched baseline next-token copying was correct on 63 of 64 orientations, from 32 disjoint word pairs. The failure is retained. At sampled block 24, final-position replacement caused **60/63 genuine donor switches** among baseline-correct orientations; source-word replacement caused **0/63**. Earlier sampled source-word replacements were strongly effective. At block 19, a small final-position score effect caused no donor switches. Scores and discrete decisions must not be conflated.

At the final block, some endpoint outcomes follow from the architecture itself: changing the final-position state reproduces donor logits, while changing only an earlier position has no later mixing step to reach the output. These are checks, not discoveries of a circuit.

The supported finding is a **coarse map of interchange sensitivity** for one model checkpoint, prompt template and selected word panel. Replacing a whole vector can change many features simultaneously. The study does not establish exclusive storage, a unique circuit, a precise transition block, brain correspondence, or generalization to other tasks/models. The historical arithmetic capability gates remain failures; copy-task success does not repair them.

## 4. Choose an inspection path

- To learn the geometry: run the tiny offline demo above, then use [the root README](../README.md) and [setup guide](../SETUP.md) for the original synthetic/digit commands. Use fresh output paths; do not overwrite committed results. Original numerical experiments need their documented dependencies, unlike the tiny demo.
- To inspect copying: start with [the scientific report and full condition map](../studies/copy-activation-interchange/report/REPORT.md), then [download/reconstruction instructions](../studies/copy-activation-interchange/GITHUB_DOWNLOAD.md). Byte-for-byte reconstruction of retained evidence is not new model inference. Weights and the numerical runtime are separate.
- To follow the full sequence, including unsuccessful gates and controls, use [MILESTONES.md](../MILESTONES.md). Historical source freezes can say “unrun”; read the later linked results for current status rather than editing those records.

## Useful next questions

1. Which conclusions survive different similarity measures and compensated versus fixed-readout transformations?
2. Does the same interchange map hold for new prompt templates and a separately held-out word panel?
3. Which smaller, controlled perturbations separate correlated information from information the computation needs?

These are future tests, not claims this repository has already established.
