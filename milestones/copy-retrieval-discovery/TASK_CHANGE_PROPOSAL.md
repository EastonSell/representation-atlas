# Proposed task change: preservation and causal sensitivity of copying subspaces

**Proposal only. Not an executed, frozen held-out study. Parent review is required before a new dataset freeze or any held-out/causal execution.** The original arithmetic gates remain failed and immutable. Discovery does not establish arithmetic, general retrieval, semantic truth, induction circuits or brain alignment.

## Question and primary representation

Can a train-fitted leading-variance subspace be removed from a pre-answer residual state while preserving a demonstrated, narrow word-copying behavior? Does removal affect copying differently from rank- and energy-matched random perturbations?

Use the selected plain completion template exactly: `Repeat the word.\nWord: {word}\nRepeated word:`. Do not add a chat template or optimize the prompt on validation/test. The target is one exact next token, including its leading-space tokenization; this is not complete-response generation. A ceiling-level copying baseline motivates **preservation and causal sensitivity**, not an accuracy-improvement claim.

Use only the last prompt position, before any answer token is appended. Fix the intervention site to the output residual stream of zero-based decoder block 14, the fifteenth of this model's 30 blocks, before block 15 executes. This is an a priori midpoint choice, not a localization result. Never append a correct/foil answer and label the resulting difference a truth direction. No supervised target-word probe is proposed: lexical-disjoint labels would make a trained word classifier's held-out interpretation problematic.

## New inputs and separation

After review, author and publish a fixed pool of 80 distinct lowercase ordinary English words, excluding all 12 discovery words. Publish the raw pool and its provenance before model inference. Tokenizer-only preflight must verify each word is a single answer token, each complete prompt has the same length, and swapping paired words changes exactly one source-position token. Reject an incompatible pool before inference rather than filtering on model outcomes. Any revised pool gets a new documented freeze; this proposal has not generated one.

Sort the accepted words by SHA-256 of UTF-8 `copy-subspace-v1|` concatenated with the word. Assign the first 40 words to training, next 20 to validation, last 20 to test. Pair adjacent words within each split and include both orientations: prompt copies A with foil B, then prompt copies B with foil A. This gives 40/20/20 distinct prompts and 20/10/10 disjoint unordered groups. Each word is correct once and a foil once within its split. No word, exact prompt or unordered pair may overlap splits or discovery. Shared template structure is intentional; this tests lexical transfer within a fixed format, not template transfer.

Before inference, freeze the complete dataset, tokenizer audit, exact IDs, protocol, source, dependency versions and model hashes. Show answer-only lexical-priority baseline at 50% paired accuracy and zero entirely-correct pairs. Report exact/commutative overlap as zero. These are metadata controls, not model evidence. Independent means new instances with disjoint lexical content; it does not imply independent pretraining exposure or random population sampling.

## Train gate and fitting

Run the new training split only. Require at least 36/40 strict correct-versus-foil logit margins, 36/40 unrestricted greedy target tokens, and 18/20 groups with both margins positive. Ties and wrong-format output fail. These are prospective operational engineering thresholds, not power-derived statistical claims. Stop without held-out inference if any threshold fails. Archive every row including failures; never fit only successful examples.

If the gate passes, collect all 40 pre-answer states at the fixed site. Fit the mean and PCA/SVD basis using training states only. Freeze all fit arrays and source hashes. Candidate removals are k = 0, 1, 2, 4. For h and training mean mu, the intervention is h' = h - Q_k Q_k^T (h - mu). It preserves the mean component rather than subtracting the entire mean. Training centering itself is a geometric coordinate choice, not evidence of a behavioral improvement. No test-fitted centering, layer search, vocabulary-driven axis, or answer-conditioned state is allowed.

## Controls and validation selection

For each nonzero k, create ten fixed orthonormal random k-dimensional bases by QR of standard-normal arrays with NumPy seeds 20261022 through 20261031. Fix these before held-out inference; retain their hashes. Run two controls per basis:

1. Unscaled random-subspace removal, matched in rank but not generally in energy.
2. Energy-matched random perturbation: rescale the random projected delta at each example to the norm of that example's PCA-removal delta. This is a pointwise perturbation, **not** an orthogonal projection. Its norm uses the current state but no label, outcome or fitted test statistic. If a random delta norm is below 1e-12 while the PCA delta is nonzero, mark that control invalid and stop rather than resampling a favorable direction. Zero PCA delta implies an exact no-op.

Keep an untouched no-hook forward pass and a no-op-hook pass for every example. Their full float32 logit vectors must agree within absolute 1e-6 before interpreting an intervention. Run eval mode, CPU float32, eager attention and no KV cache under the pinned runtime. Target and foil text are not fed during these forwards.

On validation only, select the largest k whose greedy accuracy is no more than 0.05 below untouched accuracy and whose mean target-minus-foil logit margin is at least 0.90 times the untouched mean. Require untouched validation mean margin > 0; otherwise stop and report a failed validation baseline without examining test. Select k = 0 if no nonzero k qualifies. Archive the complete validation selection record and every candidate/control outcome before opening test. Random controls never choose the winner. These preservation tolerances are operational, not proof of equivalence.

## Once-only test and interpretation

After a separate reviewed execution checkpoint, evaluate test once with the frozen selected k and its fixed controls. Do not select a different k, layer, prompt, threshold or random seed on test. If k = 0, report that no nonzero preserving intervention was selected; do not present a vacuous no-op as causal progress.

Primary endpoint: paired change in target-minus-foil logit margin relative to the untouched forward pass, averaged first across the two orientations of each unordered group. Also report unrestricted exact-token accuracy change, completely-correct groups, target probability, removal energy, residual norm and per-row results. Compare PCA removal to each fixed random control and to the across-seed average; do not pick the best seed. With only ten held-out groups, report descriptive paired means and all group-level values rather than treating rows/controls as independent statistical replications.

Archive full untouched logits, scalar intervention outcomes, pre-answer states, train fits, random bases, exact hook/site specification, selected hyperparameter, raw IDs and artifact hashes. The implementation must pass mock/state-shape, no-op, all-zero-removal, missing-ID, split-overlap and overwrite-refusal tests before inference.

Preservation supports only robustness to this particular subspace removal on this task/site. A larger effect than matched controls supports targeted causal sensitivity under this intervention, not exclusive necessity, a complete circuit, semantic truth or general knowledge. A null effect does not prove the subspace carries no useful information. No steering or brain mapping claim follows. More ambitious source-token interchange patching is outside this proposed checkpoint.

## Primary-source grounding

- Olsson et al., [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html), motivates studying simple copying but defines induction using prefix matching plus copying. Our observed word repetition alone does not identify an induction head or mechanism.
- Zhang and Nanda, [Towards Best Practices of Activation Patching in Language Models](https://arxiv.org/html/2309.16042v2), shows that corruption methods and metrics can change conclusions. This motivates explicit interventions, logit-difference endpoints and distinct controls; our PCA-removal experiment is not a reproduction of their patching study.
- [Transformers 4.46.3 LLaMA API](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/llama) documents causal-LM logits and hidden-state interfaces. Hooks must additionally be checked against the installed pinned source before implementation. A final normalized hidden state and a decoder-block residual are not interchangeable sites.
