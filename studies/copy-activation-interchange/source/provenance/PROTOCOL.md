# Copy-target interchange interventions: prospective design v0.1

Status: proposal only. No new model inference, tokenizer preflight, training, implementation, or publication has been performed for this study. Parameters below are proposed for review, not a claim of executed or registered research. The candidate pool is authored now and remains subject to review before its prospective freeze.

## Question and improvement over the previous experiment

On fixed-format word copying in SmolLM2-135M-Instruct, which sampled layer/position residual replacements causally redirect next-token preference from a recipient word to a donor word?

The prior study removed four of 576 train-PCA directions at block 14's final prompt position. Every test condition retained 20/20 exact predictions; the PCA mean margin change was −0.01443, inside the energy-matched random-condition range. This is a small-perturbation preservation observation, not evidence that target information is absent. Its 80 study words and 12 discovery words are now inspected and must all be excluded. They are historical development data, not fresh confirmation data for this proposal.

Instead of asking whether a small subtraction breaks an already easy task, interchange interventions ask whether replacing an actual residual from another natural prompt changes *which* answer is preferred. This tests counterfactual causal influence and specificity, without fitting a probe or choosing a perturbation to preserve accuracy. A null remains informative only about this exact replacement, these sites and this task; redundancy or interactions can mask effects.

## Model and task, fixed before outputs

- HuggingFaceTB/SmolLM2-135M-Instruct, revision 12fd25f77366fa6b3b4b768ec3050bf629380bac; local safetensors only, offline, trust_remote_code=False.
- Inspected configuration: 30 decoder blocks, residual width 576, vocabulary 49,152, LlamaForCausalLM. Pin torch 2.5.1+cpu, transformers 4.46.3, NumPy 2.3.5; record Python, OS, CPU and package/source hashes.
- CPU float32, eager attention, eval/inference mode, batch size one, two CPU threads, no KV cache, no generation, no gradients, no optimizer.
- Exact prefix: `Repeat the word.\nWord: {word}\nRepeated word:`. Score precisely the next token, not an entire answer or termination behavior. No chat template or automatic BOS.
- 64 new words, 32 disjoint unordered pairs, both orientations. No fitting/selection split is needed because there is no parameter fitting or outcome-dependent selection. This is one prospective fixed-panel evaluation, not a random population sample or blinded experiment.

## Prospective lexical freeze

1. Freeze candidate_words.json, the 92-word exclusion list, this protocol, source revision and deterministic selection policy before ANY new model output. Record UTC and SHA-256; do not claim public preregistration unless publication actually preceded execution.
2. Tokenizer-only filtering is allowed after source review. Preserve every candidate and rejection reason. Exclude exact lowercase duplicates, all historical words, nonalphabetic tokens, and words for which `" "+word` is not one token or appending that answer changes existing prefix token IDs.
3. Require the in-prompt word to occupy exactly one token and replacement to preserve all other prefix IDs, sequence length and source position. Use the fixed prompt tokenization for the first eligible word as reference; reject any mismatch. All pair directions must have distinct answer IDs and exact append-only answers in both recipient prefixes. Reject, do not silently repair.
4. Sort eligible words by ascending SHA-256 of UTF-8 `copy-patch-v1|` + word; first 64 constitute the panel. Pair adjacent words in that ordering; orientation zero uses the first as recipient. There is no frequency, logit, accuracy, embedding, or effect filter. If fewer than 64 qualify, stop. Any new pool is a separately versioned, pre-output amendment, never a response to competence failure.
5. Freeze selected words, full prompt IDs, answer IDs, source position, final position, pairing, tokenizer hashes, rejection ledger and dataset hash before baseline inference. All later baseline failures remain in raw data and analysis. No replacement words after outputs.

## Interventions and exact controls

Let recipient prompt contain A and donor prompt contain B; their token IDs differ at the source-word position p only. Reverse A/B for the second orientation. Let f=L−1, and q=p−1 (guaranteed earlier than p). Patch *post-block, pre-next-block residual*, after both attention and MLP residual additions; blocks are zero-indexed.

Fixed blocks: 0, 5, 10, 14, 19, 24, 29. At each block independently, run exactly six conditions, always starting from the original recipient prefix:

1. donor_word: replace residual at p with donor residual at p.
2. donor_final: replace residual at f with donor residual at f.
3. donor_prefix_negative: replace residual at q with donor residual at q.
4. identity_word: replace residual at p with cached recipient residual at p.
5. identity_final: replace residual at f with cached recipient residual at f.
6. donor_all: replace entire sequence residual with donor sequence residual.

No cumulative layer patches, scaling, random directions, windows, head selection, or tuned strengths. The donor is another competent-or-incompetent natural prompt, not Gaussian noise; mixed states can nevertheless be out of distribution.

Architectural controls, NOT discoveries:
- q precedes the differing word, so causal masking implies identical donor/recipient q states. This negative control is a causal-mask/alignment check, NOT an equally strong perturbation or a generic irrelevant-position test.
- Full-sequence donor replacement must reproduce donor final logits at every block when masks/positions are identical and caches are disabled. This is implementation identity, not evidence of localized mechanism.
- At block 29, donor_final must reproduce donor final logits because only tokenwise norm and linear readout remain. donor_word must leave final logits unchanged since p != f and no later token mixing remains. Neither outcome is mechanistic localization.
- Identity replacements must reproduce recipient logits. Read-only hooks must not alter baseline logits.
- Use full-vocabulary max-absolute-error tolerance 1e-6 for these checks, fixed prospectively. Report exact max errors; a failure invalidates mechanistic claims and stops the run. Do not relax tolerance after seeing failure; diagnose in a separately versioned implementation amendment.

Blocks 0/5/10/14/19/24 × word/final are the 12 substantive cells. Effects concern where a replacement can influence the answer, not the location where a feature is uniquely computed or stored. Residuals accumulate earlier computations, so positive block effects are not attribution to that block's attention/MLP.

## Stages and competence gate

A. Review protocol, candidate selection, architecture and budget. No model forward during this stage.
B. Implement and test with synthetic tensors/mock models; tokenizer-only audit; freeze code/data. Review this freeze before inference.
C. Baseline gate: for all 64 prefixes, first untouched forward (full final logits), then read-only-hook forward caching all seven full sequence states and verifying no-hook equivalence. Maximum 128 forwards. Record all outcomes before deciding the competence gate; a technical/nonfinite/numerical-identity failure aborts immediately and marks the stage incomplete.
D. Gate passes only if >=58/64 exact greedy answers, >=58/64 strictly positive own-versus-paired-foil margins, and >=27/32 pairs have BOTH exact greedy answers AND strictly positive margins in both orientations. These are feasibility thresholds, not inferential results; ties fail. Also all 64 numerical identity checks must pass. If gate fails, stop and report complete baseline including failures. Do not alter words, prompt or thresholds.
E. Review complete gate and immutable artifacts. If accepted, execute exactly 2,688 patch forwards (64×7×6), including failures in the same panel. Review correctness controls before interpreting results. Do not infer missing cells if interrupted.
F. Independent no-inference audit re-scores every stored final-logit vector, validates coverage/hashes/control identities, then produces all cell results. Publication is a separate decision.

No automatic baseline-to-patching transition. A source bug discovered after outputs requires a visible amendment and retained old evidence; the data are no longer prospectively untouched for any revised hypothesis.

## Outcomes and denominators

For direction A recipient, B donor, define m(z)=z[B]−z[A]. Primary raw transfer Δ=m(z_patch)−m(z_A). Also retain own-target margin z[A]−z[B], donor foil margin, both individual logits and probabilities, top-1 ID, top-1 logit and full-vector logits. Compute probabilities in float64 log-sum-exp from stored float32 logits.

For each cell report:
- Unconditional donor replacement rate: fraction of all 64 with argmax=B.
- Genuine switch rate: numerator argmax_baseline=A AND argmax_patch=B, denominator baseline argmax=A; report numerator and denominator explicitly.
- Off-target output rate: argmax_patch not in {A,B}; off-target move rate: baseline argmax in {A,B} and patch argmax outside, denominator those baseline in {A,B}. Also give counts/64 for transparency.
- Recipient retention, positive donor margin and exact ties; all computed separately, never equating pairwise preference with greedy correctness.
- Mean Δ and median Δ, pair-level distributions, donor/recipient probabilities, full-distribution KL(p_A || p_patch), requested/realized residual delta norm, recipient norm and norm ratio.
- Secondary normalized transfer Δ / (m(z_B)−m(z_A)) ONLY for pairs satisfying the competence definition above AND denominator >=1 logit for both directions. Eligibility comes solely from fixed baseline facts. Publish denominator values and exclusion counts; no clipping [0,1], do not hide overshoot/negative values. Raw Δ remains primary across all pairs. A normalized ratio is not a fraction of all task computation explained.

Average both directional outcomes within each unordered pair first. N=32 independent lexical groups for descriptive uncertainty, not N=64 or N=2,688. Pair members are unique, but investigator-selected words do not justify population-wide English claims.

Use fixed RNG seed 20261022 and 10,000 bootstrap resamples of the 32 whole groups, retaining all directions/cells together. Report pointwise percentile 95% intervals labeled descriptive. To avoid cherry-picking the largest of 12 cells, additionally report an approximate simultaneous band for mean raw Δ: compute per-cell SE across groups; for each group bootstrap compute max over 12 cells of |mean*−mean| / original_SE; take its 95th percentile c and report mean ± c×SE. Mark zero-variance cells separately (degenerate empirical interval, not population certainty). This resampling approximation with only 32 groups is not an exact randomization test. No p-value fishing, winner-only report, or data-dependent layer expansion.

Primary evidence of localized transfer is a positive raw-margin effect with a simultaneous band excluding zero at a nontrivial cell, valid controls, and its reported donor/off-target profile. Do not require a significant replacement rate or claim necessity from a replacement. If no cell meets this description, publish the bounded null map and uncertainty. If all cells transfer strongly, report coarse non-specific localization; no narrow circuit claim. Differences between cells are descriptive unless a future contrast is separately frozen.

## Novelty and limits

Activation patching and counterfactual logit differences are established methods. This is not a new algorithm, a neuroscience result, or comprehensive circuit discovery. The contribution is an auditable, low-resource, lexically fresh localization map with bidirectional controls, full-logit evidence, explicit endpoint identities, and an honest test beyond the previous accuracy ceiling. No neural data means no layer-to-brain-anatomy claims. It does not test arithmetic, truth, unrestricted copying, other templates/models, or full-answer generation. Strong transfer is intervention-specific causal sufficiency in this context, not exclusive necessity, semantic purity or on-manifold proof.

## Primary sources and directly inspected implementation

- Zhang and Nanda, “Towards Best Practices of Activation Patching in Language Models: Metrics and Methods,” https://arxiv.org/abs/2309.16042 ; primary text recommends counterfactual token replacement and logit-difference metrics and documents sensitivity to metric/corruption/site choices. Verified through paper passages on 2026-10-22. This motivates design choices, not a claim that mixed patched states are always in distribution.
- Meng et al., “Locating and Editing Factual Associations in GPT,” https://arxiv.org/abs/2202.05262 ; verified primary abstract on 2026-10-22. Precedent for causal tracing, not evidence of equivalent copying mechanisms in this model.
- Transformers 4.46.3 Llama implementation: https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py . Local installed source inspected: LlamaDecoderLayer returns post-MLP residual as tuple element zero; LlamaModel applies final norm AFTER the last layer. Consequently hidden_states[30] is normalized and is NOT the raw block-29 hook state. Capture raw block outputs directly and test this distinction.
