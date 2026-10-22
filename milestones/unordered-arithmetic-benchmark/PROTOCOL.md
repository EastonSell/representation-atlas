# Frozen arithmetic construction protocol v1

Construction date: 2026-10-22. Parent public checkpoint: `9422c08f6e65948405e0bd33e3630df86f0e49c3`.

## Scope and chronology

This is an original synthetic benchmark construction, not model evidence. The dataset, generator, integrity audit, baseline controls, tests and this protocol are frozen together before any model inference on these new prompts. Prior exploratory 0–9-operand model results and the subsequent split audit informed the design. Therefore this is **not** a blind externally preregistered study or an untouched external final test. Publication hashes record bytes, not an independently timestamped preregistration.

## Dataset and split

Generate every ordered pair `(a,b)` with integer operands 10 through 19 inclusive, giving 100 rows and 55 unordered arithmetic problems. IDs are exactly `a-b`, each once. The prompt is `Question: What is the last digit of a + b?` followed by newline and `Answer:`. The correct answer is `(a+b) mod 10` as a single decimal character. The incorrect answer is `(correct+5) mod 10`. It is always different.

Both orders of an off-diagonal problem stay in the same split. For each answer digit, assign six rows to training, two to validation and two to test. Each split thus has uniform correct and incorrect answer counts and symmetric ordered answer comparisons. The resulting sizes are 60/20/20 rows and 33/11/11 distinct unordered problems. The ten diagonal problems are allocated 6/2/2.

Use seed 20261022. Ordering is ascending SHA-256 of compact UTF-8 JSON `[seed, namespace, value]`, with namespaces fixed in the generator. Allocate each even-answer digit's two diagonal problems together. Assign three even-answer digits' diagonal blocks to training, one to validation and one to test using the hashed order. Allocate the remaining size-two off-diagonal groups by answer digit and hashed order, filling training, validation then test quotas. Row serialization is ascending operand order, with sorted JSON keys and a trailing newline.

## Feasibility boundary

With the full 10×10 operand grid and uniform per-answer counts, each odd answer digit has five size-two unordered groups and no diagonal examples. Consequently each split's per-answer count must be even. A uniform 50/30/20-row split, requiring counts 5/3/2 for each digit, is impossible without dropping groups, breaking symmetry, or changing the data or constraints. The generator rejects it rather than leaking reversed operands.

The supported family is exactly three positive even per-answer counts summing to ten, such as 6/2/2 or 4/4/2. The construction supports every member of this family and balances diagonal fractions at 10%. The paired-diagonal allocation is a deliberate construction choice, not a required mathematical property of every feasible split.

## Controls and audit

Before any model use:
1. Verify exact IDs, operands, prompt strings and arithmetic labels against the full grid.
2. Verify mutually exclusive, exhaustive split membership and no unordered-problem overlap.
3. Verify uniform correct/incorrect answer frequencies, symmetric answer-pair counts, and diagonal fractions within every split.
4. Fit the count-difference answer-identity baseline only on training labels. Report wins, ties, losses and half-tie accuracy on all splits.
5. Report fixed ascending-digit and descending-digit answer scorers. Both must be at 50%. These expose the previous +3-negative cycle's shortcut: descending digit order scores 70% under uniform correct-answer counts despite equal label marginals.
6. Report the exact symbolic arithmetic oracle solely as a generator sanity control, not a model score.
7. Hash dataset and audited source artifacts; reproduce byte-for-byte in a new directory. Do not overwrite a freeze.

If an answer-only scalar scoring function assigns values s(x), symmetry gives an equal number of `(x,y)` and `(y,x)` comparisons. Their half-tie credits sum to one. Thus every such fixed scoring function has exactly 50% pair-ranking accuracy. This excludes fixed answer-identity rankings, not all prompt-conditioned or non-scalar shortcuts.

## Rules for a future model experiment

No model choice, extraction layer, tokenizer check, transformation, validation selection, test evaluation or causal intervention is executed in this milestone. A future model runner must be a separate reviewed change that records its frozen model/runtime and candidate configurations before evaluation, fits transformations only on training, selects among candidates only on validation, and reports the designated test once. If tokenization changes inclusion, do not silently filter: regenerate and audit a newly versioned benchmark or report the inability to preserve its constraints.

The test partition is public and synthetic. Do not promote it to a general truthfulness, factuality, reasoning, steering, or biological-correspondence claim. First and second operand identities, the template, and answers are shared across splits. The arithmetic is simple and narrowly constructed, and +5 negatives may change difficulty. Compare future results to historical +3/one-digit-operand results only with these protocol changes explicit. Treat reversed rows as clustered observations, not independent tasks. Any inference uncertainty must respect that grouping; no confidence interval or significance claim is supplied here.
