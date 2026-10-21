# Small pretrained language-model activation protocol

## Question
Does a held-out linear correctness direction extracted from a small pretrained language model survive train-only centering/projection and matched-answer controls on an exactly verifiable task?

## Data and controls
Generate the 100 two-digit operands a,b in 0..9 with requested target (a+b) mod 10. For each prompt, append the correct one-digit answer and a deterministically chosen incorrect one-digit answer. Balance answer identity distributions using a fixed modular offset. Verify tokenizer answer lengths are matched; report and exclude mismatched token-length pairs. Split by operand pair before fitting transformations or direction. Include label-shuffled training directions, orthogonal rotation invariance, and removal of 0/1/2 leading PCs. Store only generated public arithmetic prompts and outputs.

## Model and artifacts
Reuse a pinned, license-checked 135M-class pretrained checkpoint from the register project; no second download. Keep tokenizer/model manifest, revision, weight hashes, framework versions and actual CPU resource settings. Extract the hidden state at the final answer token in fixed evaluation mode. This reads representations conditioned on supplied answers; it does not claim the model independently generated correct answers or understands a human's comprehension level.

## Evaluation
Report held-out matched-pair ranking and conditional answer log-likelihood correctness preference, separately. Centering cannot change pair differences; mean direction is invariant to pairing permutations. PCA can remove task information. Small task sample size and overlapping seed splits preclude treating seeds as independent populations. A useful outcome includes chance-level correctness separation or mismatch between hidden-state and likelihood preferences.

## Next causal step
Any claim of steering requires a separately preregistered activation intervention with task accuracy and side-effect controls on held-out prompts. This observational experiment does not establish causal mechanisms, biological alignment, general truthfulness or safety.

## Exploratory split refinement
After the initial random-pair run produced below-chance separation, a possible answer-token composition artifact was identified. The global correct/incorrect answer histograms are identical, but random complementary subsets need not preserve that balance. A second run adds an explicitly exploratory split by first operand: six operand groups train, four test. Each group contains every second operand, so both partitions preserve equal correct/incorrect answer histograms. The original output is retained as `results/initial-random-split.json`. This refinement is a diagnosis informed by initial results, not an untouched confirmatory test. The two forward passes produced identical activation SHA-256 hashes.
