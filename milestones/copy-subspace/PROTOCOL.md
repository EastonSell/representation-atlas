# Frozen implementation scope: copying-subspace study v1

The scientific design is the approved [task-change proposal](../copy-retrieval-discovery/TASK_CHANGE_PROPOSAL.md). This implementation retains its template, lexical independence, exact thresholds, block-14 pre-answer site, k values, train-only fitting, preservation selection and random controls. No original arithmetic gate is changed.

The 80-word raw pool was fixed before tokenizer preflight and passed on its first attempt. DATA_FREEZE.json records the actual local timestamp and SHA-256 of inputs/audit. This is a local freeze, not a claim that GitHub publication preceded it. SOURCE_FREEZE.json is created after implementation tests and before real hook verification or the train gate. Publication may follow; no claim of public preregistration is made.

## Stages and stop boundaries

- `verify-hook`: exactly one training row, two forwards. Compare untouched no-hook logits/state with a read-only decoder-block hook. No PCA, fitting, validation or test inference.
- `train`: 40 training rows, 80 forwards including per-row no-op verification. Cache baseline states together with logits for hook correctness and reproducibility, but fit no mean/PCA/random basis until all frozen capability thresholds pass. This caching-before-gate implementation detail does not constitute a representation-analysis result. A failed gate stops without any validation/test inference or fit.
- `select`: requires the exact successful training artifact-manifest digest. Uses 20 validation rows and at most 1,300 forwards: two baseline/no-op calls plus 3 k values × (1 PCA + 10 rank-random + 10 energy-random) = 65 calls per row. Freeze the largest preserving k and all validation evidence before any test command. An invalid baseline stops selection.
- `test`: requires the exact frozen selection artifact-manifest digest. Uses 20 test rows and at most 460 forwards for nonzero k: two baseline/no-op plus 21 interventions per row. k=0 makes only 40 baseline/no-op forwards and is explicitly not a nonzero preservation result. No test-driven choice is exposed.

Every run uses a new directory. Default/max cooperative wall budgets are 120/300/900/600 seconds respectively; a smaller positive budget is allowed. Checks happen before each model forward, so a currently executing forward can finish beyond the deadline. Model verification/loading counts against wall time. There is no claim of hard preemptive cancellation. Two CPU threads, CPU float32, eager attention, eval mode, no cache and offline local safetensors only are fixed.

## Evidence and numerical details

Each baseline immediately writes compressed binary float32 logits and the block residual to an individual NPZ, then appends/fsyncs its row. Each no-op and intervention row is appended/fsynced immediately. If interrupted, completed rows/arrays remain; ordinary caught failures also produce failure.json and a manifest. Process termination during a file write may leave a truncated latest file; there is no in-place resume or claim of atomic multi-file completion. Array hashes and file sizes are included in manifests. Test data is not needed to replay a train/selection artifact's integrity check.

The untouched pass uses no hook; hidden_states[15] is the pre-block-15 state, equivalent to output of decoder block 14 in the installed 30-block LLaMA implementation. The hooked pass must reproduce both that state and complete logits to absolute 1e-6. The no-op's returned state must additionally be exactly equal. Hook removal occurs in a finally block. Actual numerical hook equivalence must be reviewed before the training capability gate.

PCA and random matrices use NumPy float64; the intervention delta is cast to the residual's float32 dtype before subtraction. NumPy 2.3.5, torch 2.5.1+cpu and transformers 4.46.3 are enforced. Fixed random bases use NumPy default_rng/QR with the ten declared seeds separately for each k. Realized perturbation norms may differ from float64 requested norms at float32 rounding precision; this is not a claim of bitwise equal energy.

All outputs are pre-answer. Candidate IDs are used only in scoring, never appended to model input. Primary margins use full-vocabulary logits; full untouched logits permit independent argmax/probability checks. Candidate/control scalar outcomes and group-level differences are retained, including all seeds and their descriptive averages. Fitting/scores use every row, not only successful ones. Preserving accuracy on this small fixed-format sample is not proof of statistical equivalence, exclusive causal necessity, an induction circuit, semantic truth, general knowledge or brain alignment.

The driver has separate stage commands, not an automatic train-to-test pipeline. Review train results before selection and review frozen selection before test. This source/data checkpoint authorizes neither an unreviewed test run nor changes to hypotheses after outcomes.
