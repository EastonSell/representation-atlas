# Unordered-problem-disjoint arithmetic benchmark

A frozen, model-free follow-on to the [split-confound audit](../split-confound-audit/README.md). This constructs new prompts using operands 10–19, with **no unordered arithmetic problem shared between training, validation and test**. No model was loaded, downloaded or evaluated.

## Measured construction and controls

| Split | Ordered rows | Unordered problems | Diagonal rows | Count-only rank | Ascending / descending rank | Symbolic oracle |
|---|---:|---:|---:|---:|---:|---:|
| Train | 60 | 33 | 6 | 50% | 50% / 50% | 100% |
| Validation | 20 | 11 | 2 | 50% | 50% / 50% | 100% |
| Test | 20 | 11 | 2 | 50% | 50% / 50% | 100% |

Each split has uniform correct and incorrect answer identities. Count-only scores are fitted on training and all tie at zero. Fixed-answer ranking is strengthened beyond marginal balance: equal numbers of `(answer x, answer y)` and `(answer y, answer x)` comparisons force every fixed scalar answer-only ranker to 50%, counting ties as half credit.

Why change the negatives? The older +3-modulo-10 wrong-answer rule admits a 70% descending-digit ranker even with uniform label marginals. This benchmark uses the symmetric +5 rule. That is a changed task condition, not a repair of the old model results. It may change difficulty.

The split is feasible because odd-answer problems can be allocated in size-two unordered groups. This also makes a fully balanced 50/30/20 allocation impossible on this exact grid: odd per-answer counts cannot be assembled from size-two groups. The generator rejects incompatible counts with an explanation. See the [frozen protocol](PROTOCOL.md) for the constructive proof and future-use restrictions.

## Reproduce without network or model dependencies

From the repository root:

```sh
python arithmetic_benchmark.py --output-dir /tmp/arithmetic-benchmark-fresh
cmp /tmp/arithmetic-benchmark-fresh/dataset.json milestones/unordered-arithmetic-benchmark/dataset.json
cmp /tmp/arithmetic-benchmark-fresh/audit.json milestones/unordered-arithmetic-benchmark/audit.json
python -m unittest discover -s tests -v
sha256sum -c milestones/unordered-arithmetic-benchmark/FREEZE.sha256
```

Choose a new output directory on every invocation. Existing directories are refused. The generator and audit use only Python's standard library plus the repository's local `split_audit.py`; full root tests also need the existing NumPy/scikit-learn dependencies. The command does not install anything.

`dataset.json` contains all exact prompts, labels, IDs and split assignments. `audit.json` contains measured controls and the dataset SHA-256. `FREEZE.sha256` records source, protocol and output bytes before model inference on this benchmark. No independent timestamped preregistration is claimed. Prior exploratory findings informed these construction choices.

## Verification and limitations

The 24-test root suite passes, including 60 combinations of feasible count allocations and seeds, malformed/corrupted datasets, exact arithmetic, disjointness, shortcut controls, deterministic CLI replay and overwrite refusal. All five existing pretrained milestone tests pass. The separate preanswer independent-verifier setup still requires an unavailable tokenizer/model path: seven tests pass, one setup errors, and its 11 methods do not execute. This milestone does not change or rerun that failed capability gate.

These 100 new prompts are not broad or independent evidence of model capability. Reversed rows within a split remain dependent, operand identities/templates remain shared, tokenization is untested, and public synthetic test partitions can be overfit. No new inference, learned probe, selected projection, causal steering, confidence interval or biological claim is included. A future model experiment needs a separate frozen runner and train-only fitting/validation-only selection. Historical experiment artifacts remain unchanged.

Original generator, tests, synthetic data and audit: MIT under the repository LICENSE. No weights, downloaded datasets or paper text are bundled.
