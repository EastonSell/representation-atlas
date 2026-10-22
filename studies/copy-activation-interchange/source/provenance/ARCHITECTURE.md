# Source architecture and acceptance contract (not implemented)

The package should be independent of historical output directories. Model/runtime paths are CLI arguments, not personal absolute paths in public source. Read-only historical data supply only the exclusion list and reference provenance.

## Proposed modules

- `dataset.py`: pure prompt construction, deterministic candidate eligibility/ordering, alignment and exclusion audit. Tokenizer-only `freeze-data`; no model class imports.
- `config.py`: immutable schema validation and fixed cells, constants, thresholds and budgets; reject unknown settings. Digest includes all scientific and numerical parameters.
- `model_adapter.py`: offline local model loading and architecture assertions. Separate raw tuple[0] decoder hooks from normalized hidden states. Read-only capture clones all seven block outputs to CPU. Patching clones output before replacing requested positions; preserves tuple tail; requires exactly one hook invocation; removes hooks in finally. No cached KV, padded batches or residual reuse across trials.
- `runner.py`: distinct `baseline` and `patch` commands. Patch requires reviewed baseline manifest digest plus the exact source/data/config/model digests. Immutable new output directory only. No fitting, optimization or layer-selection API.
- `metrics.py`: pure NumPy float64 scoring from retained float32 final logits. Both directions averaged before group inference. Distinguish pairwise correctness, greedy correctness, replacement, real switch and off-target.
- `audit.py`: independent formulas/reconstruction rather than importing scoring functions under test. Check full coverage, numerical control identities, baseline gate, patch vector provenance, every scalar and every summary, source/data/model digests. No model inference.
- `report.py`: deterministic full-map tables/plots from audited archive; all 14 word/final cells displayed with block29 visibly tagged architectural controls. Full-sequence and prefix controls separately displayed. No missing or failed cells erased. No invented result text if incomplete.

## Artifact lifecycle

Before a forward, append/fsync a planned-call record with monotonically increasing global count, phase, pair, direction, block, condition. Afterward save final logits as float32 NPY using temporary-file + fsync + atomic rename; then append/fsync completed row with hash. This preserves exact evidence ordering; a killed latest call may remain uncertain and counts against budget. Capture full states only during baseline read-only calls. Independent reader validates every referenced file before trusting a row. Output directories contain no weights, credentials or unrelated files.

Failure writes status/failure manifest when possible. Watchdog SIGTERM allows flush, followed after 15 seconds by SIGKILL. On timeout, crash, nonfinite values, duplicate call, identity failure, hash mismatch or resource overrun: stop, retain partial artifacts, classify uncompleted cells missing. No silent same-run retry, no post-hoc exclusion and no automatic resumption. Any approved continuation must retain original ledger and cumulative caps; absent such approval, a partial run stays partial.

## Minimum pre-inference tests

1. Tokenizer audits exact single changed source token, append-only answer in both prefixes, same sequence lengths, lexically disjoint 64 unique words, 32 groups and both orientations; q<p<f.
2. Pure hook tests select one row/position, clone instead of mutating donor/cache, preserve tuple tail, invoke once, remove after errors, never mutate another layer or position.
3. Synthetic causal toy network: identity and identical-prefix checks; full-sequence replay; last-block final replay; last-block nonfinal no effect. Deliberately incorrect position/layer must fail.
4. Last-layer norm regression: raw final block output differs from normalized hidden state; correct adapter must capture the former. Hidden_states[l+1] may compare to raw block output only for l<29.
5. Margin sign/orientation test, switch-vs-replacement test, off-target denominators, tie/nonfinite tests, unstable normalized denominator exclusion and no ratio clipping.
6. Group-bootstrap test uses pair IDs, retains both directions together, never treats 64 rows as independent; fixed-seed reproducibility; zero-SE cell handling.
7. Budget arithmetic and stage barriers; fail-before-forward when count cap reached, mocked timeout, no model loading in data/audit commands, source/config tamper rejection.
8. Write interruption and artifact corruption; repeated IDs/calls, missing/extra cells, invalid array dimensions, no silent overwrites, uncertain attempts count.
9. Numerical controls are actually asserted against complete 49,152-element vectors, not just target logits or argmax. Independent auditor catches a non-target logit alteration.

## Review boundaries

Protocol review precedes executable implementation. Synthetic tests and data/source freeze review precede baseline inference. Complete baseline gate review precedes patching. Independent artifact audit precedes any publication. This package itself includes neither an executable runner nor inference results.
