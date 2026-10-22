# Split-confound audit: metadata controls, not a new model run

This additive, post-hoc milestone audits the 100 published arithmetic pairs and all 18 projection result rows from `smollm-activations`. It groups those rows into six distinct seed/split combinations, rejects invalid partitions and inconsistent projection splits, and fits an answer-identity-only baseline using **training labels only**. Historical outputs and the failed pre-answer capability gate remain unchanged.

## Fresh measured results

| Split | Seed | Train answer TV | Test answer TV | Answer-only test accuracy | Shared unordered problems |
|---|---:|---:|---:|---:|---:|
| Random pairs | 11 | 0.1500 | 0.2250 | 42.50% | 24 |
| Random pairs | 29 | 0.1167 | 0.1750 | 28.75% | 26 |
| Random pairs | 47 | 0.1333 | 0.2000 | 43.75% | 27 |
| Balanced operand groups | 11 | 0 | 0 | 50.00% | 24 |
| Balanced operand groups | 29 | 0 | 0 | 50.00% | 24 |
| Balanced operand groups | 47 | 0 | 0 | 50.00% | 24 |

There are no shared exact pair IDs or prompt strings. Random splits share first-operand values; balanced splits do not. All splits share second-operand values. Crucially, the balanced splits still put reversed operands (for example a+b versus b+a) across the boundary: 24 distinct unordered problems per split. This is a **shared-content diagnostic**, not proof of training contamination or a claim the original experiment violated its stated first-operand protocol.

Answer total variation (TV) is half the sum of absolute differences between correct-answer and incorrect-answer frequencies within a split. Global TV is zero, yet random-split TV is nonzero. The answer-only score for an answer string is its training correct count minus its training incorrect count. A held-out pair receives one point if the correct answer has a larger score, half a point for a tie, and zero otherwise. Unseen strings score zero. Balanced splits have all-zero scores, so their 50% is entirely ties, not successful task solving.

The mean random-split answer-only accuracy is 38.33%, demonstrating that a below-chance ranking can arise from answer composition without any model. Because global counts are balanced and the partition is exhaustive, the test count difference is the negative of the training difference. These finite-population constraints can produce an anti-generalizing shortcut. This does **not** quantify the contribution of this mechanism to the archived 40.83% activation ranking. Per-pair model margins and activation arrays are not public artifacts, so a residualized model analysis is unavailable.

## Reproduce offline

From the repository root, using Python 3.10+ (the new audit uses only the standard library):

```sh
python -m unittest discover -s tests -v
python split_audit.py --output /tmp/representation-audit-fresh.json
cmp /tmp/representation-audit-fresh.json milestones/split-confound-audit/results.json
(cd milestones/smollm-activations && python -m unittest discover -v)
```

Use a new output filename on each run. The CLI refuses to overwrite any existing output. Paths are resolved relative to the source file, not the working directory. Output is deterministic and embeds SHA-256 hashes of both archived inputs. No model weights, tokenizer, network access, or new inference are used. Root geometry tests additionally use the repository's existing NumPy/scikit-learn dependencies.

Validation on Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.8.0: root and pretrained-milestone test counts are in `validation.json`. The separate preanswer suite had seven passing tests and one setup error in `EvidenceTests`: its tokenizer verifier requires `SMOLLM_MODEL_PATH`. The 11 methods in that class did not execute. No model download, model replay, or capability-gate repair was attempted.

## Next controlled experiment, not performed here

Freeze a new dataset and split before inspecting model outcomes. Hold out unordered operand groups if the desired claim is generalization to new arithmetic problems; also verify within-split answer balance rather than assuming grouping supplies it. The existing first-operand protocol cannot simultaneously claim unordered-problem disjointness. Archive per-pair scores and split IDs to enable paired controls and uncertainty estimates. Use fresh test data for confirmatory comparison, not further selection on these six overlapping exploratory splits.

This audit provides no significance test, causal intervention result, biological correspondence, new capability evidence, or general model-quality conclusion. Original code and generated analysis are MIT under the repository LICENSE. Inputs are existing public synthetic arithmetic artifacts; no model or paper text is redistributed.
