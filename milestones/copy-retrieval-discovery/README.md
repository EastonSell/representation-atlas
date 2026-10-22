# Training-only task-change discovery: copying, not arithmetic recovery

## Observed result

The pinned SmolLM2-135M-Instruct model can perform the narrowly specified **plain one-token word-copying** discovery task. Copying achieved 24/24 strict correct-versus-foil likelihood comparisons, 12/12 unique prompts with unrestricted greedy exact target tokens (24/24 repeated rows) and 12/12 entirely-correct counterfactual groups. Its log-probability margin ranged from 3.9164 to 11.4322 (mean 8.2390). There were no ties or tokenization invalids.

**Those 24 rows contain only 12 unique prompts**, each compared against two neighboring foils. The 12 circular groups share words and are not independent replications. This is a small convenience-vocabulary discovery result, not held-out capability evidence or a statistical estimate of general copying ability.

All three predeclared candidates were run and retained:

| Fixed priority | Strict wins / 24 | Exact next tokens / 24 | Entire groups / 12 | Operational gate |
|---|---:|---:|---:|---|
| Plain copying | 24 | 24 | 12 | Pass, selected |
| Labeled value retrieval | 16 | 1 | 4 | Fail |
| Repeated-key continuation | 15 | 14 | 6 | Fail |

The retrieval failure is specific to this prompt and strict next-token format: many outputs were ` The` or an end-of-message token. It does not show the model lacks all semantic retrieval ability. No post-hoc stripping, extra generation, prompt repair or relaxed scoring was used. Copying success does not repair or relabel either earlier arithmetic failure.

## Fixed design, selection and input provenance

`inputs.json` was generated and hashed before model inference. It records three fixed templates, 12 investigator-chosen ordinary English words, every prompt, correct/foil string, group and discovery-training ID. All words are synthetic task inputs, not personal data or a fetched corpus. Adjacent words on the declared circular list form 12 pairs; both target/foil orientations are included. Two-field templates alternate field order by group. Each candidate answer occurs equally often as correct and incorrect. No original arithmetic inputs were used for candidate construction or selection.

The first passing template in the declared priority wins. Passing requires all three: at least 22/24 strict likelihood wins, 10/12 entirely-correct groups and 20/24 exact greedy next tokens. Thresholds are operational engineering choices, not established statistical capability criteria. All candidates run even after the first success. Ties and nonmatching output fail; invalid tokenization rejects the entire discovery rather than filtering examples.

The scoring instrument validates every candidate suffix is exactly one token and preserves the full prefix. One model forward on the prompt yields both candidate log probabilities and the unrestricted vocabulary argmax. No correct/foil answer is fed to the model. Exact greedy means exactly the first predicted token ID, not a complete response. Candidate margins therefore equal next-token logit differences. The model is loaded locally with safetensors, no remote code, float32/eager CPU evaluation and two threads; all eight model-file hashes are verified before inference.

`probe.py` is an archived one-off discovery instrument, not a production or held-out runner. `SOURCE_FREEZE.sha256` records its pre-inference source, tests and input bytes. `run/` retains raw predictions, prompt/suffix token IDs, runtime/model metadata, every candidate summary and artifact hashes. The source/model runtime is reused; nothing was installed or downloaded.

## Offline audit and residual limitations

`audit.json` independently checks saved ID alignment, token validity, margin arithmetic, greedy-token flags and semantic identity of source/saved inputs. The two input files have different hashes because the archive writer sorts JSON keys; their parsed content is identical. The frozen imported helper's hash is recorded separately. All six local tests and 62 root aggregate core tests pass; the aggregate's optional tokenizer suite was explicitly not rerun for this milestone.

A post-hoc continuation diagnostic gets 11/12 likelihood and greedy successes when A is the second/most-recent field, versus 4/12 likelihood and 3/12 greedy successes when A is first. Word-pair identity and field order are coupled, so this suggests positional sensitivity but does not establish its cause or change selection. All templates share the same narrow vocabulary. Immediate repetition without distractors cannot distinguish instruction following from a repetition/recency shortcut.

This discovery instrument is intentionally fixed to these inputs, not a hardened general evaluator. Its six unit tests cover balance and selection; tokenizer/logit-path behavior is exercised by the actual run, not independent mocked regression tests. Only candidate log probabilities and the returned greedy token are archived, not complete vocabulary logits. Therefore saved evidence verifies arithmetic and token flags but cannot independently reconstruct the global argmax without inference. The proposed future causal protocol requires full untouched logits and stronger adapter tests.

## Boundaries and next proposal

No model weights were fitted, no hidden representations were extracted, no causal intervention was run, and no new validation/test data was generated or evaluated. The original frozen arithmetic benchmark and failed-gate evidence were not changed.

[TASK_CHANGE_PROPOSAL.md](TASK_CHANGE_PROPOSAL.md) is a **proposal only** for new lexical-disjoint data and train-only PCA of pre-answer states. It targets preservation and causal sensitivity of copying, with unchanged logits, no-op checks and matched random controls. It requires review and a new input/protocol freeze before future inference. It does not claim semantic truth, an induction circuit, general knowledge or brain alignment.

## Reproduction

Use the already pinned runtime described in the root setup documentation and an existing model directory:

```sh
python -m unittest discover -s milestones/copy-retrieval-discovery -v
sha256sum -c milestones/copy-retrieval-discovery/SOURCE_FREEZE.sha256
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /path/to/model-runtime/bin/python milestones/copy-retrieval-discovery/probe.py --model-path /path/to/verified/model --out /tmp/copy-discovery-new
```

Output directories must not exist. Run the root aggregate checks separately; they do not yet discover this milestone's six local tests. No held-out or causal option exists in this discovery instrument.
