# Completed research milestones

The project began with synthetic geometry and digit activations and now includes pretrained diagnostics, frozen benchmarks and explicit failed capability gates. The root README is the current index; historical artifacts and their source freezes retain their original scope.

1. [Synthetic geometry and learned digit activations](README.md): centering controls and destructive top-PC-removal cases. On digit hidden features, nearest-centroid accuracy fell from 95.49% to 91.05% after removing one principal component.
2. [Pretrained SmolLM2 activation geometry](milestones/smollm-activations/README.md): 100 matched arithmetic-answer pairs, three overlapping seed splits, and shuffled-label controls. Center-only ranking was 40.83% for random pairs versus 53.33% for an exploratory answer-balanced group split. [Full results](milestones/smollm-activations/results/experiment.json) and [protocol](milestones/smollm-activations/PROTOCOL.md).

The group split was introduced after inspecting the initial result; it is diagnostic, not independent confirmation. Candidate answers are supplied to the model. Hidden-state ranking is distinct from likelihood preference, generated-answer correctness, causal steering, or brain alignment. The milestone supplies a pinned model downloader and reproduction commands; pretrained weights are not bundled. Original work is MIT; source digits and model provenance retain their separately documented licenses.

3. [Pre-answer capability gate](milestones/preanswer-capability/README.md): the frozen development gate failed (14/24 constrained choices; 12/24 correct target propositions in a separate post-hoc audit). No causal steering was run. Includes portable repaired runner, archived outputs and exact numerical reproduction.

4. [Post-hoc split-confound audit](milestones/split-confound-audit/README.md): metadata-only answer-identity controls and reversed-operand overlap diagnostics; no new model inference.
5. [Frozen unordered-arithmetic benchmark](milestones/unordered-arithmetic-benchmark/README.md): 100 two-digit-operand prompts, balanced 60/20/20 splits, no cross-split unordered problems, symmetric wrong-answer comparisons and explicit feasibility constraints. Dataset/control construction, not model capability evidence.
6. [Frozen evaluation runner source](milestones/frozen-evaluation-runner/README.md): training-only capability gate, conditional train-only fitting, validation-only projection selection and per-pair evidence archiving. Published before the following outcome.
7. [Repeated failed arithmetic gate](milestones/frozen-evaluation-results/README.md): actual pretrained likelihood evaluation scored 30/60 strict-correct training rows and 16/33 entirely-correct groups, below frozen 45/60 and 24/33 thresholds. Repeat evidence is byte-identical. Both invocations stopped before held-out inference, activation extraction or projections.
8. [Release-safety and reproducibility checks](milestones/release-safety/README.md): exact observed runtime locks, safe fresh-output CLI paths, historical source preservation, and aggregate checks with explicit optional tokenizer-verification coverage. No new model evaluation or scientific result.

For current setup and aggregate verification, use [SETUP.md](SETUP.md). Old manifests and source freezes retain their historical scope; they are not silently rewritten to describe later versions.
