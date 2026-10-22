"""Offline metadata controls; never loads a model or changes historical results."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

PAIRS = 'milestones/smollm-activations/results/pairs.json'
EXPERIMENT = 'milestones/smollm-activations/results/experiment.json'


def fit_answer_scores(train):
    """Fit integer count differences using training labels only."""
    positive = Counter(row['correct'] for row in train)
    negative = Counter(row['incorrect'] for row in train)
    return {key: positive[key] - negative[key] for key in sorted(positive.keys() | negative.keys())}


def score_answers(rows, scores):
    if not rows:
        raise ValueError('Cannot score empty rows')
    margins = [scores.get(row['correct'], 0) - scores.get(row['incorrect'], 0) for row in rows]
    wins = sum(value > 0 for value in margins)
    ties = sum(value == 0 for value in margins)
    return {'wins': wins, 'ties': ties, 'losses': len(rows) - wins - ties,
            'accuracy_half_ties': (wins + .5 * ties) / len(rows)}


def composition(rows):
    if not rows:
        raise ValueError('Cannot summarize empty rows')
    positive = Counter(row['correct'] for row in rows)
    negative = Counter(row['incorrect'] for row in rows)
    support = sorted(positive.keys() | negative.keys())
    return {'pairs': len(rows), 'correct_counts': dict(sorted(positive.items())),
            'incorrect_counts': dict(sorted(negative.items())),
            'label_answer_tv': sum(abs(positive[k] - negative[k]) for k in support) / (2 * len(rows))}


def audit_split(pairs, train_ids, test_ids):
    lookup = {row['id']: row for row in pairs}
    train_set, test_set = set(train_ids), set(test_ids)
    if (len(lookup) != len(pairs) or len(train_set) != len(train_ids)
            or len(test_set) != len(test_ids) or not train_set or not test_set
            or train_set & test_set or train_set | test_set != set(lookup)):
        raise ValueError('Expected unique pair IDs and a disjoint, nonempty, exhaustive partition')
    # This auditor is scoped to the published arithmetic task's operand IDs.
    operands = {}
    for key in lookup:
        parts = key.split('-')
        if len(parts) != 2 or any(p not in [str(i) for i in range(10)] for p in parts):
            raise ValueError('Expected arithmetic IDs a-b with operands 0 through 9')
        operands[key] = tuple(map(int, parts))
    train, test = [lookup[k] for k in train_ids], [lookup[k] for k in test_ids]
    def groups(ids, kind):
        if kind == 'commutative':
            return {'-'.join(map(str, sorted(operands[k]))) for k in ids}
        return {operands[k][kind] for k in ids}
    scores = fit_answer_scores(train)
    return {'train': composition(train), 'test': composition(test),
            'train_answer_scores': scores,
            'answer_only_train': score_answers(train, scores),
            'answer_only_test': score_answers(test, scores),
            'overlap': {'pair_ids': sorted(train_set & test_set),
                        'prompt_count': len({r['prompt'] for r in train} & {r['prompt'] for r in test}),
                        'first_operands': sorted(groups(train_ids, 0) & groups(test_ids, 0)),
                        'second_operands': sorted(groups(train_ids, 1) & groups(test_ids, 1)),
                        'commutative_problem_keys': sorted(groups(train_ids, 'commutative') & groups(test_ids, 'commutative'))}}


def audit_experiment(pairs, experiment):
    grouped = {}
    for row in experiment['rows']:
        key = row['split_mode'], row['seed']
        partition = (set(row['train_pair_ids']), set(row['test_pair_ids']))
        result = audit_split(pairs, row['train_pair_ids'], row['test_pair_ids'])
        if key in grouped:
            old = grouped[key]
            if old['partition'] != partition:
                raise ValueError('Projection configurations changed split membership')
            if row['remove_top_pcs'] in old['components']:
                raise ValueError('Duplicate projection configuration')
            old['components'].append(row['remove_top_pcs'])
        else:
            grouped[key] = {'partition': partition, 'components': [row['remove_top_pcs']], 'result': result}
    if not grouped:
        raise ValueError('No archived splits')
    return [{'split_mode': mode, 'seed': seed,
             'projection_rows_audited': len(value['components']),
             'remove_top_pcs': sorted(value['components']), **value['result']}
            for (mode, seed), value in sorted(grouped.items())]


def build_report(root):
    root = Path(root)
    inputs = {name: (root / name).read_bytes() for name in (PAIRS, EXPERIMENT)}
    return {'schema_version': 1,
            'measurement': 'Post-hoc metadata-only split/confound audit; no new inference',
            'source_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()},
            'global_composition': composition(json.loads(inputs[PAIRS])),
            'splits': audit_experiment(json.loads(inputs[PAIRS]), json.loads(inputs[EXPERIMENT])),
            'limitations': [
                'Answer identity counts are a diagnostic baseline, not an attribution of model scores.',
                'Operand or commutative overlap is a shared-content flag, not proof of contamination.',
                'Globally balanced answer identities need not be balanced within either split.',
                'No activations or per-pair model margins are archived; adjusted model accuracy cannot be recovered.',
                'Overlapping seeds and projection rows are not independent replications.',
                'No new model inference, causal steering, statistical significance, or capability-gate repair is claimed.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output file; existing files are never overwritten')
    args = parser.parse_args()
    report = build_report(Path(__file__).resolve().parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
