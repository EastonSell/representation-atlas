"""Construct and audit a frozen, model-free unordered-arithmetic benchmark."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

from split_audit import composition, fit_answer_scores, score_answers

SPLITS = ('train', 'validation', 'test')
DEFAULT_SEED = 20261022


def validate_counts(per_answer):
    if (len(per_answer) != 3 or any(type(n) is not int or n <= 0 or n % 2 for n in per_answer)
            or sum(per_answer) != 10):
        raise ValueError('Per-answer counts must be three positive even integers that sum to 10; '
                         'odd-sum unordered groups have size two, so odd counts are impossible')


def ordered(values, seed, namespace):
    """Hash order avoids dependence on language-runtime RNG versions."""
    return sorted(values, key=lambda value: hashlib.sha256(
        json.dumps([seed, namespace, value], separators=(',', ':')).encode('utf-8')).hexdigest())


def make_row(a, b, split):
    correct = (a + b) % 10
    return {'id': f'{a}-{b}', 'a': a, 'b': b, 'split': split,
            'prompt': f'Question: What is the last digit of {a} + {b}?\nAnswer:',
            'correct': str(correct), 'incorrect': str((correct + 5) % 10)}


def generate(seed=DEFAULT_SEED, per_answer=(6, 2, 2)):
    """Allocate size-two blocks, including paired diagonal groups, by answer."""
    validate_counts(per_answer)
    if type(seed) is not int:
        raise ValueError('Seed must be an integer')
    diagonal_answers = ordered(list(range(0, 10, 2)), seed, 'diagonal-host')
    host = {}
    cursor = 0
    for split, count in zip(SPLITS, per_answer):
        for digit in diagonal_answers[cursor:cursor + count // 2]:
            host[digit] = split
        cursor += count // 2
    assignments = {}
    for digit in range(10):
        groups = [(a, b) for a in range(10, 20) for b in range(a, 20) if (a + b) % 10 == digit]
        off_diagonal = ordered([g for g in groups if g[0] != g[1]], seed, f'answer-{digit}')
        cursor = 0
        for split, count in zip(SPLITS, per_answer):
            diagonal_here = host.get(digit) == split
            needed = count // 2 - int(diagonal_here)
            for group in off_diagonal[cursor:cursor + needed]:
                assignments[group] = split
            cursor += needed
            if diagonal_here:
                for group in groups:
                    if group[0] == group[1]:
                        assignments[group] = split
    rows = [make_row(a, b, assignments[tuple(sorted((a, b)))])
            for a in range(10, 20) for b in range(10, 20)]
    return {'schema_version': 1, 'seed': seed, 'per_answer_counts': dict(zip(SPLITS, per_answer)),
            'operand_range_inclusive': [10, 19], 'incorrect_offset_mod_10': 5,
            'status': 'Frozen construction and controls before any inference on this benchmark',
            'rows': rows}


def audit(data):
    """Validate independently of assignment generation; reject violations."""
    if data.get('schema_version') != 1 or data.get('operand_range_inclusive') != [10, 19] or data.get('incorrect_offset_mod_10') != 5:
        raise ValueError('Unsupported benchmark metadata')
    try:
        counts = [data['per_answer_counts'][s] for s in SPLITS]
        validate_counts(counts)
        rows = data['rows']
        expected = {f'{a}-{b}' for a in range(10, 20) for b in range(10, 20)}
        ids = [r['id'] for r in rows]
        if len(ids) != 100 or set(ids) != expected:
            raise ValueError('IDs must contain each of the 100 exact operand pairs once')
        for row in rows:
            if (type(row['a']) is not int or type(row['b']) is not int
                    or row['a'] not in range(10, 20) or row['b'] not in range(10, 20)
                    or row['split'] not in SPLITS or row != make_row(row['a'], row['b'], row['split'])):
                raise ValueError('Invalid row ID, operands, labels, prompt, or split')
    except (KeyError, TypeError) as exc:
        raise ValueError('Malformed benchmark') from exc
    subsets = {s: [r for r in rows if r['split'] == s] for s in SPLITS}
    keys = {s: {tuple(sorted((r['a'], r['b']))) for r in subset} for s, subset in subsets.items()}
    overlaps = sorted(set().union(*(keys[a] & keys[b] for a, b in itertools.combinations(SPLITS, 2))))
    if overlaps:
        raise ValueError('Unordered arithmetic problems cross split boundaries')
    train_scores = fit_answer_scores(subsets['train'])
    reports = {}
    for split, count in zip(SPLITS, counts):
        subset = subsets[split]
        positive = Counter(r['correct'] for r in subset)
        negative = Counter(r['incorrect'] for r in subset)
        expected_counts = {str(i): count for i in range(10)}
        if dict(positive) != expected_counts or dict(negative) != expected_counts:
            raise ValueError('Within-split answer balance or size violated')
        answer_pairs = Counter((r['correct'], r['incorrect']) for r in subset)
        asymmetry = sum(abs(n - answer_pairs[(b, a)]) for (a, b), n in answer_pairs.items())
        if asymmetry:
            raise ValueError('Answer-pair comparison is not symmetric')
        diagonals = sum(r['a'] == r['b'] for r in subset)
        if diagonals != count:
            raise ValueError('Diagonal-problem composition differs from target')
        reports[split] = {**composition(subset), 'unordered_problems': len(keys[split]),
                         'diagonal_pairs': diagonals, 'asymmetric_answer_pair_count': asymmetry,
                         'answer_pair_counts': {f'{a}:{b}': n for (a, b), n in sorted(answer_pairs.items())},
                         'prompt_character_lengths': sorted({len(r['prompt']) for r in subset}),
                         'count_baseline': score_answers(subset, train_scores),
                         'ascending_answer_baseline': score_answers(subset, {str(i): i for i in range(10)}),
                         'descending_answer_baseline': score_answers(subset, {str(i): -i for i in range(10)}),
                         'symbolic_oracle_accuracy': sum(str((r['a'] + r['b']) % 10) == r['correct'] for r in subset) / len(subset)}
    return {'schema_version': 1, 'measurement': 'Pre-inference construction integrity and deterministic metadata controls',
            'cross_split_unordered_overlap': overlaps, 'train_answer_scores': train_scores,
            'first_operand_overlap_counts': {f'{a}:{b}': len({r['a'] for r in subsets[a]} & {r['a'] for r in subsets[b]})
                                             for a, b in itertools.combinations(SPLITS, 2)},
            'splits': reports, 'model_inference_performed': False,
            'limitations': ['New prompts and partitions, but motivated by prior exploratory model results.',
                            'This benchmark is not an untouched external confirmatory test set.',
                            'Operand values, prompt template, and answer identities remain shared across splits.',
                            'Reverse-order prompts within each split are not independent trials.',
                            'Symmetric negatives block fixed scalar answer-only ranking shortcuts, not all prompt-conditioned shortcuts.',
                            'Character lengths are checked; tokenizer identity and lengths are not checked.',
                            'Symbolic oracle validates generated arithmetic, not model capability.']}


def serialized(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='Must not exist; freezes two new JSON files')
    args = parser.parse_args()
    dataset = generate()
    report = audit(dataset)
    payload = serialized(dataset)
    report['dataset_sha256'] = hashlib.sha256(payload).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'dataset.json').write_bytes(payload)
    (args.output_dir / 'audit.json').write_bytes(serialized(report))


if __name__ == '__main__':
    main()
