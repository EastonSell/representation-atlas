"""Pre-inference benchmark construction and integrity regressions."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import arithmetic_benchmark as benchmark

ROOT = Path(__file__).resolve().parents[1]


class ArithmeticBenchmarkTests(unittest.TestCase):
    def test_complete_exact_ids_and_arithmetic(self):
        data = benchmark.generate()
        rows = data['rows']
        self.assertEqual(len(rows), 100)
        self.assertEqual({r['id'] for r in rows}, {f'{a}-{b}' for a in range(10, 20) for b in range(10, 20)})
        for r in rows:
            self.assertEqual(r['correct'], str((r['a'] + r['b']) % 10))
            self.assertEqual(r['incorrect'], str((int(r['correct']) + 5) % 10))
            self.assertEqual(r['prompt'], f"Question: What is the last digit of {r['a']} + {r['b']}?\nAnswer:")

    def test_disjoint_balanced_splits_and_controls(self):
        report = benchmark.audit(benchmark.generate())
        self.assertEqual(report['cross_split_unordered_overlap'], [])
        for name, n in [('train', 60), ('validation', 20), ('test', 20)]:
            s = report['splits'][name]
            self.assertEqual(s['pairs'], n)
            self.assertEqual(s['diagonal_pairs'], n // 10)
            self.assertEqual(s['label_answer_tv'], 0)
            self.assertEqual(s['asymmetric_answer_pair_count'], 0)
            self.assertEqual(s['count_baseline']['accuracy_half_ties'], .5)
            self.assertEqual(s['ascending_answer_baseline']['accuracy_half_ties'], .5)
            self.assertEqual(s['descending_answer_baseline']['accuracy_half_ties'], .5)
            self.assertEqual(s['symbolic_oracle_accuracy'], 1)

    def test_reject_incompatible_counts(self):
        for counts in [(5, 3, 2), (6, 0, 4), (6, 2, 4), (6., 2, 2), (6, 4)]:
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, 'positive even integers.*sum to 10'):
                benchmark.generate(per_answer=counts)

    def test_all_feasible_counts_and_seeds(self):
        for a in (2, 4, 6):
            for b in (2, 4, 6):
                c = 10 - a - b
                if c <= 0:
                    continue
                for seed in range(10):
                    data = benchmark.generate(seed=seed, per_answer=(a, b, c))
                    report = benchmark.audit(data)
                    self.assertEqual([report['splits'][s]['pairs'] for s in benchmark.SPLITS], [10*a, 10*b, 10*c])

    def test_seed_determinism_and_effect(self):
        self.assertEqual(benchmark.generate(), benchmark.generate())
        self.assertNotEqual(benchmark.generate(seed=1), benchmark.generate(seed=2))
        self.assertNotEqual([r['split'] for r in benchmark.generate(seed=1)['rows']], [r['split'] for r in benchmark.generate(seed=2)['rows']])

    def test_reject_corruptions(self):
        data = benchmark.generate()
        for field, value in [('id', '99-99'), ('correct', '9'), ('incorrect', '9'), ('prompt', 'bad'), ('split', 'other'), ('a', 99)]:
            altered = copy.deepcopy(data)
            altered['rows'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                benchmark.audit(altered)
        altered = copy.deepcopy(data)
        altered['rows'].append(altered['rows'][0])
        with self.assertRaises(ValueError):
            benchmark.audit(altered)
        altered = copy.deepcopy(data)
        altered['rows'].pop()
        with self.assertRaises(ValueError):
            benchmark.audit(altered)
        altered = copy.deepcopy(data)
        row = next(r for r in altered['rows'] if r['a'] != r['b'])
        row['split'] = next(s for s in benchmark.SPLITS if s != row['split'])
        with self.assertRaises(ValueError):
            benchmark.audit(altered)

    def test_equal_marginals_do_not_imply_pair_ranking_balance(self):
        legacy = [{'correct': str(i), 'incorrect': str((i + 3) % 10)} for i in range(10)]
        descending = {str(i): -i for i in range(10)}
        self.assertEqual(benchmark.score_answers(legacy, descending)['accuracy_half_ties'], .7)
        for s in benchmark.SPLITS:
            rows = [r for r in benchmark.generate()['rows'] if r['split'] == s]
            self.assertEqual(benchmark.score_answers(rows, descending)['accuracy_half_ties'], .5)
            scores = {str(i): (i * 17) % 7 for i in range(10)}
            self.assertEqual(benchmark.score_answers(rows, scores)['accuracy_half_ties'], .5)

    def test_cli_freeze_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'new'
            cmd = [sys.executable, str(ROOT / 'arithmetic_benchmark.py'), '--output-dir', str(out)]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            snapshot = {p.name: p.read_bytes() for p in out.iterdir()}
            for name in ('dataset.json', 'audit.json'):
                archived = ROOT / 'milestones/unordered-arithmetic-benchmark' / name
                self.assertEqual(snapshot[name], archived.read_bytes())
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(snapshot, {p.name: p.read_bytes() for p in out.iterdir()})


if __name__ == '__main__':
    unittest.main()
