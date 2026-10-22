"""Behavioral checks for the metadata-only split-confound control."""
import copy
import json
from pathlib import Path
import unittest
import subprocess
import sys
import tempfile

import split_audit as audit

ROOT = Path(__file__).resolve().parents[1]


class SplitAuditTests(unittest.TestCase):
    def setUp(self):
        self.pairs = [
            {'id': '0-1', 'prompt': 'p', 'correct': '1', 'incorrect': '4'},
            {'id': '1-0', 'prompt': 'q', 'correct': '1', 'incorrect': '4'},
            {'id': '2-2', 'prompt': 'r', 'correct': '4', 'incorrect': '1'},
        ]

    def test_train_only_baseline_and_half_credit_ties(self):
        scores = audit.fit_answer_scores(self.pairs[:1])
        self.assertEqual(scores, {'1': 1, '4': -1})
        result = audit.score_answers(self.pairs[1:], scores)
        self.assertEqual(result['accuracy_half_ties'], 0.5)
        self.assertEqual(result['wins'], 1)
        self.assertEqual(audit.score_answers(self.pairs, {})['ties'], 3)
        self.assertEqual(audit.score_answers(self.pairs, {})['accuracy_half_ties'], 0.5)

    def test_composition_total_variation(self):
        self.assertEqual(audit.composition(self.pairs[:1])['label_answer_tv'], 1.0)
        self.assertEqual(audit.composition(self.pairs[1:])['label_answer_tv'], 0.0)

    def test_commutative_overlap_not_hidden_by_distinct_ids(self):
        result = audit.audit_split(self.pairs, ['0-1'], ['1-0', '2-2'])
        self.assertEqual(result['overlap']['commutative_problem_keys'], ['0-1'])
        self.assertEqual(result['overlap']['pair_ids'], [])

    def test_reject_invalid_partition_and_duplicate_ids(self):
        cases = [(['0-1', '0-1'], ['1-0', '2-2']),
                 (['0-1'], ['0-1', '1-0', '2-2']),
                 (['missing'], ['1-0', '2-2']),
                 (['0-1'], ['1-0']), ([], ['0-1', '1-0', '2-2'])]
        for train, test in cases:
            with self.subTest(train=train, test=test), self.assertRaises(ValueError):
                audit.audit_split(self.pairs, train, test)
        with self.assertRaises(ValueError):
            audit.audit_split(self.pairs + [self.pairs[0]], ['0-1'], ['1-0', '2-2'])

    def test_duplicate_prompt_flagged(self):
        pairs = copy.deepcopy(self.pairs)
        pairs[1]['prompt'] = pairs[0]['prompt']
        self.assertEqual(audit.audit_split(pairs, ['0-1'], ['1-0', '2-2'])['overlap']['prompt_count'], 1)

    def test_archived_splits_are_deduplicated_and_balanced(self):
        report = audit.build_report(ROOT)
        self.assertEqual(len(report['splits']), 6)
        for row in report['splits']:
            self.assertEqual(row['projection_rows_audited'], 3)
            if row['split_mode'] == 'balanced_operand_groups':
                self.assertEqual(row['train']['label_answer_tv'], 0)
                self.assertEqual(row['test']['label_answer_tv'], 0)
                self.assertEqual(row['answer_only_test']['accuracy_half_ties'], .5)
                self.assertEqual(row['overlap']['first_operands'], [])
                self.assertGreater(len(row['overlap']['commutative_problem_keys']), 0)
        self.assertEqual(report, audit.build_report(ROOT))

    def test_cli_reproduces_archive_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'audit.json'
            command = [sys.executable, str(ROOT / 'split_audit.py'), '--output', str(target)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            expected = (ROOT / 'milestones/split-confound-audit/results.json').read_bytes()
            self.assertEqual(target.read_bytes(), expected)
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(target.read_bytes(), expected)

    def test_projection_split_drift_rejected(self):
        pairs = json.loads((ROOT / audit.PAIRS).read_text())
        experiment = json.loads((ROOT / audit.EXPERIMENT).read_text())
        experiment['rows'][1]['train_pair_ids'].reverse()
        audit.audit_experiment(pairs, experiment)  # ordering is not membership
        experiment['rows'][1]['train_pair_ids'][0] = experiment['rows'][1]['test_pair_ids'][0]
        with self.assertRaises(ValueError):
            audit.audit_experiment(pairs, experiment)


if __name__ == '__main__':
    unittest.main()
