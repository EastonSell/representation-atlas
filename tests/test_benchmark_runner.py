"""Orchestration controls use a tiny fake backend, never a pretrained model."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from arithmetic_benchmark import generate
import benchmark_runner as runner


class FakeBackend:
    def __init__(self, passing=True, test_shift=0):
        self.calls = []
        self.passing = passing
        self.test_shift = test_shift

    def evaluate(self, rows, activations=False):
        self.calls.append((rows[0]['split'], activations, [r['id'] for r in rows]))
        result = []
        for i, row in enumerate(rows):
            record = {'id': row['id'], 'correct_log_likelihood': -1.,
                      'incorrect_log_likelihood': -2. if self.passing else -.5}
            if activations:
                offset = self.test_shift if row['split'] == 'test' else 0
                record['positive'] = [1. + offset, i % 3, i % 5, 0.]
                record['negative'] = [-1. + offset, i % 3, i % 5, 0.]
            result.append(record)
        return result


class RunnerTests(unittest.TestCase):
    def test_failed_gate_never_extracts_or_observes_heldout(self):
        backend = FakeBackend(passing=False)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'run'
            result = runner.run_evaluation(generate(), backend, path)
            self.assertEqual(result['status'], 'gate_failed')
            self.assertEqual([(s, a) for s, a, _ in backend.calls], [('train', False)])
            self.assertFalse((path / 'selection.json').exists())
            self.assertFalse((path / 'activations.npz').exists())
            self.assertEqual(len(json.loads((path / 'gate.json').read_text())['records']), 60)

    def test_pass_archives_selection_before_test_and_train_only_fit(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'run'
            backend = FakeBackend()
            original = backend.evaluate
            def checked(rows, activations=False):
                if rows[0]['split'] == 'test':
                    self.assertTrue((path / 'selection.json').exists())
                return original(rows, activations)
            backend.evaluate = checked
            result = runner.run_evaluation(generate(), backend, path)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual([(s, a) for s, a, _ in backend.calls], [('train', False), ('train', True), ('validation', True), ('test', True)])
            states = json.loads((path / 'selection.json').read_text())
            self.assertEqual(states['selected_remove_top_pcs'], 0)
            self.assertEqual(len(result['test_records']), 20)
            saved = np.load(path / 'activations.npz', allow_pickle=False)
            self.assertEqual(saved['train_positive'].shape, (60, 4))
            np.testing.assert_allclose(states['mean'], np.concatenate([saved['train_positive'], saved['train_negative']]).mean(axis=0))
            for row in result['test_records']:
                self.assertIn('selected_geometry_margin', row)
                self.assertIn('likelihood_margin', row)
            self.assertEqual(len(states['validation_candidates']), 3)

    def test_test_values_cannot_change_selection(self):
        with tempfile.TemporaryDirectory() as d:
            selections = []
            for i, shift in enumerate((0, 10000)):
                path = Path(d) / str(i)
                runner.run_evaluation(generate(), FakeBackend(test_shift=shift), path)
                selections.append(json.loads((path / 'selection.json').read_text()))
            self.assertEqual(selections[0], selections[1])

    def test_gate_threshold_and_group_failures(self):
        rows = [r for r in generate()['rows'] if r['split'] == 'train']
        records = FakeBackend().evaluate(rows)
        self.assertTrue(runner.capability_gate(rows, records)['passed'])
        for record in records[:16]:
            record['correct_log_likelihood'] = record['incorrect_log_likelihood']
        self.assertFalse(runner.capability_gate(rows, records)['passed'])
        self.assertEqual(runner.capability_gate(rows, records)['strict_correct_rows'], 44)

    def test_gate_requires_group_success_even_when_row_threshold_passes(self):
        rows = [r for r in generate()['rows'] if r['split'] == 'train']
        records = FakeBackend().evaluate(rows)
        seen = set()
        for row, record in zip(rows, records):
            key = tuple(sorted((row['a'], row['b'])))
            if key not in seen and len(seen) < 15:
                seen.add(key)
                record['correct_log_likelihood'] = record['incorrect_log_likelihood']
        gate = runner.capability_gate(rows, records)
        self.assertEqual(gate['strict_correct_rows'], 45)
        self.assertEqual(gate['all_correct_unordered_groups'], 18)
        self.assertFalse(gate['passed'])

    def test_invalid_backend_evidence_rejected(self):
        rows = [r for r in generate()['rows'] if r['split'] == 'train']
        records = FakeBackend().evaluate(rows, True)
        for field, value in [('correct_log_likelihood', float('nan')), ('id', 'bad'), ('positive', [float('inf')]*4), ('negative', [1, 2])]:
            bad = copy.deepcopy(records)
            bad[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.validate_records(rows, bad, activations=True)
        with self.assertRaises(ValueError):
            runner.validate_records(rows, records[:-1], activations=True)

    def test_existing_output_not_touched_and_backend_not_called(self):
        with tempfile.TemporaryDirectory() as d:
            backend = FakeBackend()
            with self.assertRaises(FileExistsError):
                runner.run_evaluation(generate(), backend, Path(d))
            self.assertEqual(backend.calls, [])

    def test_exception_has_incomplete_artifact_not_completed(self):
        class Broken(FakeBackend):
            def evaluate(self, rows, activations=False):
                raise RuntimeError('test failure')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'run'
            with self.assertRaises(RuntimeError):
                runner.run_evaluation(generate(), Broken(), path)
            self.assertEqual(json.loads((path / 'failure.json').read_text())['status'], 'incomplete')
            self.assertFalse((path / 'results.json').exists())

    def test_tokenizer_preflight_rejects_prefix_and_length_changes(self):
        rows = generate()['rows']
        class Characters:
            def encode(self, text, add_special_tokens=False):
                return [ord(c) for c in text]
        encoded = runner.encode_candidates(Characters(), rows)
        self.assertEqual(len(encoded), 100)
        self.assertEqual(encoded[rows[0]['id']][1][0][-1], ord(rows[0]['correct']))
        class BadPrefix(Characters):
            def encode(self, text, add_special_tokens=False):
                return super().encode(text)[::-1]
        with self.assertRaises(ValueError):
            runner.encode_candidates(BadPrefix(), rows)
        class TooLong(Characters):
            def encode(self, text, add_special_tokens=False):
                return super().encode(text) * 10
        with self.assertRaises(ValueError):
            runner.encode_candidates(TooLong(), rows)
        class Unequal(Characters):
            def encode(self, text, add_special_tokens=False):
                result = super().encode(text)
                return result + ([999] if text.endswith('5') else [])
        with self.assertRaises(ValueError):
            runner.encode_candidates(Unequal(), rows)

    def test_frozen_dataset_and_missing_model_fail_closed(self):
        self.assertEqual(len(runner.load_frozen_dataset()['rows']), 100)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises((ValueError, FileNotFoundError)):
                runner.verify_model(Path(d))


if __name__ == '__main__':
    unittest.main()
