"""Independent arithmetic checks of archived raw model-gate evidence, no model."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'milestones/frozen-evaluation-results'


def read(path):
    return json.loads(path.read_text())


class FrozenGateEvidenceTests(unittest.TestCase):
    def test_exact_training_ids_and_finite_candidate_scores(self):
        expected = [r['id'] for r in read(ROOT / 'milestones/unordered-arithmetic-benchmark/dataset.json')['rows'] if r['split'] == 'train']
        for name in ('run', 'repeat'):
            rows = read(BASE / name / 'gate.json')['records']
            self.assertEqual([r['id'] for r in rows], expected)
            self.assertEqual(len(set(r['id'] for r in rows)), 60)
            for row in rows:
                for key in ('correct_log_likelihood', 'incorrect_log_likelihood'):
                    self.assertTrue(math.isfinite(row[key]))
                    self.assertLessEqual(row[key], 0)
                self.assertEqual(row['likelihood_margin'], row['correct_log_likelihood'] - row['incorrect_log_likelihood'])
                self.assertTrue(row['prefix_token_ids'])
                self.assertEqual(len(row['correct_suffix_token_ids']), len(row['incorrect_suffix_token_ids']))
                self.assertEqual(len(row['correct_suffix_token_ids']), 2)

    def test_independently_recomputed_gate_failure(self):
        for name in ('run', 'repeat'):
            gate = read(BASE / name / 'gate.json')
            groups = defaultdict(list)
            for row in gate['records']:
                groups[tuple(sorted(map(int, row['id'].split('-'))))].append(row['likelihood_margin'] > 0)
            wins = sum(sum(values) for values in groups.values())
            successful_groups = sum(all(values) for values in groups.values())
            self.assertEqual(wins, 30)
            self.assertEqual(successful_groups, 16)
            self.assertEqual(len(groups), 33)
            self.assertEqual(gate['strict_correct_rows'], wins)
            self.assertEqual(gate['all_correct_unordered_groups'], successful_groups)
            self.assertEqual((gate['required_rows'], gate['required_groups']), (45, 24))
            self.assertEqual(gate['passed'], wins >= 45 and successful_groups >= 24)
            self.assertFalse(gate['passed'])

    def test_stop_status_and_absence_of_heldout_artifacts(self):
        for name in ('run', 'repeat'):
            directory = BASE / name
            self.assertEqual({p.name for p in directory.iterdir()}, {'gate.json', 'results.json', 'runtime.json', 'artifact_manifest.json'})
            result = read(directory / 'results.json')
            self.assertEqual(result['status'], 'gate_failed')
            for flag in ('test_evaluated', 'validation_evaluated', 'geometry_fitted', 'causal_interventions'):
                self.assertFalse(result[flag])

    def test_artifact_hashes_and_frozen_source_match(self):
        for name in ('run', 'repeat'):
            directory = BASE / name
            manifest = read(directory / 'artifact_manifest.json')
            self.assertEqual({f['path'] for f in manifest['files']}, {'gate.json', 'results.json', 'runtime.json'})
            for f in manifest['files']:
                payload = (directory / f['path']).read_bytes()
                self.assertEqual(len(payload), f['bytes'])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), f['sha256'])
            runtime = read(directory / 'runtime.json')
            self.assertEqual(runtime['runner_sha256'], hashlib.sha256((ROOT / 'benchmark_runner.py').read_bytes()).hexdigest())
            self.assertEqual(runtime['geometry_sha256'], hashlib.sha256((ROOT / 'geometry.py').read_bytes()).hexdigest())
            result = read(directory / 'results.json')
            self.assertEqual(result['dataset_sha256'], hashlib.sha256((ROOT / 'milestones/unordered-arithmetic-benchmark/dataset.json').read_bytes()).hexdigest())

    def test_repeat_raw_gate_evidence_and_status_byte_identical(self):
        for name in ('gate.json', 'results.json'):
            self.assertEqual((BASE / 'run' / name).read_bytes(), (BASE / 'repeat' / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
