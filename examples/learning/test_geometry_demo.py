"""Tests for the illustrative, standard-library geometry lesson only."""
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import unittest

PATH = Path(__file__).with_name('geometry_demo.py')


class GeometryLessonTests(unittest.TestCase):
    def load_demo(self):
        self.assertTrue(PATH.exists(), 'The offline geometry lesson is not implemented yet')
        spec = importlib.util.spec_from_file_location('geometry_lesson', PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_translation_preserves_all_pairwise_distances(self):
        demo = self.load_demo()
        points = [(1.0, 0.0), (0.0, 1.0), (-2.0, 3.0), (4.0, -1.0)]
        shifted = [demo.translate(p, (2.0, 2.0)) for p in points]
        for i in range(len(points)):
            for j in range(len(points)):
                self.assertAlmostEqual(demo.distance(points[i], points[j]),
                                       demo.distance(shifted[i], shifted[j]))
        self.assertEqual(points[0], (1.0, 0.0))

    def test_cosine_changes_from_zero_to_twelve_thirteenths(self):
        demo = self.load_demo()
        self.assertEqual(demo.cosine((1, 0), (0, 1)), 0)
        self.assertAlmostEqual(demo.cosine((3, 2), (2, 3)), 12/13)
        self.assertAlmostEqual(demo.distance((1, 0), (0, 1)), math.sqrt(2))
        with self.assertRaises(ValueError):
            demo.cosine((0, 0), (1, 0))

    def test_fixed_readout_changes_but_compensated_readout_recovers_scores(self):
        demo = self.load_demo()
        points = [(1, 0), (0, 1)]
        original = [demo.readout(p, (1, 0), -0.5) for p in points]
        shifted = [demo.translate(p, (2, 2)) for p in points]
        fixed = [demo.readout(p, (1, 0), -0.5) for p in shifted]
        compensated = [demo.readout(p, (1, 0), -2.5) for p in shifted]
        self.assertEqual(original, [0.5, -0.5])
        self.assertEqual(fixed, [2.5, 1.5])
        self.assertEqual(compensated, original)

    def test_projection_can_merge_distinct_labels(self):
        demo = self.load_demo()
        points = [(1, 1), (1, -1)]
        self.assertEqual(demo.distance(*points), 2.0)
        projected = [demo.project_to_x(p) for p in points]
        self.assertEqual(projected, [(1, 0), (1, 0)])
        self.assertEqual(demo.distance(*projected), 0.0)
        self.assertNotEqual(points[0][1] > 0, points[1][1] > 0)

    def test_cli_is_deterministic_and_labels_illustration(self):
        demo = self.load_demo()
        outputs = []
        for _ in range(2):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                demo.main()
            outputs.append(output.getvalue())
        self.assertEqual(outputs[0], outputs[1])
        self.assertIn('ILLUSTRATION', outputs[0])
        self.assertIn('not benchmark evidence', outputs[0])


if __name__ == '__main__':
    unittest.main()
