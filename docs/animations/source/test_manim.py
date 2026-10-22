"""Check real-Manim identity, scene geometry, and finite frame timing."""
import ast
import inspect
from pathlib import Path
import json
import unittest
import manim
import numpy as np
import scenes


class ManimContractTests(unittest.TestCase):
    def test_actual_package_and_scene_classes(self):
        self.assertEqual(manim.__version__,'0.22.0')
        self.assertEqual(manim.Scene.__module__,'manim.scene.scene')
        self.assertIn('site-packages',manim.__file__)
        stories=json.loads((Path(__file__).parent/'storyboards.json').read_text())
        for story in stories.values():
            self.assertTrue(issubclass(getattr(scenes,story['class']),manim.Scene))

    def test_three_phase_timing(self):
        self.assertEqual(scenes.DURATION,24)
        self.assertEqual(scenes.config.frame_rate,15)
        stories=json.loads((Path(__file__).parent/'storyboards.json').read_text())
        for story in stories.values():
            self.assertEqual([(s['start'],s['end']) for s in story['scenes']],[(0,8),(8,16),(16,24)])

    def test_animation_durations_fit_whole_frames(self):
        tree=ast.parse(Path(scenes.__file__).read_text())
        durations=[ast.literal_eval(k.value) for node in ast.walk(tree) if isinstance(node,ast.Call)
                   for k in node.keywords if k.arg=='run_time']
        self.assertGreater(len(durations),30)
        for seconds in durations:
            self.assertGreater(seconds,0)
            self.assertAlmostEqual(seconds*15,round(seconds*15))

    def test_translation_plot_has_equal_unit_scales(self):
        node=ast.parse(inspect.getsource(scenes.CoordinateReadout))
        call=next(n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='xy_axes')
        args=[ast.literal_eval(a) for a in call.args[:4]]
        ax=scenes.xy_axes(*args)
        self.assertAlmostEqual(np.linalg.norm(ax.c2p(1,0)-ax.c2p(0,0)),np.linalg.norm(ax.c2p(0,1)-ax.c2p(0,0)))

    def test_stage_annotation_and_point_share_integer_value(self):
        tree=ast.parse(inspect.getsource(scenes.CapacityLatency))
        parents={id(child):node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        reads=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and n.func.attr=='get_value' and isinstance(n.func.value,ast.Name) and n.func.value.id=='tracker']
        self.assertEqual(len(reads),4)  # Point x/y, H label, Tcomm label.
        for read in reads:
            parent=parents[id(read)]
            self.assertIsInstance(parent,ast.Call)
            self.assertEqual(ast.unparse(parent.func),'round')



if __name__=='__main__':unittest.main()
