import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
try:
    import timeline
except ImportError:
    timeline=None

class MeasuredTimelineTests(unittest.TestCase):
    def test_measurement_drives_whole_frame_boundaries(self):
        self.assertIsNotNone(timeline, 'Measured timeline implementation is missing')
        result=timeline.build_timeline({'segments':[{'id':'a','text':'Hello.'},{'id':'b','text':'World.'}]},
            {'segments':[{'id':'a','text':'Hello.','duration':1.011},{'id':'b','text':'World.','duration':2.05}]},30)
        self.assertEqual(result['segments'][0]['frames'],31)
        self.assertEqual(result['segments'][1]['frames'],62)
        self.assertAlmostEqual(result['segments'][1]['start'],31/30)
        self.assertAlmostEqual(result['segments'][-1]['end'],93/30)

    def test_stale_speech_is_rejected(self):
        self.assertIsNotNone(timeline, 'Measured timeline implementation is missing')
        with self.assertRaises(ValueError):
            timeline.build_timeline({'segments':[{'id':'a','text':'New.'}]},
                {'segments':[{'id':'a','text':'Old.','duration':2}]},30)

    def test_missing_and_nonfinite_speech_are_rejected(self):
        self.assertIsNotNone(timeline, 'Measured timeline implementation is missing')
        for rows in [[],[{'id':'a','text':'A.','duration':float('nan')}],[{'id':'a','text':'A.','duration':0}]]:
            with self.assertRaises(ValueError):
                timeline.build_timeline({'segments':[{'id':'a','text':'A.'}]},{'segments':rows},30)

if __name__=='__main__':unittest.main()
