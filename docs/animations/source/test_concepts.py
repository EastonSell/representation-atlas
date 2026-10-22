"""Deterministic mathematical and timeline checks for educational animation."""
import math
import unittest
import concepts as c


class ConceptTests(unittest.TestCase):
    def test_replay_tasks_are_compatible(self):
        self.assertEqual(c.loss((1., -1.), c.TASK_A), 0.)
        self.assertEqual(c.loss((1., -1.), c.TASK_B), 0.)

    def test_matched_toy_budgets_and_forgetting(self):
        r = c.replay_results()
        self.assertAlmostEqual(r['current_only']['weights'][0], .5, places=4)
        self.assertGreater(r['current_only']['a_loss'], .12)
        self.assertLess(r['replay']['a_loss'], 1e-5)
        self.assertEqual(r['current_only']['updates'], r['replay']['updates'])
        self.assertEqual(r['current_only']['presentations'], 240)
        self.assertEqual(r['replay']['stored_old_examples'], 1)

    def test_selector_rankings_use_signed_loss_change(self):
        rows = c.selection_scores()
        self.assertEqual(max(rows, key=lambda r:r['now'])['name'], 'Hard now')
        self.assertEqual(max(rows, key=lambda r:r['delta'])['name'], 'Threatened next')
        self.assertAlmostEqual(rows[1]['delta'], -.21875)
        self.assertAlmostEqual(rows[2]['delta'], .125)

    def test_geometry_distance_cosine_readout_and_projection(self):
        a, b = (1,0), (0,1)
        ta, tb = (3,2), (2,3)
        self.assertAlmostEqual(math.dist(a,b), math.dist(ta,tb))
        self.assertEqual(c.cosine(a,b), 0.)
        self.assertAlmostEqual(c.cosine(ta,tb), 12/13)
        self.assertEqual(c.readout(b,-.5), -.5)
        self.assertEqual(c.readout(tb,-.5), 1.5)
        self.assertEqual(c.readout(tb,-2.5), -.5)
        self.assertEqual(c.project((1,1)), c.project((1,-1)))

    def test_templates_preserve_exact_literals_not_truth(self):
        for name in c.REGISTERS:
            text = c.format_facts(c.FACTS, name)
            self.assertTrue(all(value in text for value in c.FACTS.values()))
        wrong = dict(c.FACTS, place='Bay 9')
        self.assertIn('Bay 9', c.format_facts(wrong, 'working'))
        self.assertNotIn('Bay 3', c.format_facts(wrong, 'working'))

    def test_capacity_units_and_lower_bounds(self):
        m = c.mesh_numbers()
        self.assertEqual(m['weights_bytes'], 35_000_000_000)
        self.assertEqual(m['kv_bytes'], 2.5*c.GIB)
        self.assertAlmostEqual(m['envelope_gib'], 43.6155481338501, places=8)
        self.assertEqual((m['weight_pools'],m['envelope_pools']), (3,4))
        self.assertAlmostEqual(m['crossing_s'], .0265536)
        self.assertAlmostEqual(m['serial_4_s'], .0796608)
        self.assertAlmostEqual(m['serial_16_s'], .398304)
        self.assertAlmostEqual(m['ceiling_16'], 1/.398304)

    def test_lost_reply_contract_cases(self):
        self.assertEqual(c.retry_cases(), [2,1,2,1])
        for dedup, life in [(False,None),(True,None),(True,1)]:
            s = c.ToyService(deduplicates=dedup,lifetime=life)
            with self.assertRaises(c.LostReply) as caught:
                s.submit('op-1','add A',now=0,lose='after')
            self.assertEqual(str(caught.exception), 'reply unavailable')
        service=c.ToyService()
        service.submit('op-1','add A',now=0)
        with self.assertRaises(ValueError):
            service.submit('op-1','add B',now=1)

    def test_timelines_cover_full_duration_without_overlap(self):
        for name,story in c.STORIES.items():
            self.assertEqual(story['scenes'][0]['start'],0)
            self.assertEqual(story['scenes'][-1]['end'],c.DURATION)
            for a,b in zip(story['scenes'],story['scenes'][1:]):
                self.assertEqual(a['end'],b['start'])
            self.assertTrue(all(s['end']-s['start'] >= 4 for s in story['scenes']))
            self.assertEqual(c.FPS*c.DURATION,288)


if __name__=='__main__': unittest.main()
