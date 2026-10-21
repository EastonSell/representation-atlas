import json,unittest
from collections import Counter
from pathlib import Path
class ResultIntegrityTests(unittest.TestCase):
 def setUp(self):
  self.root=Path(__file__).parent
  self.result=json.loads((self.root/'results/experiment.json').read_text())
  self.pairs={r['id']:r for r in json.loads((self.root/'results/pairs.json').read_text())}
 def test_disjoint_complete_splits(self):
  for r in self.result['rows']:
   tr,te=set(r['train_pair_ids']),set(r['test_pair_ids'])
   self.assertFalse(tr&te);self.assertEqual(tr|te,set(self.pairs));self.assertEqual(len(tr),60);self.assertEqual(len(te),40)
 def test_group_answer_balance(self):
  for r in self.result['rows']:
   if r['split_mode']!='balanced_operand_groups':continue
   for key in ['train_pair_ids','test_pair_ids']:
    rows=[self.pairs[i] for i in r[key]]
    self.assertEqual(Counter(x['correct'] for x in rows),Counter(x['incorrect'] for x in rows))
 def test_forward_repeat(self):
  initial=json.loads((self.root/'results/initial-random-split.json').read_text())
  self.assertEqual(initial['activation_sha256'],self.result['activation_sha256'])
 def test_metrics_valid(self):
  for r in self.result['rows']:
   self.assertEqual(len(r['shuffled_label_rankings']),20)
   for x in [r['pair_ranking'],r['likelihood_correct_preference'],*r['shuffled_label_rankings']]:self.assertTrue(0<=x<=1)
if __name__=='__main__':unittest.main()
