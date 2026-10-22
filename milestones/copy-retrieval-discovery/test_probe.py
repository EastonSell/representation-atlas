import collections
import copy
import json
from pathlib import Path
import unittest
import probe

class ProbeTests(unittest.TestCase):
 def test_inputs_are_balanced_and_counterfactual(self):
  data=probe.load_inputs()
  self.assertEqual(data['priority'],['copy','retrieval','continuation'])
  self.assertEqual(len(data['rows']),72)
  self.assertEqual(len({r['id'] for r in data['rows']}),72)
  for task in data['priority']:
   rows=[r for r in data['rows'] if r['task']==task]
   self.assertEqual(set(r['split'] for r in rows),{'discovery_train'})
   self.assertEqual(set(collections.Counter(r['correct'] for r in rows).values()),{2})
   self.assertEqual(collections.Counter(r['correct'] for r in rows),collections.Counter(r['incorrect'] for r in rows))
   for group in {r['group'] for r in rows}:
    a,b=[r for r in rows if r['group']==group]
    self.assertEqual((a['correct'],a['incorrect']),(b['incorrect'],b['correct']))
 def records(self):
  return [dict(id=r['id'],margin=1.,greedy_exact=True) for r in probe.load_inputs()['rows']]
 def test_priority_not_best_score(self):
  records=self.records();records[0]['margin']=-1;records[0]['greedy_exact']=False
  result=probe.summarize(probe.load_inputs(),records)
  self.assertEqual(result['selected_task'],'copy')
 def test_ties_and_nonmatching_greedy_fail(self):
  records=self.records()
  for r in records:r.update(margin=0.,greedy_exact=False)
  result=probe.summarize(probe.load_inputs(),records)
  self.assertIsNone(result['selected_task'])
  self.assertEqual(result['candidates'][0]['ties'],24)
  self.assertEqual(result['candidates'][0]['strict_correct_rows'],0)
 def test_thresholds_are_conjunctive(self):
  records=self.records()
  for r in records[:5]:r['greedy_exact']=False
  self.assertEqual(probe.summarize(probe.load_inputs(),records)['selected_task'],'retrieval')
 def test_record_order_and_missing_rows_rejected(self):
  for records in [self.records()[:-1],list(reversed(self.records()))]:
   with self.assertRaises(ValueError):probe.summarize(probe.load_inputs(),records)
 def test_nonfinite_margin_rejected(self):
  records=self.records();records[0]['margin']=float('nan')
  with self.assertRaises(ValueError):probe.summarize(probe.load_inputs(),records)
if __name__=='__main__':unittest.main()
