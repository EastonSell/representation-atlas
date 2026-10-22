import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import dataset
import driver

class DatasetTests(unittest.TestCase):
 def test_exact_disjoint_frozen_inputs(self):
  data=dataset.load();audit=dataset.audit(data)
  self.assertTrue(audit['passed']);self.assertEqual([audit['splits'][s]['rows'] for s in ('train','validation','test')],[40,20,20])
 def test_overlap_and_ids_rejected(self):
  data=dataset.load();data['rows'][1]['id']=data['rows'][0]['id']
  with self.assertRaises(ValueError):dataset.audit(data)
  data=dataset.load();data['rows'][40]['correct']=data['rows'][0]['correct']
  with self.assertRaises(ValueError):dataset.audit(data)

class DriverTests(unittest.TestCase):
 def test_scores_use_full_vocabulary_argmax_and_strict_margin(self):
  row={'id':'x','group':'g','split':'train','correct_token_id':0,'incorrect_token_id':1}
  result=driver.scored(row,np.array([3.,2.,4.]))
  self.assertEqual(result['margin'],1.);self.assertFalse(result['greedy_exact']);self.assertEqual(result['greedy_token_id'],2)
  with self.assertRaises(ValueError):driver.scored(row,np.array([float('nan'),0.]))
 def test_mean_preserving_projection_and_energy_control(self):
  h=np.array([4.,5.,6.]);mean=np.array([1.,1.,1.]);q=np.eye(3)[:,:1];r=np.eye(3)[:,1:2]
  np.testing.assert_allclose(driver.delta(h,mean,q),[3.,0.,0.])
  matched=driver.energy_delta(h,mean,q,r)
  self.assertAlmostEqual(np.linalg.norm(matched),3.)
  np.testing.assert_allclose(driver.delta(mean,mean,q),np.zeros(3))
 def test_degenerate_random_control_fails(self):
  with self.assertRaises(ValueError):driver.energy_delta(np.array([1.,0.]),np.zeros(2),np.array([[1.],[0.]]),np.array([[0.],[1.]]))
 def test_train_gate_requires_all_criteria_and_ties_fail(self):
  rows=[r for r in dataset.load()['rows'] if r['split']=='train'];records=[dict(id=r['id'],group=r['group'],margin=1.,greedy_exact=True) for r in rows]
  self.assertTrue(driver.gate(records)['passed'])
  for r in records[:5]:r['margin']=0.
  self.assertFalse(driver.gate(records)['passed'])
 def test_selection_largest_preserving_k(self):
  baseline={'accuracy':1.,'mean_margin':10.}
  candidates={1:{'accuracy':1.,'mean_margin':9.5},2:{'accuracy':.95,'mean_margin':9.},4:{'accuracy':.9,'mean_margin':9.}}
  self.assertEqual(driver.select_k(baseline,candidates),2)
  self.assertEqual(driver.select_k(baseline,{k:{'accuracy':0.,'mean_margin':0.} for k in (1,2,4)}),0)
  with self.assertRaises(ValueError):driver.select_k({'accuracy':1.,'mean_margin':0.},candidates)
 def test_selected_test_requires_matching_digest(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'artifact_manifest.json').write_text('{}')
   with self.assertRaises(ValueError):driver.verify_archive(p,'wrong')
 def test_fresh_output_refuses_existing(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(FileExistsError):driver.fresh_output(Path(d))
 def test_forward_counts_are_bounded(self):
  self.assertEqual(driver.expected_forwards('train'),80)
  self.assertEqual(driver.expected_forwards('select'),1300)
  self.assertEqual(driver.expected_forwards('test',4),460)
  self.assertEqual(driver.expected_forwards('test',0),40)
 def test_random_bases_deterministic_and_orthonormal(self):
  a=driver.fit(np.arange(40*8,dtype=float).reshape(40,8));b=driver.fit(np.arange(40*8,dtype=float).reshape(40,8))
  np.testing.assert_array_equal(a['random_4'],b['random_4'])
  for q in a['random_4']:np.testing.assert_allclose(q.T@q,np.eye(4),atol=1e-12)
class FakeBackend:
 def __init__(self,journal,fail_after=None):self.journal=journal;self.fail_after=fail_after
 def forward(self,row,change=None,noop=False):
  self.journal.before_forward()
  if self.fail_after is not None and self.journal.count>self.fail_after:raise TimeoutError('test budget')
  h=np.arange(8,dtype=np.float32)+int(row['id'].split('-')[1])
  logits=np.zeros(max(row['correct_token_id'],row['incorrect_token_id'])+1,dtype=np.float32)
  logits[row['correct_token_id']]=10.
  captured=h.astype(np.float64) if noop or change is not None else None
  if change is not None:logits[row['correct_token_id']]-=float(np.linalg.norm(change(captured)))*.001
  return logits,h,captured

class StageTests(unittest.TestCase):
 def test_train_stage_uses_only_exact_train_rows_and_saves_binary(self):
  with tempfile.TemporaryDirectory() as d:
   journal=driver.Journal(Path(d)/'run','train',30)
   rows=[r for r in dataset.load()['rows'] if r['split']=='train']
   result,fit=driver.execute('train',rows,FakeBackend(journal),journal)
   self.assertEqual(result['status'],'train_passed');self.assertEqual(journal.count,80)
   self.assertEqual(len(list((journal.out/'arrays').glob('*.npz'))),40)
   self.assertEqual(fit['mean'].shape,(8,))
 def test_validation_stage_all_candidates_with_fixed_controls(self):
  with tempfile.TemporaryDirectory() as d:
   journal=driver.Journal(Path(d)/'run','select',30)
   rows=[r for r in dataset.load()['rows'] if r['split']=='validation']
   fit=driver.fit(np.random.default_rng(1).normal(size=(40,8)))
   result,_=driver.execute('select',rows,FakeBackend(journal),journal,fit)
   self.assertEqual(journal.count,1300);self.assertEqual(result['status'],'selection_frozen')
   self.assertEqual(set(result['candidates']),{1,2,4})
 def test_interruption_preserves_rows(self):
  with tempfile.TemporaryDirectory() as d:
   journal=driver.Journal(Path(d)/'run','train',30)
   rows=[r for r in dataset.load()['rows'] if r['split']=='train']
   with self.assertRaises(TimeoutError):driver.execute('train',rows,FakeBackend(journal,2),journal)
   saved=[json.loads(line) for line in (journal.out/'rows.jsonl').read_text().splitlines()]
   self.assertEqual([r['condition'] for r in saved],['untouched','noop'])
 def test_bad_noop_is_rejected(self):
  class Bad(FakeBackend):
   def forward(self,row,change=None,noop=False):
    logits,h,c=super().forward(row,change,noop)
    if noop:logits[0]+=1.
    return logits,h,c
  with tempfile.TemporaryDirectory() as d:
   journal=driver.Journal(Path(d)/'run','verify-hook',30)
   with self.assertRaises(ValueError):driver.execute('verify-hook',dataset.load()['rows'][:1],Bad(journal),journal)
 def test_zero_selected_test_has_no_perturbations(self):
  with tempfile.TemporaryDirectory() as d:
   journal=driver.Journal(Path(d)/'run','test',30,0)
   rows=[r for r in dataset.load()['rows'] if r['split']=='test']
   result,_=driver.execute('test',rows,FakeBackend(journal),journal,{},0)
   self.assertEqual(journal.count,40);self.assertEqual(result['paired_group_changes'],[])
if __name__=='__main__':unittest.main()

