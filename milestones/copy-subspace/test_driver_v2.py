from pathlib import Path
import tempfile
import unittest
import numpy as np
import dataset
import driver_v2 as v2
from test_driver import FakeBackend

class Backend(FakeBackend):
 def forward(self,row,change=None,noop=False):
  self.last_perturbation=None
  def wrapped(h):
   requested=change(h);after=h.astype(np.float32)-requested.astype(np.float32)
   self.last_perturbation=v2.perturbation_norms(h,after,requested);return requested
  return super().forward(row,wrapped if change is not None else None,noop)

class AmendmentTests(unittest.TestCase):
 def test_failed_validation_baseline_prevents_all_interventions(self):
  class Zero(Backend):
   def forward(self,row,change=None,noop=False):
    self.assert_no_change=change is None
    if change is not None:raise AssertionError('No intervention allowed')
    logits,h,c=super().forward(row,change,noop);logits[:]=0;return logits,h,c
  with tempfile.TemporaryDirectory() as d:
   journal=v2.Journal(Path(d)/'run','select',30)
   rows=[r for r in dataset.load()['rows'] if r['split']=='validation']
   result,_=v2.execute('select',rows,Zero(journal),journal,{})
   self.assertEqual(result['status'],'validation_baseline_failed');self.assertEqual(journal.count,40)
 def test_complete_margin_and_greedy_groups_are_distinct(self):
  result=v2.metrics([dict(group='g',margin=1.,greedy_exact=True),dict(group='g',margin=1.,greedy_exact=False)])
  self.assertEqual(result['positive_margin_groups'],1);self.assertEqual(result['greedy_exact_groups'],0)
 def test_realized_float32_norm_can_differ_from_requested(self):
  before=np.array([1e10],dtype=np.float32);requested=np.array([1.]);after=before-requested.astype(np.float32)
  result=v2.perturbation_norms(before,after,requested)
  self.assertEqual(result,{'requested_removed_norm':1.,'realized_removed_norm':0.})
 def test_upstream_exact_training_compatibility(self):
  prior={'status':'train_passed','source_freeze_sha256':v2.APPROVED_TRAIN_SOURCE}
  v2.check_upstream('select',prior,v2.APPROVED_TRAIN_ARCHIVE,'v2digest')
  with self.assertRaises(ValueError):v2.check_upstream('select',prior,'other','v2digest')
  with self.assertRaises(ValueError):v2.check_upstream('test',prior,v2.APPROVED_TRAIN_ARCHIVE,'v2digest')
 def test_v2_test_requires_own_selection_source(self):
  v2.check_upstream('test',{'status':'selection_frozen','source_freeze_sha256':'current'},'digest','current')
  with self.assertRaises(ValueError):v2.check_upstream('test',{'status':'selection_frozen','source_freeze_sha256':'old'},'digest','current')
 def test_selection_and_test_archive_both_norms_and_groups(self):
  fit=v2.fit(np.random.default_rng(7).normal(size=(40,8)))
  with tempfile.TemporaryDirectory() as d:
   journal=v2.Journal(Path(d)/'selection','select',30)
   rows=[r for r in dataset.load()['rows'] if r['split']=='validation']
   result,_=v2.execute('select',rows,Backend(journal),journal,fit)
   self.assertEqual(journal.count,1300);self.assertEqual(result['untouched']['greedy_exact_groups'],10)
   journal=v2.Journal(Path(d)/'test','test',30,1)
   rows=[r for r in dataset.load()['rows'] if r['split']=='test']
   result,_=v2.execute('test',rows,Backend(journal),journal,fit,1)
   self.assertEqual(journal.count,460);self.assertEqual(len(result['condition_metrics']),21)
   for condition in result['condition_metrics']:
    self.assertIn('positive_margin_groups',condition);self.assertIn('greedy_exact_groups',condition)
    self.assertIn('mean_requested_removed_norm',condition);self.assertIn('mean_realized_removed_norm',condition)
if __name__=='__main__':unittest.main()
