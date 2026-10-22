import unittest
from patchstudy import report
class ReportTests(unittest.TestCase):
 def test_no_results_invented(self):
  s={'complete':False,'observed_patch_count':0,'missing_cells':[1], 'cells':[]}
  text=report.render(s,{'passed':False})
  self.assertIn('Incomplete',text);self.assertIn('No mechanistic conclusion',text)
 def test_endpoint_labeled(self):
  s={'complete':True,'observed_patch_count':2688,'missing_cells':[], 'cells':[dict(block=29,condition='donor_word',complete=True,mean_raw_transfer_delta=0,median_raw_transfer_delta=0,observed_count=64)]}
  self.assertIn('architectural control',report.render(s,{'passed':True}))
 def test_scientific_audit_without_resource_acceptance_cannot_claim_complete(self):
  s={'complete':True,'observed_patch_count':2688,'missing_cells':[], 'cells':[]}
  self.assertIn('No mechanistic conclusion',report.render(s,{'passed':True}))
