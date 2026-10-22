import unittest,tempfile,json
from pathlib import Path
from patchstudy import execution
from patchstudy.storage import sha,json_write
class ExecutionTests(unittest.TestCase):
 def test_acceptance_binds_every_file_and_rejects_extra(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);json_write(root/'status.json',{'status':'complete'})
   json_write(root/'RESULTS-MANIFEST.json',{'status.json':sha(root/'status.json')})
   json_write(root/'SUPERVISOR-RESULT.json',{'status':'accepted','manifest_sha256':sha(root/'RESULTS-MANIFEST.json'),'returncode':0,'stop_reason':None})
   execution.verify_acceptance(root)
   (root/'extra').write_text('bad')
   with self.assertRaises(ValueError):execution.verify_acceptance(root)
 def test_worker_marker_not_acceptance(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);json_write(root/'status.json',{'status':'complete'})
   with self.assertRaises((ValueError,FileNotFoundError)):execution.verify_acceptance(root)
 def test_derived_limits_and_no_budget_expansion(self):
  self.assertEqual(execution.limits('baseline')['wall_seconds'],300)
  self.assertEqual(execution.limits('patch',299)['wall_seconds'],2100)
  self.assertEqual(execution.limits('patch',350)['wall_seconds'],2050)
  with self.assertRaises(ValueError):execution.limits('patch',2400)
 def test_clearance_matches_exact_stage_source_data(self):
  expected={'stage':'baseline','source_sha256':'a','data_freeze_sha256':'b'}
  with self.assertRaises(ValueError):execution.validate_clearance({'approved':True,**expected,'stage':'patch'},expected)
  execution.validate_clearance({'approved':True,**expected},expected)
 def test_fresh_output_lifecycle_is_reserved_parent_new_child(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'phase';execution.reserve_phase(out)
   self.assertEqual({p.name for p in out.iterdir()},{'status.json'})
   self.assertFalse((out/'evidence').exists())
   with self.assertRaises(FileExistsError):execution.reserve_phase(out)
 def test_dependency_hashes_unchanged(self):
  root=Path(__file__).parents[1]/'vendor/supervision'
  self.assertEqual(sha(root/'supervise.py'),'35c91a94456bb23339f971a20c898e0a51140904f720c73dfc4c808be8818429')
  self.assertEqual(sha(root/'evidence.py'),'179877eae7a13c917f6de545e47b5ebeb090028de8f25cd45e954317c1f3463e')
 def test_disk_budget_includes_prior_phase_and_supervisor_directory(self):
  from patchstudy import storage
  import numpy as np
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);storage.PHASE_OUTPUT_ROOT=root;storage.PRIOR_OUTPUT_BYTES=storage.BUDGET['max_output_bytes']-1
   try:
    a=storage.Archive(root/'evidence','baseline')
    with self.assertRaises(RuntimeError):a.array('x.npy',np.zeros(49152,dtype='float32'),(49152,))
   finally:storage.PHASE_OUTPUT_ROOT=None;storage.PRIOR_OUTPUT_BYTES=0
 def test_final_acceptance_rejects_missing_phase_evidence(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);json_write(root/'status.json',{'status':'complete'})
   json_write(root/'RESULTS-MANIFEST.json',{'status.json':sha(root/'status.json')})
   json_write(root/'SUPERVISOR-RESULT.json',{'status':'accepted','manifest_sha256':sha(root/'RESULTS-MANIFEST.json'),'returncode':0,'stop_reason':None})
   with self.assertRaises(FileNotFoundError):execution.accepted_phase(root)
 def test_runtime_module_root_mismatch_fails_before_reading_model(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'Executing module'):execution.frozen_inputs(d,'/not-a-model')
 def test_patch_acceptance_requires_baseline(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'evidence').mkdir()
   json_write(root/'status.json',{'status':'complete'})
   json_write(root/'evidence/run_metadata.json',{'stage':'patch'})
   json_write(root/'RESULTS-MANIFEST.json',{'status.json':sha(root/'status.json'),'evidence/run_metadata.json':sha(root/'evidence/run_metadata.json')})
   json_write(root/'SUPERVISOR-RESULT.json',{'status':'accepted','manifest_sha256':sha(root/'RESULTS-MANIFEST.json'),'returncode':0,'stop_reason':None})
   with self.assertRaisesRegex(ValueError,'explicit baseline'):execution.accepted_phase(root)
 def test_integrated_audit_rejects_resource_failure(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);json_write(root/'SUPERVISOR-RESULT.json',{'status':'rejected','returncode':124,'stop_reason':'wall_budget'})
   result=execution.audit_phase(root)
   self.assertFalse(result['passed']);self.assertFalse(result['execution_accepted'])
 def test_output_must_not_mutate_frozen_or_prior_directories(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   with self.assertRaises(ValueError):execution.disjoint_output(root/'source/run',[root/'source'])
   with self.assertRaises(ValueError):execution.disjoint_output(root,[root/'source'])
   execution.disjoint_output(root/'run',[root/'source',root/'model'])
 def test_upstream_acceptance_checks_identity_not_only_time(self):
  base=dict(source_sha256='s',data_freeze_sha256='d',config_sha256='c',model_files=['m'])
  patch=dict(base,upstream_baseline_manifest_sha256='one')
  with self.assertRaises(ValueError):execution.verify_upstream(patch,base,'two')
  with self.assertRaises(ValueError):execution.verify_upstream(patch,dict(base,model_files=['other']),'one')
  execution.verify_upstream(patch,base,'one')
 def test_runtime_metadata_reads_actual_runtime(self):
  import sys,importlib.metadata
  value=execution.runtime_metadata()
  self.assertEqual(value['python'],sys.version)
  self.assertEqual(value['versions']['numpy'],importlib.metadata.version('numpy'))
