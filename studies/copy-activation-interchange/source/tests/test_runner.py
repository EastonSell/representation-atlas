import unittest,numpy as np
from patchstudy import runner,config
class RunnerTests(unittest.TestCase):
 def test_exact_order(self):
  p=list(config.plan('patch'));self.assertEqual(len(p),2688)
  self.assertEqual([x['condition'] for x in p[:6]],list(config.CONDITIONS));self.assertEqual(p[-1]['row_id'],63)
 def test_control_uses_every_logit(self):
  a=np.zeros(49152,dtype='float32');b=a.copy();b[49000]=.01
  with self.assertRaises(ValueError):runner.identity_error(a,b)
 def test_gate_ties_fail_and_failure_retained(self):
  rows=[dict(row_id=i,recipient_token_id=1,donor_token_id=2,pair_id=i//2) for i in range(64)]
  zs={i:np.zeros(49152,dtype='float32') for i in range(64)}
  g=runner.gate(rows,zs,64);self.assertFalse(g['passed']);self.assertEqual(g['failed_row_ids'],list(range(64)))
 def test_stage_is_fail_closed(self):
  with self.assertRaises(RuntimeError):runner.require_execution_clearance(None,None)
 def test_position_mapping(self):
  r=dict(source_position=6,final_position=12,negative_position=5)
  self.assertEqual(runner.positions(r,'donor_prefix_negative'),[5]);self.assertEqual(runner.positions(r,'donor_all'),'all')
 def test_engine_failure_retains_attempt_and_binding_rejects_tamper(self):
  import tempfile,json
  from pathlib import Path
  from patchstudy.storage import json_write,sha,manifest
  rows=json.loads((Path(__file__).parents[1]/'data/dataset.json').read_text())['rows']
  class FailingAdapter:
   calls=0
   def plain(self,ids):self.calls+=1;raise TimeoutError('synthetic failure')
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);f=root/'frozen';f.mkdir();(f/'data').mkdir()
   json_write(f/'config.json',config.SETTINGS);json_write(f/'data/dataset.json',{'schema_version':1,'rows':rows})
   json_write(f/'SOURCE_MANIFEST.json',{'config.json':{'sha256':sha(f/'config.json'),'bytes':(f/'config.json').stat().st_size}})
   json_write(f/'DATA_FREEZE.json',{'source_sha256':sha(f/'SOURCE_MANIFEST.json'),'rows_sha256':sha(f/'data/dataset.json')})
   meta=dict(stage='baseline',source_sha256=sha(f/'SOURCE_MANIFEST.json'),data_freeze_sha256=sha(f/'DATA_FREEZE.json'),config_sha256=sha(f/'config.json'))
   adapter=FailingAdapter()
   with self.assertRaises(TimeoutError):runner.baseline_engine(adapter,rows,root/'run',meta,f)
   events=[json.loads(x) for x in (root/'run/attempts.jsonl').read_text().splitlines()]
   self.assertEqual([x['status'] for x in events],['planned','failed']);self.assertEqual(adapter.calls,1)
   (f/'config.json').write_text('{}')
   with self.assertRaises(ValueError):runner.baseline_engine(adapter,rows,root/'tamper',meta,f)
   self.assertEqual(adapter.calls,1)
