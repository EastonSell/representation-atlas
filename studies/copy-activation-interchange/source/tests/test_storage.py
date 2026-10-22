import unittest,tempfile,numpy as np
from pathlib import Path
from patchstudy import storage,config
class StorageTests(unittest.TestCase):
 def test_fixed_config(self):
  self.assertEqual(len(config.BLOCKS)*len(config.CONDITIONS)*64,2688)
  with self.assertRaises(ValueError):config.validate({'unknown':1})
 def test_attempts_uncertain_count_and_no_retries(self):
  with tempfile.TemporaryDirectory() as d:
   a=storage.Archive(Path(d)/'run','baseline');a.plan(dict(row_id=0,pair_id=0,orientation=0,block=None,condition='untouched'))
   with self.assertRaises(ValueError):a.plan(dict(row_id=0,pair_id=0,orientation=0,block=None,condition='untouched'))
   self.assertEqual(a.count,1)
 def test_full_vector_and_no_overwrite(self):
  with tempfile.TemporaryDirectory() as d:
   a=storage.Archive(Path(d)/'run','baseline')
   with self.assertRaises(ValueError):a.array('x.npy',np.zeros(2,dtype='float32'),(49152,))
   a.array('x.npy',np.zeros(49152,dtype='float32'),(49152,))
   with self.assertRaises(FileExistsError):a.array('x.npy',np.zeros(49152,dtype='float32'),(49152,))
 def test_binding_tamper(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'a').write_text('one');m=storage.manifest(p);storage.verify(p,m)
   (p/'a').write_text('two')
   with self.assertRaises(ValueError):storage.verify(p,m)
 def test_escape_rejected_before_write(self):
  with tempfile.TemporaryDirectory() as d:
   a=storage.Archive(Path(d)/'run','baseline')
   with self.assertRaises(ValueError):a.array('../outside.npy',np.zeros(49152,dtype='float32'),(49152,))
   self.assertFalse((Path(d)/'outside.npy').exists())
 def test_failure_retains_attempt(self):
  import json
  with tempfile.TemporaryDirectory() as d:
   a=storage.Archive(Path(d)/'run','baseline');a.plan(dict(row_id=0,pair_id=0,orientation=0,block=None,condition='untouched'));a.fail(TimeoutError('synthetic timeout'));a.seal()
   events=[json.loads(s) for s in (a.path/'attempts.jsonl').read_text().splitlines()]
   self.assertEqual([r['status'] for r in events],['planned','failed']);self.assertEqual(a.count,1)
