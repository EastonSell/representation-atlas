import tempfile,unittest,json
from pathlib import Path
import gate
class Repairs(unittest.TestCase):
 def test_rejects_incomplete_pair(self):
  self.assertTrue(hasattr(gate,'complete_pair_count'))
  with self.assertRaises(ValueError):gate.complete_pair_count([dict(pair='p',answer='red',constrained_correct=True)])
 def test_rejects_duplicate_answers(self):
  self.assertTrue(hasattr(gate,'complete_pair_count'))
  with self.assertRaises(ValueError):gate.complete_pair_count([dict(pair='p',answer='red',constrained_correct=True)]*2)
 def test_artifact_enforcement(self):
  self.assertTrue(hasattr(gate,'enforce_artifact_cap'))
  with tempfile.TemporaryDirectory() as p:
   Path(p,'a').write_bytes(b'0123456789')
   self.assertEqual(gate.enforce_artifact_cap(Path(p),limit=10),10)
   with self.assertRaises(RuntimeError):gate.enforce_artifact_cap(Path(p),limit=9)
   with self.assertRaises(RuntimeError):gate.enforce_artifact_cap(Path(p),limit=10,additional=1)
 def test_replay(self):
  self.assertTrue(hasattr(gate,'summarize'))
  r=json.loads((Path(__file__).resolve().parent/'results.json').read_text())
  got=gate.summarize(r['rows'],r['tokenizer_valid'])
  self.assertEqual(got,{k:r[k] for k in ('constrained_correct','free_exact','both_pair_correct','gate_pass')})
if __name__=='__main__':unittest.main()
