"""Small synthetic reconstruction safety/coverage tests; no model or subprocess."""
import json,stat,tempfile,unittest,zipfile
from pathlib import Path
from reconstruct import reconstruct,load_map,verify_tree,sha
class ReconstructionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.b=b'synthetic evidence\n';self.h=sha(self.b)
  self.m={'version':1,'format':'byte-exact content-addressed file reconstruction','blob_path':'blobs/<sha256>','files':{'phase/a.bin':{'bytes':len(self.b),'sha256':self.h},'phase/b.bin':{'bytes':len(self.b),'sha256':self.h}},'original_manifest_bytes_unchanged':True}
  self.write_map();self.z=self.root/'blobs.zip';self.zip([(self.h,self.b)])
 def write_map(self):self.mp=self.root/'map.json';self.mp.write_text(json.dumps(self.m));self.mh=sha(self.mp.read_bytes())
 def zip(self,items,mode=stat.S_IFREG|0o644):
  with zipfile.ZipFile(self.z,'w') as z:
   for name,b in items:
    i=zipfile.ZipInfo('blobs/'+name);i.external_attr=mode<<16;z.writestr(i,b)
 def run_it(self):return reconstruct(self.mp,self.mh,[self.z],self.root/'out')
 def test_roundtrip_dedup(self):
  r=self.run_it();self.assertTrue(r['byte_hash_roundtrip']);self.assertEqual(r['files'],2);self.assertEqual((self.root/'out/phase/a.bin').read_bytes(),self.b)
 def test_existing_output_not_overwritten(self):
  o=self.root/'out';o.mkdir();(o/'keep').write_text('keep')
  with self.assertRaises(ValueError):self.run_it()
  self.assertEqual((o/'keep').read_text(),'keep')
 def test_corruption_retains_reserved_output(self):
  self.zip([(self.h,b'x'*len(self.b))])
  with self.assertRaises(ValueError):self.run_it()
  self.assertTrue((self.root/'out').is_dir());self.assertEqual(list((self.root/'out').iterdir()),[])
 def test_missing(self):
  self.zip([])
  with self.assertRaises(ValueError):self.run_it()
 def test_extra(self):
  self.zip([(self.h,self.b),(sha(b'extra'),b'extra')])
  with self.assertRaises(ValueError):self.run_it()
 def test_duplicate_blob_across_archives(self):
  with self.assertRaises(ValueError):reconstruct(self.mp,self.mh,[self.z,self.z],self.root/'out')
 def test_blob_symlink(self):
  self.zip([(self.h,self.b)],stat.S_IFLNK|0o777)
  with self.assertRaises(ValueError):self.run_it()
 def test_path_traversal_and_ambiguous(self):
  for n in ('../a','/abs','a//b','a/./b','a\\b','a:b'):
   with self.subTest(n=n):
    self.m['files']={n:{'bytes':len(self.b),'sha256':self.h}};self.write_map()
    with self.assertRaises(ValueError):load_map(self.mp,self.mh)
 def test_duplicate_json_key(self):
  self.mp.write_text('{"version":1,"version":1}');self.mh=sha(self.mp.read_bytes())
  with self.assertRaises(ValueError):load_map(self.mp,self.mh)
 def test_wrong_trusted_hash(self):
  with self.assertRaises(ValueError):load_map(self.mp,'0'*64)
 def test_file_directory_collision(self):
  self.m['files']['phase']=self.m['files']['phase/a.bin'];self.write_map()
  with self.assertRaises(ValueError):load_map(self.mp,self.mh)
 def test_case_collision(self):
  self.m['files']['phase/A.bin']=self.m['files']['phase/a.bin'];self.write_map()
  with self.assertRaises(ValueError):load_map(self.mp,self.mh)
 def test_declared_size_rejected_before_decompression(self):
  self.m['files']['phase/a.bin']['bytes']=16*1024**2+1;self.write_map()
  with self.assertRaises(ValueError):load_map(self.mp,self.mh)
 def test_extra_reconstructed_file(self):
  self.run_it();(self.root/'out/extra').write_text('extra')
  with self.assertRaises(ValueError):verify_tree(self.root/'out',self.m)
 def test_mutated_reconstructed_file(self):
  self.run_it();(self.root/'out/phase/a.bin').write_text('change')
  with self.assertRaises(ValueError):verify_tree(self.root/'out',self.m)
 def test_no_partial_overwrite_after_io_failure(self):
  # Simulated later file failure retains the first file; no blanket cleanup.
  from unittest.mock import patch
  original=Path.open
  def failing(path,*a,**kw):
   if str(path).endswith('/phase/b.bin') and a and a[0]=='xb':raise OSError('synthetic write failure')
   return original(path,*a,**kw)
  with patch.object(Path,'open',failing):
   with self.assertRaises(OSError):self.run_it()
  self.assertEqual((self.root/'out/phase/a.bin').read_bytes(),self.b)
if __name__=='__main__':unittest.main(verbosity=2)
