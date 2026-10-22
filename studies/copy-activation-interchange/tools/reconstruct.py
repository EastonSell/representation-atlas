"""Strict lossless reconstruction from a trusted SHA-bound map and blob ZIPs.
Standard library only; no model imports, subprocesses or network. Not a sandbox
against concurrent filesystem mutation. Use a trusted, otherwise idle workspace.
"""
import argparse,contextlib,hashlib,json,os,re,stat,zipfile
from pathlib import Path
SHA=re.compile(r'[0-9a-f]{64}\Z');NAME=re.compile(r'[A-Za-z0-9_.-]+\Z')
MAX_FILES=10000;MAX_BYTES=1024**3;MAX_BLOB=16*1024**2;MAX_MAP=2*1024**2

def unique(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('Duplicate JSON key')
  d[k]=v
 return d

def path_ok(s):
 if not isinstance(s,str) or not s or any(p in ('','.','..') or not NAME.fullmatch(p) for p in s.split('/')):raise ValueError('Unsafe path')
 return s

def sha(b):return hashlib.sha256(b).hexdigest()
def load_json(b):return json.loads(b,object_pairs_hook=unique)

def load_map(path,expected_sha):
 if not SHA.fullmatch(expected_sha):raise ValueError('Invalid trusted map hash')
 if path.stat().st_size>MAX_MAP:raise ValueError('Map too large')
 b=path.read_bytes()
 if sha(b)!=expected_sha:raise ValueError('Map hash mismatch')
 m=load_json(b)
 if set(m)!= {'version','format','blob_path','files','original_manifest_bytes_unchanged'} or type(m['version']) is not int or m['version']!=1 or m['format']!='byte-exact content-addressed file reconstruction' or m['blob_path']!='blobs/<sha256>' or m['original_manifest_bytes_unchanged'] is not True:raise ValueError('Unsupported map')
 files=m['files'];blobs={};fold=set()
 if not isinstance(files,dict) or not 0<len(files)<=MAX_FILES:raise ValueError('Invalid file count')
 for n,r in files.items():
  path_ok(n)
  if n.casefold() in fold:raise ValueError('Case collision')
  fold.add(n.casefold())
  if not isinstance(r,dict) or set(r)!={'bytes','sha256'} or type(r['bytes']) is not int or not 0<=r['bytes']<=MAX_BLOB or not isinstance(r['sha256'],str) or not SHA.fullmatch(r['sha256']):raise ValueError('Invalid file record')
  if r['sha256'] in blobs and blobs[r['sha256']]!=r['bytes']:raise ValueError('Conflicting blob size')
  blobs[r['sha256']]=r['bytes']
 if sum(r['bytes'] for r in files.values())>MAX_BYTES:raise ValueError('Reconstruction cap exceeded')
 for n in files:
  parts=n.split('/')
  if any('/'.join(parts[:i]) in files for i in range(1,len(parts))):raise ValueError('File/directory collision')
 return m,blobs

def verify_tree(out,m):
 if out.is_symlink():raise ValueError('Symlink root')
 found=set()
 for p in out.rglob('*'):
  if p.is_symlink():raise ValueError('Symlink in reconstructed tree')
  if p.is_file():found.add(p.relative_to(out).as_posix())
  elif not p.is_dir():raise ValueError('Nonregular reconstructed path')
 if found!=set(m['files']):raise ValueError('Reconstructed file coverage mismatch')
 for n,r in m['files'].items():
  p=out/n
  if p.stat().st_size!=r['bytes'] or sha(p.read_bytes())!=r['sha256']:raise ValueError('Reconstructed file mismatch: '+n)
 return {'files':len(found),'bytes':sum(r['bytes'] for r in m['files'].values()),'byte_hash_roundtrip':True}

def verify_phase_links(out,m):
 roots={n.split('/')[0] for n in m['files']};meta={}
 for name in roots:
  root=out/name;entries={n[len(name)+1:]:r for n,r in m['files'].items() if n.startswith(name+'/')}
  outer=load_json((root/'RESULTS-MANIFEST.json').read_bytes())
  if set(outer)!=set(entries)-{'RESULTS-MANIFEST.json','SUPERVISOR-RESULT.json'}:raise ValueError('Outer coverage mismatch')
  if any(outer[n]!=entries[n]['sha256'] for n in outer):raise ValueError('Outer hash mismatch')
  inner=load_json((root/'evidence/manifest.json').read_bytes());expected={n[len('evidence/'):]:r for n,r in entries.items() if n.startswith('evidence/') and n!='evidence/manifest.json'}
  if inner!=expected:raise ValueError('Inner manifest mismatch')
  status=load_json((root/'status.json').read_bytes());result=load_json((root/'SUPERVISOR-RESULT.json').read_bytes())
  if status['status']!='complete' or status['evidence_manifest_sha256']!=entries['evidence/manifest.json']['sha256'] or result['status']!='accepted' or result['returncode']!=0 or result['manifest_sha256']!=entries['RESULTS-MANIFEST.json']['sha256']:raise ValueError('Acceptance linkage mismatch')
  metadata=load_json((root/'evidence/run_metadata.json').read_bytes());meta[metadata['stage']]=(metadata,entries['evidence/manifest.json']['sha256'])
 if set(meta)!={'baseline','patch'} or len(roots)!=2:raise ValueError('Expected exactly baseline and patch')
 if meta['patch'][0]['upstream_baseline_manifest_sha256']!=meta['baseline'][1]:raise ValueError('Upstream mismatch')
 for key in ('source_sha256','data_freeze_sha256','revision','model_files'):
  if meta['baseline'][0][key]!=meta['patch'][0][key]:raise ValueError('Phase provenance mismatch')
 return True

def reconstruct(map_path,expected_sha,zips,out):
 m,blobs=load_map(Path(map_path),expected_sha);out=Path(out)
 if out.exists() or out.is_symlink():raise ValueError('Output must not exist')
 for p in [out.parent,*out.parent.parents]:
  if p.is_symlink():raise ValueError('Output ancestor is a symlink')
 if not out.parent.is_dir():raise ValueError('Output parent must already exist')
 # Reserve before archive checks; a rejected attempt retains its empty/partial root.
 out.mkdir(exist_ok=False)
 with contextlib.ExitStack() as stack:
  members={}
  for zpath in zips:
   z=stack.enter_context(zipfile.ZipFile(zpath))
   if len(z.infolist())>MAX_FILES:raise ValueError('Too many archive members')
   for i in z.infolist():
    n=i.filename
    if i.orig_filename!=n or not n.startswith('blobs/') or not SHA.fullmatch(n[len('blobs/'):]) or n in members:raise ValueError('Unsafe or duplicate blob member')
    digest=n[len('blobs/'):]
    if not stat.S_ISREG(i.external_attr>>16) or i.flag_bits&1 or i.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):raise ValueError('Nonregular/encrypted/unsupported blob')
    if digest not in blobs or i.file_size!=blobs[digest] or i.file_size>MAX_BLOB:raise ValueError('Extra blob or size mismatch')
    members[n]=(z,i)
  if set(members)!={'blobs/'+h for h in blobs}:raise ValueError('Missing blob')
  for n,(z,i) in members.items():
   b=z.read(i)
   if len(b)!=i.file_size or sha(b)!=n[len('blobs/'):]:raise ValueError('Blob hash mismatch')
  for n,r in sorted(m['files'].items()):
   z,i=members['blobs/'+r['sha256']];b=z.read(i)
   if sha(b)!=r['sha256']:raise ValueError('Blob changed during reconstruction')
   dest=out/n;dest.parent.mkdir(exist_ok=True,parents=True)
   with dest.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 return verify_tree(out,m)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--map',type=Path,required=True);p.add_argument('--map-sha256',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--zips',type=Path,nargs='*',default=[]);p.add_argument('--verify-only',action='store_true');p.add_argument('--scientific-phase-links',action='store_true');a=p.parse_args()
 m,_=load_map(a.map,a.map_sha256)
 result=verify_tree(a.out,m) if a.verify_only else reconstruct(a.map,a.map_sha256,a.zips,a.out)
 if a.scientific_phase_links:result['original_phase_manifest_linkage']=verify_phase_links(a.out,m)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
