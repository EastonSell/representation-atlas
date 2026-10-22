"""Append-only attempt evidence and atomic, never-overwritten numeric artifacts."""
import hashlib,json,os,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from .config import BUDGET,plan
PRIOR_OUTPUT_BYTES=0
PHASE_OUTPUT_ROOT=None

def utc():return datetime.now(timezone.utc).isoformat()
def canonical(obj):return json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def json_write(path,obj):
 path=Path(path);payload=canonical(obj)+b'\n'
 with open(path,'xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
def manifest(root):
 return {p.relative_to(root).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(Path(root).rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
def verify(root,entries):
 root=Path(root).resolve()
 for name,r in entries.items():
  p=(root/name).resolve()
  if root not in p.parents or not p.is_file() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('Artifact binding mismatch: '+name)
def append(path,obj):
 with open(path,'ab') as f:f.write(canonical(obj)+b'\n');f.flush();os.fsync(f.fileno())
class Archive:
 def __init__(self,path,stage):
  if stage not in ('baseline','patch'):raise ValueError('Unknown stage')
  self.path=Path(path);self.path.mkdir(parents=False,exist_ok=False);self.stage=stage;self.count=0;self.start=time.monotonic();self.expected=list(plan(stage));self.active=None
 def budget(self,reserve=0):
  if time.monotonic()-self.start>BUDGET[self.stage+'_wall_seconds']:raise TimeoutError('Phase wall cap')
  root=Path(PHASE_OUTPUT_ROOT) if PHASE_OUTPUT_ROOT is not None else self.path
  size=PRIOR_OUTPUT_BYTES+sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
  if size+reserve>BUDGET['max_output_bytes']:raise RuntimeError('Output byte cap')
 def plan(self,cell):
  self.budget()
  if self.active is not None or self.count>=len(self.expected) or cell!=self.expected[self.count]:raise ValueError('Duplicate, out-of-order, outstanding or over-budget attempt')
  self.count+=1;row=dict(schema_version=1,attempt_index=self.count+(128 if self.stage=='patch' else 0),stage=self.stage,**cell,started_utc=utc(),completed_utc=None,status='planned',elapsed_seconds=None,output_file=None,error_type=None)
  append(self.path/'attempts.jsonl',row);self.active=(row,time.monotonic());return row['attempt_index']
 def array(self,name,value,shape):
  self.budget(value.nbytes+8192)
  if value.dtype!=np.float32 or value.shape!=shape or not np.isfinite(value).all():raise ValueError('Invalid full numeric evidence')
  p=(self.path/name).resolve()
  if self.path.resolve() not in p.parents:raise ValueError('Artifact path escapes archive')
  p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():raise FileExistsError(p)
  tmp=p.with_suffix(p.suffix+'.tmp')
  with open(tmp,'xb') as f:np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
  os.link(tmp,p);tmp.unlink();return dict(file=name,sha256=sha(p),shape=list(shape))
 def complete(self,outcome):
  row,start=self.active;outcome=dict(outcome,attempt_index=row['attempt_index'],elapsed_seconds=time.monotonic()-start)
  append(self.path/'outcomes.jsonl',outcome)
  append(self.path/'attempts.jsonl',dict(row,status='completed',completed_utc=utc(),elapsed_seconds=outcome['elapsed_seconds'],output_file=outcome['logits_file']))
  self.active=None
 def fail(self,error):
  if self.active:
   row,start=self.active;append(self.path/'attempts.jsonl',dict(row,status='failed',completed_utc=utc(),elapsed_seconds=time.monotonic()-start,error_type=type(error).__name__))
  json_write(self.path/'failure.json',dict(schema_version=1,error_type=type(error).__name__,message=str(error),complete=False,attempted_count=self.count))
 def seal(self):json_write(self.path/'manifest.json',manifest(self.path))
