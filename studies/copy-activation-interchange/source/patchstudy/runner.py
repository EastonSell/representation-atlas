"""Phase engine. Pre-inference release: executable entry point intentionally locked.

A reviewed external resource-supervision integration is required before enabling
baseline or patch commands. No execution-clearance records belong in evidence.
"""
import json,time,signal
from pathlib import Path
import numpy as np
from .config import BLOCKS,CONDITIONS,SETTINGS,plan
from .dataset import validate
from .storage import Archive,json_write,sha,verify

def require_execution_clearance(clearance,supervision):
 raise RuntimeError('Inference disabled: source/data review and independently reviewed external resource supervisor required')
def identity_error(a,b):
 if a.shape!=(49152,) or b.shape!=(49152,) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid full-vocabulary control')
 error=float(np.max(np.abs(a.astype('float64')-b.astype('float64'))))
 if error>1e-6:raise ValueError('Numerical identity failed: '+str(error))
 return error
def positions(row,condition):
 if condition=='donor_all':return 'all'
 return [row['negative_position'] if condition=='donor_prefix_negative' else row['source_position'] if condition.endswith('word') else row['final_position']]
def gate(rows,logits,readonly_count):
 good=[];positive=[]
 for r in rows:
  z=logits[r['row_id']];good.append(int(np.argmax(z))==r['recipient_token_id']);positive.append(bool(z[r['recipient_token_id']]>z[r['donor_token_id']]))
 competent=[i for i in range(32) if all(good[j] and positive[j] for j in (2*i,2*i+1))]
 complete=len(logits)==64 and readonly_count==64
 return dict(schema_version=1,total_rows=64,total_pairs=32,greedy_correct_count=sum(good),positive_margin_count=sum(positive),jointly_competent_pair_count=len(competent),readonly_identity_pass_count=readonly_count,failed_row_ids=[i for i in range(64) if not(good[i] and positive[i])],failed_pair_ids=[i for i in range(32) if i not in competent],numeric_failure_ids=[],thresholds=dict(greedy_correct_min=58,positive_margin_min=58,jointly_competent_pairs_min=27),passed=bool(complete and sum(good)>=58 and sum(positive)>=58 and len(competent)>=27),complete=complete)
def _row(arc,r,cell,z,base,donor,states=None,norms=None):
 from .metrics import score_logits
 rec=arc.array('logits/%04d.npy'%arc.active[0]['attempt_index'],z,(49152,))
 condition=cell['condition'];expected='none'
 if condition in ('capture_check','identity_word','identity_final','donor_prefix_negative') or (cell['block']==29 and condition=='donor_word'):expected='recipient'
 if condition=='donor_all' or (cell['block']==29 and condition=='donor_final'):expected='donor'
 error=identity_error(z,base if expected=='recipient' else donor) if expected!='none' else None
 result=dict(schema_version=1,**cell,donor_row_id=r['donor_row_id'],logits_file=rec['file'],logits_sha256=rec['sha256'],logits_dtype='float32',logits_shape=[49152],baseline_logits_sha256=None,donor_logits_sha256=None,patched_positions=positions(r,condition) if cell['block'] is not None else [],expected_identity=expected,identity_max_abs_error=error,finite=True,**score_logits(z,base,donor,r['recipient_token_id'],r['donor_token_id']))
 if norms:result.update(norms)
 else:result.update(residual_delta_norm=None,recipient_residual_norm=None,residual_norm_ratio=None)
 if states is not None:
  result['block_state_files']={str(b):arc.array('states/%02d-%02d.npy'%(r['row_id'],b),states[b],(r['prefix_length'],576)) for b in BLOCKS}
 return result

def _bind(arc,rows,metadata,frozen_dir):
 import shutil
 root=Path(frozen_dir)
 source=json.loads((root/'SOURCE_MANIFEST.json').read_text());verify(root,source)
 freeze=json.loads((root/'DATA_FREEZE.json').read_text())
 if sha(root/'SOURCE_MANIFEST.json')!=metadata['source_sha256'] or freeze['source_sha256']!=metadata['source_sha256']:raise ValueError('Source binding mismatch')
 if sha(root/'DATA_FREEZE.json')!=metadata['data_freeze_sha256'] or sha(root/'config.json')!=metadata['config_sha256']:raise ValueError('Config/data freeze mismatch')
 if sha(root/'data/dataset.json')!=freeze['rows_sha256'] or json.loads((root/'data/dataset.json').read_text())['rows']!=rows:raise ValueError('Dataset binding mismatch')
 for source_name,target_name in [('data/dataset.json','dataset.json'),('DATA_FREEZE.json','DATA_FREEZE.json'),('config.json','config.json'),('SOURCE_MANIFEST.json','SOURCE_MANIFEST.json')]:
  shutil.copyfile(root/source_name,arc.path/target_name)
 json_write(arc.path/'run_metadata.json',metadata)

def baseline_engine(adapter,rows,out,metadata,frozen_dir):
 """Injected-adapter engine; production invocation remains blocked by CLI barrier."""
 validate(rows);arc=Archive(out,'baseline');zs={};caches={};readonly=0
 try:
  _bind(arc,rows,metadata,frozen_dir)
  for cell in plan('baseline'):
   r=rows[cell['row_id']];arc.plan(cell)
   if cell['condition']=='untouched':
    z=adapter.plain(r['recipient_prefix_ids']);zs[r['row_id']]=z;states=None
   else:
    z,states=adapter.capture(r['recipient_prefix_ids']);caches[r['row_id']]=states
    identity_error(z,zs[r['row_id']]);readonly+=1
   result=_row(arc,r,cell,z,zs[r['row_id']],zs[r['row_id']],states)
   arc.complete(result)
  for r in rows:
   q=r['negative_position']
   for b in BLOCKS:
    err=np.max(np.abs(caches[r['row_id']][b][q].astype('float64')-caches[r['donor_row_id']][b][q].astype('float64')))
    if not np.isfinite(err) or err>1e-6:raise ValueError('Causal prefix cache identity failed')
  result=gate(rows,zs,readonly);json_write(arc.path/'gate.json',result)
  return result
 except BaseException as e:arc.fail(e);raise
 finally:arc.seal()

def patch_engine(adapter,rows,out,metadata,baseline_path,frozen_dir):
 validate(rows);baseline_path=Path(baseline_path)
 entries=json.loads((baseline_path/'manifest.json').read_text());verify(baseline_path,entries)
 if sha(baseline_path/'manifest.json')!=metadata['upstream_baseline_manifest_sha256']:raise ValueError('Baseline digest differs from reviewed digest')
 if not json.loads((baseline_path/'gate.json').read_text())['passed']:raise ValueError('Baseline gate did not pass')
 br=json.loads((baseline_path/'dataset.json').read_text())['rows']
 if br!=rows:raise ValueError('Baseline dataset differs')
 baseline_meta=json.loads((baseline_path/'run_metadata.json').read_text())
 for k in ('source_sha256','config_sha256','data_freeze_sha256','model_files'):
  if metadata[k]!=baseline_meta[k]:raise ValueError('Cross-phase binding mismatch: '+k)
 records=[json.loads(x) for x in (baseline_path/'outcomes.jsonl').read_text().splitlines()]
 zs={r['row_id']:np.load(baseline_path/r['logits_file'],allow_pickle=False) for r in records if r['condition']=='untouched'}
 states={r['row_id']:{int(b):np.load(baseline_path/v['file'],allow_pickle=False) for b,v in r['block_state_files'].items()} for r in records if r['condition']=='capture_check'}
 if len(zs)!=64 or len(states)!=64:raise ValueError('Incomplete baseline')
 arc=Archive(out,'patch')
 try:
  _bind(arc,rows,metadata,frozen_dir)
  for cell in plan('patch'):
   r=rows[cell['row_id']];b=cell['block'];c=cell['condition'];p=positions(r,c);src=r['row_id'] if c.startswith('identity') else r['donor_row_id'];a=states[r['row_id']][b];replacement=states[src][b]
   aa=a if p=='all' else a[p];bb=replacement if p=='all' else replacement[p];norm=float(np.linalg.norm(aa.astype('float64')));delta=float(np.linalg.norm(bb.astype('float64')-aa.astype('float64')))
   arc.plan(cell);z=adapter.patch(r['recipient_prefix_ids'],b,p,replacement)
   outcome=_row(arc,r,cell,z,zs[r['row_id']],zs[r['donor_row_id']],norms=dict(residual_delta_norm=delta,requested_residual_delta_norm=delta,recipient_residual_norm=norm,residual_norm_ratio=delta/norm if norm else None))
   outcome['realized_residual_delta_norm']=adapter.last_patch_diagnostics['realized_residual_delta_norm']
   if not np.isclose(outcome['realized_residual_delta_norm'],delta,rtol=0,atol=1e-6):raise ValueError('Realized versus cached requested norm mismatch')
   outcome['baseline_logits_sha256']=next(x['logits_sha256'] for x in records if x['row_id']==r['row_id'] and x['condition']=='untouched')
   outcome['donor_logits_sha256']=next(x['logits_sha256'] for x in records if x['row_id']==r['donor_row_id'] and x['condition']=='untouched')
   arc.complete(outcome)
  from .metrics import summarize
  patch_records=[]
  for line in (arc.path/'outcomes.jsonl').read_text().splitlines():
   rec=json.loads(line);rec['logits']=np.load(arc.path/rec['logits_file'],allow_pickle=False);patch_records.append(rec)
  json_write(arc.path/'summary.json',summarize(rows,zs,patch_records))
 except BaseException as e:arc.fail(e);raise
 finally:arc.seal()

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['baseline','patch']);p.parse_args();require_execution_clearance(None,None)
if __name__=='__main__':main()
