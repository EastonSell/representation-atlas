"""Locally frozen, staged pre-answer copying subspace evaluation. Offline only."""
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import dataset
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from benchmark_runner import verify_model
KS=(1,2,4)
BLOCK=14
DEFAULT_BUDGET={'verify-hook':120.,'train':300.,'select':900.,'test':600.}

def save(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def fresh_output(path):Path(path).mkdir(parents=True,exist_ok=False)
def expected_forwards(stage,k=4):return {'verify-hook':2,'train':80,'select':1300,'test':40 if k==0 else 460}[stage]
def delta(h,mean,q):return q@(q.T@(h-mean))
def energy_delta(h,mean,q,r):
 target=delta(h,mean,q);random=delta(h,mean,r);n=float(np.linalg.norm(random));energy=float(np.linalg.norm(target))
 if energy==0:return np.zeros_like(target)
 if n<1e-12:raise ValueError('Degenerate energy-matched random control')
 return random*(energy/n)
def fit(states):
 states=np.asarray(states,dtype=np.float64)
 if states.ndim!=2 or len(states)!=40 or states.shape[1]<4 or not np.isfinite(states).all():raise ValueError('Invalid training states')
 mean=states.mean(0);_,s,v=np.linalg.svd(states-mean,full_matrices=False)
 result={'mean':mean,'pcs':v[:4].T,'singular_values':s}
 for k in KS:result['random_'+str(k)]=np.stack([np.linalg.qr(np.random.default_rng(seed).normal(size=(states.shape[1],k)),mode='reduced')[0] for seed in range(20261022,20261032)])
 return result

def gate(records):
 if len(records)!=40 or len({r['id'] for r in records})!=40:raise ValueError('Expected all 40 training IDs')
 groups=defaultdict(list)
 for r in records:
  if not math.isfinite(r['margin']) or type(r['greedy_exact']) is not bool:raise ValueError('Invalid gate evidence')
  groups[r['group']].append(r['margin']>0)
 if len(groups)!=20 or any(len(v)!=2 for v in groups.values()):raise ValueError('Expected 20 paired training groups')
 wins=sum(r['margin']>0 for r in records);greedy=sum(r['greedy_exact'] for r in records);complete=sum(all(v) for v in groups.values())
 return dict(passed=wins>=36 and greedy>=36 and complete>=18,strict_wins=wins,greedy_exact=greedy,complete_groups=complete,rows=40,groups=20,ties=sum(r['margin']==0 for r in records))
def metrics(records):return {'accuracy':sum(r['greedy_exact'] for r in records)/len(records),'mean_margin':sum(r['margin'] for r in records)/len(records)}
def select_k(baseline,candidates):
 if baseline['mean_margin']<=0:raise ValueError('Validation baseline mean margin must be positive')
 return max([0]+[k for k in KS if candidates[k]['accuracy']+1e-12>=baseline['accuracy']-.05 and candidates[k]['mean_margin']+1e-12>=.9*baseline['mean_margin']])
def verify_archive(path,expected):
 path=Path(path)
 if dataset.sha(path/'artifact_manifest.json')!=expected:raise ValueError('Archive digest mismatch')
 manifest=json.loads((path/'artifact_manifest.json').read_text())
 for entry in manifest['files']:
  name=entry['path']
  if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe archive path')
  f=path/name
  if f.stat().st_size!=entry['bytes'] or dataset.sha(f)!=entry['sha256']:raise ValueError('Archive content mismatch')
 return json.loads((path/'result.json').read_text())
def seal(path):
 save(path/'artifact_manifest.json',{'files':[{'path':str(f.relative_to(path)),'bytes':f.stat().st_size,'sha256':dataset.sha(f)} for f in sorted(path.rglob('*')) if f.is_file()]})
def verify_source():
 freeze=HERE/'SOURCE_FREEZE.json'
 data=json.loads(freeze.read_text())
 for name,expected in data['sha256'].items():
  if dataset.sha(ROOT/name)!=expected:raise ValueError('Frozen source mismatch: '+name)
 return dataset.sha(freeze)

class Journal:
 def __init__(self,out,stage,budget,k=4):
  self.out=Path(out);fresh_output(self.out);(self.out/'arrays').mkdir();self.deadline=time.monotonic()+budget;self.count=0
  self.expected=expected_forwards(stage,k)
  save(self.out/'plan.json',dict(stage=stage,maximum_forwards=self.expected,budget_seconds=budget,budget_type='Cooperative check before each forward; one forward may finish beyond deadline',threads=2,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),outputs_incremental=True))
 def before_forward(self):
  if time.monotonic()>=self.deadline:raise TimeoutError('Cooperative inference budget exhausted')
  if self.count>=self.expected:raise RuntimeError('Forward count exceeded plan')
  self.count+=1
 def row(self,record):
  with (self.out/'rows.jsonl').open('a') as f:f.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 def array(self,id,logits,state):
  path=self.out/'arrays'/(id+'.npz')
  with path.open('xb') as f:np.savez_compressed(f,logits=logits.astype(np.float32),state=state.astype(np.float32))
  return {'path':str(path.relative_to(self.out)),'sha256':dataset.sha(path)}

class Backend:
 def __init__(self,path,journal):
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
  import torch,transformers
  if torch.__version__!='2.5.1+cpu' or transformers.__version__!='4.46.3' or np.__version__!='2.3.5':raise ValueError('Untested runtime')
  self.manifest=verify_model(path);torch.set_num_threads(2);self.torch=torch;self.journal=journal
  self.model=transformers.AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,token=False,use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager');self.model.eval()
  if self.model.config.num_hidden_layers!=30 or self.model.config.hidden_size!=576:raise ValueError('Unexpected architecture')
  self.versions={'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__}
 def forward(self,row,change=None,noop=False):
  self.journal.before_forward();torch=self.torch;captured=[]
  def hook(module,args,output):
   if not isinstance(output,tuple):raise ValueError('Unexpected decoder output')
   h=output[0][0,-1].detach().cpu().numpy().astype(np.float64);captured.append(h)
   if noop:return output
   updated=output[0].clone();updated[0,-1]-=torch.as_tensor(change(h),dtype=updated.dtype,device=updated.device)
   return (updated,)+output[1:]
  handle=self.model.model.layers[BLOCK].register_forward_hook(hook) if noop or change is not None else None
  try:
   ids=torch.tensor([row['prefix_token_ids']],dtype=torch.long)
   with torch.inference_mode():out=self.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,output_hidden_states=True,return_dict=True)
   logits=out.logits[0,-1].detach().cpu().numpy().copy();h=out.hidden_states[BLOCK+1][0,-1].detach().cpu().numpy().copy()
   if handle is not None and len(captured)!=1:raise ValueError('Hook count mismatch')
   return logits,h,captured[0] if captured else None
  finally:
   if handle is not None:handle.remove()

def scored(row,logits):
 if logits.ndim!=1 or not np.isfinite(logits).all():raise ValueError('Invalid logits')
 a,b=row['correct_token_id'],row['incorrect_token_id'];best=int(np.argmax(logits));x=logits.astype(np.float64);z=float(np.max(x)+np.log(np.exp(x-np.max(x)).sum()))
 return dict(id=row['id'],group=row['group'],split=row['split'],margin=float(x[a]-x[b]),greedy_token_id=best,greedy_exact=best==a,target_probability=float(np.exp(x[a]-z)))
def baseline(row,backend,journal):
 logits,h,_=backend.forward(row);record=scored(row,logits);record.update(condition='untouched',array=journal.array(row['id'],logits,h),state_norm=float(np.linalg.norm(h)));journal.row(record)
 noop,nh,captured=backend.forward(row,noop=True);error=float(np.max(np.abs(noop-logits)));state_error=float(np.max(np.abs(captured-h)))
 journal.row(dict(id=row['id'],condition='noop',max_logit_error=error,max_state_error=state_error))
 if error>1e-6 or state_error>1e-6 or not np.array_equal(nh,h):raise ValueError('No-op or block-index verification failed')
 return record,h.astype(np.float64)
def interventions(row,backend,journal,fitdata,ks):
 records=[]
 for k in ks:
  q=fitdata['pcs'][:,:k];mean=fitdata['mean'];conditions=[('pca',None,q)]
  for index,r in enumerate(fitdata['random_'+str(k)]):conditions.extend([('rank_random',index,r),('energy_random',index,r)])
  for name,index,r in conditions:
   removed=[]
   def change(h):
    value=energy_delta(h,mean,q,r) if name=='energy_random' else delta(h,mean,r)
    removed.append(float(np.linalg.norm(value)));return value
   logits,_,_=backend.forward(row,change=change);record=scored(row,logits);record.update(condition=name,k=k,random_seed=None if index is None else 20261022+index,removed_norm=removed[0]);journal.row(record);records.append(record)
 return records

def execute(stage,rows,backend,journal,fitdata=None,k=0):
 bases=[];states=[];changed=[]
 for row in rows:
  b,h=baseline(row,backend,journal);bases.append(b);states.append(h)
  if stage in ('select','test'):changed.extend(interventions(row,backend,journal,fitdata,KS if stage=='select' else (() if k==0 else (k,))))
 if stage=='verify-hook':return {'status':'hook_verified','stage':stage,'rows':1,'site':{'decoder_block':BLOCK,'position':'last_prompt_token','hidden_states_index':BLOCK+1}},None
 if stage=='train':
  g=gate(bases)
  return {'status':'train_passed' if g['passed'] else 'gate_failed','stage':stage,'gate':g,'train_ids':[r['id'] for r in rows]},fit(states) if g['passed'] else None
 if stage=='select':
  base=metrics(bases);candidates={candidate:metrics([r for r in changed if r['condition']=='pca' and r['k']==candidate]) for candidate in KS}
  if base['mean_margin']<=0:return {'status':'validation_baseline_failed','stage':stage,'untouched':base,'candidates':candidates},None
  selected=select_k(base,candidates)
  return {'status':'selection_frozen','stage':stage,'selected_k':selected,'untouched':base,'candidates':candidates,'validation_ids':[r['id'] for r in rows]},None
 groups=defaultdict(list)
 lookup={r['id']:r for r in bases}
 for r in changed:groups[(r['condition'],r['random_seed'],r['group'])].append(r['margin']-lookup[r['id']]['margin'])
 paired=[dict(condition=c,random_seed=s,group=g,mean_margin_change=float(np.mean(v))) for (c,s,g),v in groups.items()]
 conditions=defaultdict(list)
 for r in changed:conditions[(r['condition'],r['random_seed'])].append(r)
 summaries=[dict(condition=c,random_seed=s,**metrics(v),accuracy_change=metrics(v)['accuracy']-metrics(bases)['accuracy'],mean_removed_norm=float(np.mean([r['removed_norm'] for r in v]))) for (c,s),v in conditions.items()]
 seed_averages=[]
 for condition in ('rank_random','energy_random'):
  bygroup=defaultdict(list)
  for r in paired:
   if r['condition']==condition:bygroup[r['group']].append(r['mean_margin_change'])
  seed_averages.extend(dict(condition=condition,group=g,mean_margin_change_across_seeds=float(np.mean(v))) for g,v in bygroup.items())
 return {'status':'test_completed','stage':stage,'selected_k':k,'untouched':metrics(bases),'paired_group_changes':paired,'condition_metrics':summaries,'random_seed_averages':seed_averages,'test_ids':[r['id'] for r in rows],'interpretation':'Descriptive group-level causal sensitivity, not proof of exclusive necessity'},None

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['verify-hook','train','select','test']);parser.add_argument('--model-path',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--upstream',type=Path);parser.add_argument('--upstream-digest');parser.add_argument('--budget-seconds',type=float)
 args=parser.parse_args();source_digest=verify_source();data=dataset.load();upstream=None;fitdata=None;k=0
 if args.stage in ('select','test'):
  if args.upstream is None or args.upstream_digest is None:parser.error('select/test require --upstream and exact --upstream-digest')
  upstream=verify_archive(args.upstream,args.upstream_digest)
  expected='train_passed' if args.stage=='select' else 'selection_frozen'
  if upstream['status']!=expected or upstream['source_freeze_sha256']!=source_digest:raise ValueError('Invalid upstream stage or source')
  with np.load(args.upstream/'fit.npz',allow_pickle=False) as f:fitdata={key:f[key].copy() for key in f.files}
  if args.stage=='test':k=upstream['selected_k']
 elif args.upstream is not None or args.upstream_digest is not None:parser.error('Upstream is only valid for select/test')
 budget=args.budget_seconds if args.budget_seconds is not None else DEFAULT_BUDGET[args.stage]
 if not math.isfinite(budget) or budget<=0 or budget>DEFAULT_BUDGET[args.stage]:parser.error('Budget must be positive and no greater than frozen stage maximum')
 journal=Journal(args.out,args.stage,budget,k)
 try:
  split={'verify-hook':'train','train':'train','select':'validation','test':'test'}[args.stage];rows=[r for r in data['rows'] if r['split']==split]
  if args.stage=='verify-hook':rows=rows[:1]
  backend=Backend(args.model_path,journal)
  save(args.out/'metadata.json',dict(source_freeze_sha256=source_digest,data_freeze_sha256=dataset.sha(HERE/'DATA_FREEZE.json'),runtime=backend.versions,model=backend.manifest['model_id'],revision=backend.manifest['revision'],stage=args.stage,upstream_digest=args.upstream_digest))
  result,newfit=execute(args.stage,rows,backend,journal,fitdata,k)
  if newfit is not None or fitdata is not None:
   with (args.out/'fit.npz').open('xb') as f:np.savez_compressed(f,**(newfit if newfit is not None else fitdata))
  result.update(source_freeze_sha256=source_digest,forwards_completed=journal.count,heldout_test_evaluated=args.stage=='test')
  save(args.out/'result.json',result);seal(args.out);print(json.dumps(result,indent=2))
 except Exception as exc:
  save(args.out/'failure.json',dict(status='incomplete',stage=args.stage,error_type=type(exc).__name__,error=str(exc),forwards_started=journal.count,partial_rows_preserved=True));seal(args.out);raise
if __name__=='__main__':main()
