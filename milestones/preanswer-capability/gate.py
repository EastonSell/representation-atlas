"""Portable post-result packaging of repaired runner; not historical frozen bytes."""
import argparse,os,hashlib,json,platform,re,resource,time
from pathlib import Path
FAMILIES=[('color','red','blue'),('temperature','hot','cold'),('state','open','closed'),('size','large','small'),('moisture','wet','dry'),('shade','light','dark')]
def examples():
 rows=[]
 for fi,(prop,a,b) in enumerate(FAMILIES):
  for order in range(2):
   target=f'item {fi*4+order*2+1}';other=f'item {fi*4+order*2+2}'
   for value in (a,b):
    facts=[f'The {prop} of {target} is {value}.',f'The {prop} of {other} is {b}.']
    if order:facts.reverse()
    choices=(a,b) if not order else (b,a)
    if not order:prompt=' '.join(facts)+f' What is the {prop} of {target}? Answer with exactly one word: {choices[0]} or {choices[1]}.'
    else:prompt='Facts: '+' '.join(facts)+f' Using these facts, give the {prop} of {target}. Reply only {choices[0]} or {choices[1]}.'
    rows.append(dict(pair=f'dev-{fi}-{order}',family=prop,template=order,target=target,prompt=prompt,answer=value,candidates=[a,b]))
 return rows
def exact(text,answer,truncated):return not truncated and text.strip().rstrip('.').strip().lower()==answer

def complete_pair_count(rows):
 groups={}
 for row in rows:groups.setdefault(row['pair'],[]).append(row)
 for pair,members in groups.items():
  if len(members)!=2 or len({r['answer'] for r in members})!=2:
   raise ValueError(f'Incomplete or duplicate counterfactual pair: {pair}')
 return sum(all(r['constrained_correct'] for r in members) for members in groups.values())

def enforce_artifact_cap(root,limit=1024**3,additional=0):
 total=sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())
 if total+additional>limit:raise RuntimeError(f'Artifact cap exceeded: {total}+{additional}>{limit}')
 return total

def summarize(rows,tokenizer_valid):
 valid=[r for r in rows if 'constrained_correct' in r]
 both=complete_pair_count(valid)
 constrained=sum(r['constrained_correct'] for r in valid)
 free=sum(r['free_exact'] for r in valid)
 return dict(constrained_correct=constrained,free_exact=free,both_pair_correct=both,
  gate_pass=bool(tokenizer_valid and len(rows)==24 and len(valid)==24 and constrained>=20 and free>=15 and both>=6))

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--model-path",default=os.environ.get("SMOLLM_MODEL_PATH"))
 parser.add_argument("--output-dir",type=Path,default=Path("output"))
 args=parser.parse_args()
 if not args.model_path:parser.error("--model-path or SMOLLM_MODEL_PATH is required")
 model_path=Path(args.model_path)
 args.output_dir.mkdir(parents=True,exist_ok=True)
 import torch,transformers
 from transformers import AutoTokenizer,AutoModelForCausalLM
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(3107)
 started=time.perf_counter();root=args.output_dir;rows=examples()
 artifact_root=root
 enforce_artifact_cap(artifact_root)
 manifest=json.loads(Path(__file__).with_name('MODEL_MANIFEST.json').read_text())
 for f in manifest['files']:
  assert hashlib.sha256((model_path/f['name']).read_bytes()).hexdigest()==f['sha256'],f['name']
 tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32).eval()
 assert len(model.model.layers)==30
 sites=[9,19,29];captures={};handles=[]
 for site in sites:
  def hook(module,args,output,site=site):captures[site]=output[0][0,-1].detach().clone()
  handles.append(model.model.layers[site].register_forward_hook(hook))
 result=dict(model_manifest=manifest,versions=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__),threads=4,dtype='float32',architecture=dict(blocks=len(model.model.layers),hidden_size=model.config.hidden_size,one_based_sites=[10,20,30],python_indices=sites,site_semantics='block output before final RMSNorm'),rows=[],tokenizer_valid=True)
 with torch.inference_mode():
  for row in rows:
   enforce_artifact_cap(artifact_root)
   text=tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=False,add_generation_prompt=True)
   ids=tokenizer.encode(text,add_special_tokens=False); candidate_ids=[tokenizer.encode(x,add_special_tokens=False) for x in row['candidates']]
   valid=all(len(x)==1 for x in candidate_ids) and all(tokenizer.encode(text+x,add_special_tokens=False)==ids+y for x,y in zip(row['candidates'],candidate_ids))
   if not valid:
    result['tokenizer_valid']=False;result['rows'].append(dict(**row,tokenizer_valid=False,candidate_ids=candidate_ids));continue
   inputs=torch.tensor([ids]);out=model(inputs,use_cache=False)
   logits=[float(out.logits[0,-1,x[0]]) for x in candidate_ids];norms={str(k):float(v.norm()) for k,v in captures.items()}
   chosen=row['candidates'][max(range(2),key=lambda x:logits[x])]
   generated=model.generate(inputs,attention_mask=torch.ones_like(inputs),max_new_tokens=12,do_sample=False,pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id,use_cache=True)[0,len(ids):].tolist()
   truncated=tokenizer.eos_token_id not in generated;answer=tokenizer.decode(generated,skip_special_tokens=True)
   result['rows'].append(dict(**row,chat_prompt=text,prompt_token_ids=ids,candidate_ids=candidate_ids,logits=logits,constrained_answer=chosen,constrained_correct=chosen==row['answer'],generation_token_ids=generated,generation=answer,truncated=truncated,free_exact=exact(answer,row['answer'],truncated),preanswer_residual_norms=norms))
   assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <3*1024**3
   assert time.perf_counter()-started <3600
 for h in handles:h.remove()
 result.update(summarize(result['rows'],result['tokenizer_valid']))
 result['elapsed_seconds']=time.perf_counter()-started;result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 result['causal_endpoints']='NOT RUN: contingent on capability gate and separately frozen causal implementation'
 payload=json.dumps(result,indent=2)+'\n'
 enforce_artifact_cap(artifact_root,additional=len(payload.encode('utf-8')))
 (root/'results.json').write_text(payload)
 enforce_artifact_cap(artifact_root)
 print(json.dumps({k:v for k,v in result.items() if k not in ('rows','model_manifest')},indent=2))
if __name__=='__main__':main()
