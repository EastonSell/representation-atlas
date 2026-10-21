"""Offline pretrained activation extraction and held-out contrast geometry."""
import argparse,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from geometry import fit_transform,transform,direction,ranking
from task import examples

def main():
 import torch,transformers
 from transformers import AutoTokenizer,AutoModelForCausalLM
 parser=argparse.ArgumentParser();parser.add_argument('--model-path',required=True);parser.add_argument('--threads',type=int,default=2);parser.add_argument('--batch-size',type=int,default=8);args=parser.parse_args()
 if args.threads<1 or args.batch_size<1:raise ValueError('Positive threads/batch size required')
 torch.set_num_threads(args.threads);start=time.perf_counter();path=Path(args.model_path)
 provenance=json.loads((path/'MODEL_MANIFEST.json').read_text())
 tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
 if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
 model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32);model.eval()
 selected=[];encoded=[];prefix_lengths=[];excluded=[]
 for row in examples():
  prefix=tokenizer.encode(row['prompt'],add_special_tokens=False)
  pair=[tokenizer.encode(row['prompt']+' '+row[k],add_special_tokens=False) for k in ('correct','incorrect')]
  if len(pair[0])!=len(pair[1]) or any(ids[:len(prefix)]!=prefix for ids in pair):
   excluded.append(row['id']);continue
  selected.append(row);encoded.extend(pair);prefix_lengths.extend([len(prefix)]*2)
 states=[];ll=[]
 with torch.inference_mode():
  for start_idx in range(0,len(encoded),args.batch_size):
   batch=encoded[start_idx:start_idx+args.batch_size];lengths=[len(x) for x in batch];maxlen=max(lengths)
   ids=torch.full((len(batch),maxlen),tokenizer.pad_token_id,dtype=torch.long);mask=torch.zeros_like(ids)
   for i,tokens in enumerate(batch):ids[i,:len(tokens)]=torch.tensor(tokens);mask[i,:len(tokens)]=1
   out=model(input_ids=ids,attention_mask=mask,output_hidden_states=True,use_cache=False)
   logp=out.logits.log_softmax(-1)
   for i,length in enumerate(lengths):
    states.append(out.hidden_states[-1][i,length-1].cpu().numpy().copy())
    pl=prefix_lengths[start_idx+i];ll.append(float(sum(logp[i,j-1,ids[i,j]].item() for j in range(pl,length))))
  del out,logp
 states=np.stack(states);p,n=states[::2],states[1::2];likelihood=np.array(ll).reshape(-1,2);rows=[]
 for seed in [11,29,47]:
  for split_mode in ['random_pairs','balanced_operand_groups']:
   rng=np.random.default_rng(seed)
   if split_mode=='random_pairs':
    idx=rng.permutation(len(p));tr,te=idx[:60],idx[60:]
   else:
    train_a=set(rng.permutation(10)[:6].tolist())
    tr=np.array([i for i,row in enumerate(selected) if int(row['id'].split('-')[0]) in train_a])
    te=np.array([i for i,row in enumerate(selected) if int(row['id'].split('-')[0]) not in train_a])
   if not len(te):raise ValueError('Insufficient tokenizer-matched pairs')
   for k in [0,1,2]:
    state=fit_transform(np.concatenate([p[tr],n[tr]]),k);tp,tn=transform(p,state),transform(n,state)
    axis=direction(tp[tr],tn[tr]);null=[]
    for _ in range(20):
     signs=rng.choice([-1.,1.],size=len(tr));random_axis=((tp[tr]-tn[tr])*signs[:,None]).mean(0);null.append(ranking(tp[te],tn[te],random_axis))
    rows.append({'seed':seed,'split_mode':split_mode,'remove_top_pcs':k,'train_pair_ids':[selected[i]['id'] for i in tr],'test_pair_ids':[selected[i]['id'] for i in te],'pair_ranking':ranking(tp[te],tn[te],axis),'shuffled_label_rankings':null,'likelihood_correct_preference':float(np.mean(likelihood[te,0]>likelihood[te,1]))})
 result={'measurement_type':'real pretrained SmolLM2 final-layer activations on generated verifiable arithmetic pairs','model_manifest':provenance,'versions':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'transformers':transformers.__version__},'threads':args.threads,'batch_size':args.batch_size,'device':'cpu','dtype':'float32','included_pairs':len(selected),'excluded_pair_ids':excluded,'hidden_dimension':int(states.shape[1]),'activation_sha256':hashlib.sha256(states.tobytes()).hexdigest(),'elapsed_seconds':time.perf_counter()-start,'rows':rows}
 dest=Path(__file__).parent/'results';dest.mkdir(exist_ok=True)
 (dest/'experiment.json').write_text(json.dumps(result,indent=2)+'\n');(dest/'pairs.json').write_text(json.dumps(selected,indent=2)+'\n')
 for mode in ['random_pairs','balanced_operand_groups']:
  for k in [0,1,2]:
   vals=[r for r in rows if r['remove_top_pcs']==k and r['split_mode']==mode];print(mode,k,'pair ranking',np.mean([r['pair_ranking'] for r in vals]),'null',np.mean([v for r in vals for v in r['shuffled_label_rankings']]))
 print('likelihood preference',np.mean([r['likelihood_correct_preference'] for r in rows]),'pairs',len(selected),'seconds',result['elapsed_seconds'])
if __name__=='__main__':main()
