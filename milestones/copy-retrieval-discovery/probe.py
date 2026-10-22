"""One-off discovery instrument, not a production or held-out evaluation runner."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from benchmark_runner import verify_model, digest, save_json
INPUT_SHA256='358809a833305cc00cb2365ea9b47d3b38fde1cd3eec0435d7f7f88ab1e712ac'


def load_inputs():
 path=HERE/'inputs.json'
 if digest(path)!=INPUT_SHA256:raise ValueError('Discovery inputs changed')
 return json.loads(path.read_text())


def summarize(data,records):
 if [r['id'] for r in records]!=[r['id'] for r in data['rows']]:raise ValueError('Record IDs/order mismatch')
 if any(not math.isfinite(r['margin']) or type(r['greedy_exact']) is not bool for r in records):raise ValueError('Invalid evidence')
 candidates=[]
 for task in data['priority']:
  groups=defaultdict(list); wins=ties=greedy=0
  for row,record in zip(data['rows'],records):
   if row['task']!=task:continue
   win=record['margin']>0;wins+=win;ties+=record['margin']==0;greedy+=record['greedy_exact']
   groups[row['group']].append(win)
  complete=sum(all(v) for v in groups.values()); t=data['thresholds']
  candidates.append(dict(task=task,strict_correct_rows=wins,total_rows=24,ties=ties,greedy_exact_tokens=greedy,complete_groups=complete,total_groups=12,passed=wins>=t['strict_rows'] and complete>=t['complete_groups'] and greedy>=t['greedy_exact_tokens']))
 return dict(status='discovery_only',selected_task=next((c['task'] for c in candidates if c['passed']),None),candidates=candidates,thresholds=data['thresholds'],validation_test_generated=False,heldout_evaluated=False,causal_interventions=False)


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--model-path',type=Path,required=True)
 parser.add_argument('--out',type=Path,required=True)
 args=parser.parse_args()
 args.out.mkdir(parents=True,exist_ok=False)
 data=load_inputs(); save_json(args.out/'inputs.json',data)
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
 import torch
 import transformers
 if torch.__version__!='2.5.1+cpu' or transformers.__version__!='4.46.3':raise ValueError('Untested runtime')
 torch.set_num_threads(2)
 manifest=verify_model(args.model_path)
 tokenizer=transformers.AutoTokenizer.from_pretrained(args.model_path,local_files_only=True,trust_remote_code=False,token=False)
 encoded=[];invalid=[]
 for row in data['rows']:
  prefix=tokenizer.encode(row['prompt'],add_special_tokens=False)
  pair=[tokenizer.encode(row['prompt']+' '+row[k],add_special_tokens=False) for k in ('correct','incorrect')]
  valid=bool(prefix) and all(ids[:-1]==prefix for ids in pair) and pair[0][-1]!=pair[1][-1]
  encoded.append(dict(id=row['id'],prefix_token_ids=prefix,correct_token_ids=pair[0][len(prefix):],incorrect_token_ids=pair[1][len(prefix):],valid=valid))
  if not valid:invalid.append(row['id'])
 save_json(args.out/'tokenization.json',dict(records=encoded,invalid_ids=invalid,passed=not invalid))
 if invalid:raise ValueError('Entire discovery rejected: tokenization invalid; no filtering')
 save_json(args.out/'metadata.json',dict(model_id=manifest['model_id'],revision=manifest['revision'],model_manifest_sha256=digest(ROOT/'milestones/smollm-activations/MODEL_MANIFEST.json'),input_sha256=INPUT_SHA256,probe_sha256=digest(HERE/'probe.py'),python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,threads=2,precision='float32',attention='eager',chat_template=False,max_new_tokens=1))
 model=transformers.AutoModelForCausalLM.from_pretrained(args.model_path,local_files_only=True,trust_remote_code=False,token=False,use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager')
 model.eval();records=[]
 with torch.inference_mode():
  for entry in encoded:
   ids=torch.tensor([entry['prefix_token_ids']],dtype=torch.long)
   logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).logits[0,-1]
   logp=logits.log_softmax(-1);correct=entry['correct_token_ids'][0];incorrect=entry['incorrect_token_ids'][0];greedy=int(logits.argmax())
   records.append(dict(id=entry['id'],correct_log_probability=float(logp[correct]),incorrect_log_probability=float(logp[incorrect]),margin=float(logp[correct]-logp[incorrect]),greedy_token_id=greedy,greedy_text=tokenizer.decode([greedy],skip_special_tokens=False),greedy_exact=greedy==correct))
 save_json(args.out/'records.json',records)
 result=summarize(data,records);save_json(args.out/'summary.json',result)
 save_json(args.out/'artifact_manifest.json',{'files':[dict(path=f.name,bytes=f.stat().st_size,sha256=digest(f)) for f in sorted(args.out.iterdir())]})
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
