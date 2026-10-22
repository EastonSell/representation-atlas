"""Tokenizer-only builder and strict loader for lexical-disjoint copying inputs."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
DISCOVERY_WORDS={'apple','bread','chair','cloud','dance','field','glass','horse','river','stone','train','water'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def prompt(word):return f'Repeat the word.\nWord: {word}\nRepeated word:'
def audit(data):
 rows=data['rows']
 if len(rows)!=80 or len({r['id'] for r in rows})!=80:raise ValueError('Expected 80 unique IDs')
 seen=set();allprompts=set();result={}
 for split,count in [('train',40),('validation',20),('test',20)]:
  part=[r for r in rows if r['split']==split];words={r['correct'] for r in part}
  if len(part)!=count or len(words)!=count or words&seen or words&DISCOVERY_WORDS:raise ValueError('Lexical split leakage or counts')
  seen|=words
  if {r['incorrect'] for r in part}!=words:raise ValueError('Candidate imbalance')
  for i in range(0,count,2):
   a,b=part[i:i+2]
   if a['group']!=b['group'] or (a['correct'],a['incorrect'])!=(b['incorrect'],b['correct']):raise ValueError('Counterfactual mismatch')
   if a['prompt']!=prompt(a['correct']) or b['prompt']!=prompt(b['correct']):raise ValueError('Template mismatch')
   if len(a['prefix_token_ids'])!=len(b['prefix_token_ids']):raise ValueError('Unequal lengths')
   changed=[j for j,(x,y) in enumerate(zip(a['prefix_token_ids'],b['prefix_token_ids'])) if x!=y]
   if changed!=[a['source_position']] or a['source_position']!=b['source_position']:raise ValueError('Source position mismatch')
  if allprompts&{r['prompt'] for r in part}:raise ValueError('Prompt overlap')
  allprompts|={r['prompt'] for r in part}
  result[split]={'rows':count,'unique_words':len(words),'groups':count//2,'lexical_priority_wins':sum(r['correct']<r['incorrect'] for r in part),'answer_only_complete_groups':0}
 if len({len(r['prefix_token_ids']) for r in rows})!=1:raise ValueError('Nonuniform prompt length')
 return {'splits':result,'cross_split_word_overlap':0,'cross_split_prompt_overlap':0,'cross_split_unordered_pair_overlap':0,'discovery_word_overlap':0,'passed':True}
def load():
 freeze=json.loads((HERE/'DATA_FREEZE.json').read_text())
 for name,expected in freeze['sha256'].items():
  if sha(HERE/name)!=expected:raise ValueError('Frozen data mismatch: '+name)
 data=json.loads((HERE/'dataset.json').read_text());audit(data);return data
if __name__=='__main__':
 import argparse,datetime,os
 parser=argparse.ArgumentParser();parser.add_argument('--model-path',required=True);args=parser.parse_args()
 for name in ('dataset.json','tokenizer_audit.json','DATA_FREEZE.json'):
  if (HERE/name).exists():raise FileExistsError(name)
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True,trust_remote_code=False,token=False)
 words=json.loads((HERE/'words.json').read_text())['words']
 if len(words)!=80 or len(set(words))!=80 or set(words)&DISCOVERY_WORDS or any(not w.isalpha() or not w.islower() for w in words):raise ValueError('Invalid raw pool')
 words=sorted(words,key=lambda w:hashlib.sha256(('copy-subspace-v1|'+w).encode()).hexdigest())
 rows=[];invalid=[]
 for split,part in [('train',words[:40]),('validation',words[40:60]),('test',words[60:])]:
  for i in range(0,len(part),2):
   a,b=part[i:i+2];prefixes=[tokenizer.encode(prompt(w),add_special_tokens=False) for w in (a,b)]
   changed=[j for j,(x,y) in enumerate(zip(*prefixes)) if x!=y]
   for orientation,(correct,incorrect) in enumerate(((a,b),(b,a))):
    prefix=prefixes[orientation];full=[tokenizer.encode(prompt(correct)+' '+w,add_special_tokens=False) for w in (correct,incorrect)]
    valid=len(prefixes[0])==len(prefixes[1]) and len(changed)==1 and all(ids[:-1]==prefix for ids in full) and full[0][-1]!=full[1][-1]
    id=f'{split}-{i//2:02d}-{orientation}'
    if not valid:invalid.append(id)
    rows.append(dict(id=id,group=f'{split}-{i//2:02d}',split=split,prompt=prompt(correct),correct=correct,incorrect=incorrect,prefix_token_ids=prefix,correct_token_id=full[0][-1],incorrect_token_id=full[1][-1],source_position=changed[0] if len(changed)==1 else None))
 preflight={'invalid_ids':invalid,'passed':not invalid,'model_weights_loaded':False}
 (HERE/'tokenizer_audit.json').write_text(json.dumps(preflight,indent=2)+'\n')
 if invalid:raise ValueError('Pool rejected; preserve audit and version any replacement')
 data={'schema':1,'task':'copy-subspace-v1','rows':rows};preflight.update(audit(data))
 (HERE/'tokenizer_audit.json').write_text(json.dumps(preflight,indent=2)+'\n')
 (HERE/'dataset.json').write_text(json.dumps(data,indent=2)+'\n')
 (HERE/'DATA_FREEZE.json').write_text(json.dumps({'local_freeze_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'published_before_freeze':False,'model_inference_performed':False,'sha256':{name:sha(HERE/name) for name in ('words.json','dataset.json','tokenizer_audit.json')}},indent=2)+'\n')
 print(json.dumps(preflight,indent=2))
