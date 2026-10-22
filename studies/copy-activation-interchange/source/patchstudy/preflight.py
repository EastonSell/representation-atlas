"""Tokenizer-only audit. Never imports AutoModel or performs a pretrained forward."""
import argparse,json,hashlib,platform
from pathlib import Path
from .dataset import select,InsufficientCandidates,PROMPT
from .storage import json_write,sha,utc,manifest
from .config import SETTINGS,BUDGET

def run(model_dir,root):
 from transformers import AutoTokenizer
 root=Path(root);model_dir=Path(model_dir)
 tok=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False)
 candidates=json.loads((root/'provenance/candidate_words.json').read_text());excluded=json.loads((root/'provenance/historical_exclusions.json').read_text())
 try:r=select(tok,candidates['words'],excluded['words'])
 except InsufficientCandidates as e:
  json_write(root/'data/rejection_ledger.json',e.ledger);raise
 json_write(root/'data/rejection_ledger.json',{k:v for k,v in r.items() if k!='rows'})
 json_write(root/'data/dataset.json',dict(schema_version=1,task_id='copy-patch-v1',prompt_template=PROMPT,rows=r['rows']))
 names=['tokenizer.json','tokenizer_config.json','special_tokens_map.json','merges.txt','vocab.json']
 receipt=dict(schema_version=1,candidate_count=r['candidate_count'],eligible_count=r['eligible_count'],selected_count=len(r['rows']),rejected_count=len(r['rejected']),prefix_lengths=sorted(set(x['prefix_length'] for x in r['rows'])),model_inference_performed=False,tokenizer_files=[dict(relative_name=n,sha256=sha(model_dir/n),bytes=(model_dir/n).stat().st_size) for n in names],rows_sha256=sha(root/'data/dataset.json'),candidates_sha256=sha(root/'provenance/candidate_words.json'),exclusions_sha256=sha(root/'provenance/historical_exclusions.json'),selection_policy=r['selection_policy'],local_freeze_utc=utc(),published_before_freeze=False)
 json_write(root/'data/TOKENIZER_AUDIT.json',receipt)
 print(json.dumps({k:v for k,v in receipt.items() if k!='tokenizer_files'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--root',required=True);a=p.parse_args();run(a.model_dir,a.root)
