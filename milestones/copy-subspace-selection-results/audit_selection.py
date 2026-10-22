"""Read-only numerical/structural audit; never loads or executes the language model."""
from pathlib import Path
import argparse, collections, hashlib, json
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def stats(rows):
 groups=collections.defaultdict(list)
 for r in rows:groups[r['group']].append(r)
 assert len(rows)==20 and len(groups)==10 and all(len(g)==2 for g in groups.values())
 return dict(accuracy=sum(r['greedy_exact'] for r in rows)/20,mean_margin=sum(r['margin'] for r in rows)/20,positive_margin_groups=sum(all(r['margin']>0 for r in g) for g in groups.values()),greedy_exact_groups=sum(all(r['greedy_exact'] for r in g) for g in groups.values()),strict_wins=sum(r['margin']>0 for r in rows),ties=sum(r['margin']==0 for r in rows))
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,help='Optional fresh JSON output; default stdout only');args=parser.parse_args()
 run=HERE/'run';manifest=read(run/'artifact_manifest.json')
 freeze=ROOT/'milestones/copy-subspace/SOURCE_FREEZE_V2.json'
 assert sha(freeze)=='594a700c0fc1eb19b7e269a60531a3fd9fbfd6ca3a55d7dce3d90a431e01efd6'
 for name,digest in read(freeze)['sha256'].items():assert sha(ROOT/name)==digest
 for entry in manifest['files']:
  p=run/entry['path'];assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
 assert sha(run/'fit.npz')==sha(ROOT/'milestones/copy-subspace-train-results/run/fit.npz')
 result=read(run/'result.json');assert result['status']=='selection_frozen' and result['forwards_completed']==1300 and not result['heldout_test_evaluated']
 rows=[json.loads(line) for line in (run/'rows.jsonl').read_text().splitlines()];assert len(rows)==1300
 data=read(ROOT/'milestones/copy-subspace/dataset.json');expected={r['id']:r for r in data['rows'] if r['split']=='validation'}
 baseline=[r for r in rows if r['condition']=='untouched'];noop=[r for r in rows if r['condition']=='noop'];changed=[r for r in rows if r['condition'] not in ('untouched','noop')]
 assert len(baseline)==len(noop)==20 and len(changed)==1260
 assert {r['id'] for r in rows}==set(expected) and all(r['condition'] in ('untouched','noop') for r in rows[:40])
 assert all(r['condition'] not in ('untouched','noop') for r in rows[40:])
 assert len({r['id'] for r in baseline})==20 and len({r['id'] for r in noop})==20
 assert max(r['max_logit_error'] for r in noop)==max(r['max_state_error'] for r in noop)==0
 states={}
 for r in baseline:
  p=run/r['array']['path'];assert sha(p)==r['array']['sha256']
  with np.load(p,allow_pickle=False) as f:logits=f['logits'].astype(np.float64);h=f['state'].astype(np.float64)
  d=expected[r['id']];a,b=d['correct_token_id'],d['incorrect_token_id'];best=int(np.argmax(logits))
  assert logits[a]-logits[b]==r['margin'] and best==r['greedy_token_id'] and (best==a)==r['greedy_exact']
  probability=float(np.exp(logits[a]-(logits.max()+np.log(np.exp(logits-logits.max()).sum()))))
  assert abs(probability-r['target_probability'])<1e-12;states[r['id']]=h
 with np.load(run/'fit.npz',allow_pickle=False) as f:fit={key:f[key] for key in f.files}
 keys=set();conditions=collections.defaultdict(list);request_error=realized_error=0.
 for r in changed:
  k=r['k'];seed=r['random_seed'];name=r['condition'];key=(r['id'],k,name,seed);assert key not in keys;keys.add(key)
  assert k in (1,2,4) and name in ('pca','rank_random','energy_random') and (seed is None if name=='pca' else seed in range(20261022,20261032))
  h=states[r['id']];q=fit['pcs'][:,:k];center=h-fit['mean'];pca=q@(q.T@center)
  if name=='pca':delta=pca
  else:
   basis=fit['random_'+str(k)][seed-20261022];delta=basis@(basis.T@center)
   if name=='energy_random':delta=delta*(np.linalg.norm(pca)/np.linalg.norm(delta))
  requested=float(np.linalg.norm(delta));after=h.astype(np.float32)-delta.astype(np.float32);realized=float(np.linalg.norm(h-after.astype(np.float64)))
  request_error=max(request_error,abs(requested-r['requested_removed_norm']));realized_error=max(realized_error,abs(realized-r['realized_removed_norm']))
  assert np.isfinite([r['margin'],r['target_probability'],r['requested_removed_norm'],r['realized_removed_norm']]).all()
  conditions[(k,name,seed)].append(r)
 assert request_error<1e-10 and realized_error<1e-10 and len(conditions)==63
 base=stats(baseline);candidates={k:stats(conditions[(k,'pca',None)]) for k in (1,2,4)}
 for key,value in base.items():assert value==result['untouched'][key]
 for k,values in candidates.items():
  for key,value in values.items():assert value==result['candidates'][str(k)][key]
 eligible=[k for k,v in candidates.items() if v['accuracy']+1e-12>=base['accuracy']-.05 and v['mean_margin']+1e-12>=.9*base['mean_margin']]
 selected=max([0]+eligible);assert selected==result['selected_k']
 output=dict(status='passed',archive_sha256=sha(run/'artifact_manifest.json'),rows=1300,baseline_arrays_independently_scored=20,intervention_rows=1260,all_exact_validation_ids=True,baseline_before_interventions=True,fit_bytes_unchanged=True,requested_norm_max_reconstruction_error=request_error,realized_norm_max_reconstruction_error=realized_error,selected_k=selected,eligible_k=eligible,selection_rule=dict(minimum_greedy_accuracy=base['accuracy']-.05,minimum_mean_margin=.9*base['mean_margin']),untouched=base,candidates=candidates,conditions=[dict(k=k,condition=name,random_seed=seed,**stats(v),mean_requested_norm=float(np.mean([r['requested_removed_norm'] for r in v])),mean_realized_norm=float(np.mean([r['realized_removed_norm'] for r in v]))) for (k,name,seed),v in conditions.items()],heldout_test_evaluated=False,limitations='Only untouched full logits are archived. Intervention scores are summarized from recorded scalar rows, not independently reconstructed from full intervention logits. Requested and realized perturbation norms are reconstructed independently from immutable baseline states and training fit. No model inference in audit.')
 if args.out is not None:
  with args.out.open('x') as f:json.dump(output,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(output,indent=2,sort_keys=True))
if __name__=='__main__':main()
