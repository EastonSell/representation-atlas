import json,platform,time
from pathlib import Path
import numpy as np
from geometry import *
def run():
 cfg=json.loads(Path('config.json').read_text()); rows=[]; start=time.perf_counter()
 for seed in cfg['seeds']:
  for d in cfg['dimensions']:
   for regime in cfg['regimes']:
    rng=np.random.default_rng(seed+d);p,n=paired(rng,cfg['train_pairs'],d,regime);tp,tn=paired(rng,cfg['test_pairs'],d,regime)
    train=np.concatenate([p,n]); raw=direction(p,n)
    q=np.linalg.qr(rng.normal(size=(d,d)))[0]
    for remove in cfg['remove_top_pcs']:
     state=fit_transform(train,remove);zp,zn=transform(p,state),transform(n,state);ztp,ztn=transform(tp,state),transform(tn,state)
     axis=direction(zp,zn)
     rows.append(dict(seed=seed,dimension=d,regime=regime,remove_top_pcs=remove,raw_ranking=ranking(tp,tn,raw),transformed_ranking=ranking(ztp,ztn,axis),raw_pair_cosine=float(cosine(tp,tn).mean()),transformed_pair_cosine=float(cosine(ztp,ztn).mean()),mean_direction_pairing_error=float(np.max(np.abs(raw-direction(p,n[::-1])))),rotation_cka=cka(tp,tp@q),signal_axis_retention=float(np.linalg.norm(transform(np.eye(d)[:1]+state[0],state)))))
 out={'measurement_type':'synthetic CPU experiment, not model or brain activations','config':cfg,'python':platform.python_version(),'numpy':np.__version__,'elapsed_seconds':time.perf_counter()-start,'rows':rows}
 Path('results').mkdir(exist_ok=True);Path('results/experiment.json').write_text(json.dumps(out,indent=2)+'\n')
 for regime in cfg['regimes']:
  for k in cfg['remove_top_pcs']:
   r=[x for x in rows if x['regime']==regime and x['remove_top_pcs']==k]
   print(regime,k,'ranking',round(np.mean([x['transformed_ranking'] for x in r]),4))
if __name__=='__main__':run()
