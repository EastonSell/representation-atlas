import argparse
from artifact_io import check_fresh_output, write_new_json
import json,platform,time
from pathlib import Path
import numpy as np
from geometry import *
def run(out_path, config_path=None):
 target=check_fresh_output(out_path)
 config_path=Path(config_path) if config_path is not None else Path(__file__).resolve().with_name('config.json')
 cfg=json.loads(config_path.read_text()); rows=[]; start=time.perf_counter()
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
 write_new_json(target,out)
 for regime in cfg['regimes']:
  for k in cfg['remove_top_pcs']:
   r=[x for x in rows if x['regime']==regime and x['remove_top_pcs']==k]
   print(regime,k,'ranking',round(np.mean([x['transformed_ranking'] for x in r]),4))
def main():
 parser=argparse.ArgumentParser(description='Run the experiment into a new output file; existing files are refused.')
 parser.add_argument('--out',type=Path,required=True,help='Fresh output JSON path; never overwritten')
 args=parser.parse_args()
 run(args.out)
if __name__=='__main__':main()
