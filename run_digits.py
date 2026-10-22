"""Train a small MLP on local UCI digits and test learned activation geometry."""
import argparse
from artifact_io import check_fresh_output, write_new_json
import hashlib,json,platform,time,warnings
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from geometry import fit_transform,transform

def hidden(model,x):return np.maximum(0,x@model.coefs_[0]+model.intercepts_[0])
def centroid_predict(train,y,test):
 classes=np.unique(y);centers=np.array([train[y==c].mean(0) for c in classes])
 distances=((test[:,None,:]-centers[None,:,:])**2).sum(2)
 return classes[np.argmin(distances,axis=1)]
def run(out_path, config_path=None):
 target=check_fresh_output(out_path)
 config_path=Path(config_path) if config_path is not None else Path(__file__).resolve().with_name('digits_config.json')
 cfg=json.loads(config_path.read_text());data=load_digits();x=data.data/16.;y=data.target;rows=[];splits={};start=time.perf_counter()
 for seed in cfg['seeds']:
  tr,te=train_test_split(np.arange(len(y)),test_size=cfg['test_fraction'],stratify=y,random_state=seed)
  splits[str(seed)]={'train_indices':tr.tolist(),'test_indices':te.tolist()}
  model=MLPClassifier(hidden_layer_sizes=(cfg['hidden_units'],),solver='adam',max_iter=cfg['max_iter'],random_state=seed,alpha=cfg['alpha'],learning_rate_init=cfg['learning_rate'],tol=cfg['tolerance'],early_stopping=False)
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always',ConvergenceWarning);model.fit(x[tr],y[tr])
  a,b=hidden(model,x[tr]),hidden(model,x[te]);raw=centroid_predict(a,y[tr],b)
  for k in cfg['remove_top_pcs']:
   state=fit_transform(a,k);ta,tb=transform(a,state),transform(b,state);pred=centroid_predict(ta,y[tr],tb)
   rows.append(dict(seed=seed,remove_top_pcs=k,mlp_test_accuracy=float(model.score(x[te],y[te])),centroid_test_accuracy=float(np.mean(pred==y[te])),raw_centroid_accuracy=float(np.mean(raw==y[te])),prediction_agreement_with_raw=float(np.mean(pred==raw)),epochs=int(model.n_iter_),convergence_warning=any(issubclass(w.category,ConvergenceWarning) for w in caught)))
 out={'measurement_type':'real UCI digit data, CPU-trained MLP hidden activations','config':cfg,'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'dataset_sha256':hashlib.sha256(data.data.tobytes()+data.target.tobytes()).hexdigest(),'elapsed_seconds':time.perf_counter()-start,'splits':splits,'rows':rows}
 write_new_json(target,out)
 for k in cfg['remove_top_pcs']:
  r=[v for v in rows if v['remove_top_pcs']==k];print(k,round(np.mean([v['centroid_test_accuracy'] for v in r]),4))
 print('mlp',np.mean([v['mlp_test_accuracy'] for v in rows]),'warning',any(v['convergence_warning'] for v in rows))
def main():
 parser=argparse.ArgumentParser(description='Run the experiment into a new output file; existing files are refused.')
 parser.add_argument('--out',type=Path,required=True,help='Fresh output JSON path; never overwritten')
 args=parser.parse_args()
 run(args.out)
if __name__=='__main__':main()
