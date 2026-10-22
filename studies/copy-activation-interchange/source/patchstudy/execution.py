"""Portable supervised phase launcher. No model imports in the supervisor process."""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
from .config import BUDGET,SETTINGS
from .storage import sha,verify,json_write,utc

def verify_acceptance(root):
 root=Path(root).resolve();result=json.loads((root/'SUPERVISOR-RESULT.json').read_text())
 if result.get('status')!='accepted' or result.get('returncode')!=0 or result.get('stop_reason') is not None:raise ValueError('Supervisor did not accept phase')
 manifest_path=root/'RESULTS-MANIFEST.json'
 if sha(manifest_path)!=result['manifest_sha256']:raise ValueError('Supervisor manifest binding mismatch')
 entries=json.loads(manifest_path.read_text())
 actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name not in ('RESULTS-MANIFEST.json','SUPERVISOR-RESULT.json')}
 if actual!=set(entries):raise ValueError('Supervisor manifest coverage differs')
 for name,digest in entries.items():
  p=(root/name).resolve()
  if root not in p.parents or not p.is_file() or sha(p)!=digest:raise ValueError('Supervisor evidence hash mismatch')
 if json.loads((root/'status.json').read_text()).get('status')!='complete':raise ValueError('Missing complete worker marker')
 return result

def limits(stage,baseline_wall=0):
 if stage not in ('baseline','patch') or not math.isfinite(baseline_wall) or baseline_wall<0:raise ValueError('Invalid stage/time')
 wall=min(BUDGET[stage+'_wall_seconds'],BUDGET['execution_wall_seconds']-baseline_wall)
 if wall<=0:raise ValueError('Cumulative execution wall budget exhausted')
 # Two-thread CPU envelope; production monitor subtracts its fixed 5-second margin.
 return dict(wall_seconds=wall,cpu_seconds=2*wall+5,rss_bytes=BUDGET['max_rss_bytes'],sampling_seconds=.1)

def validate_clearance(clearance,expected):
 if clearance.get('approved') is not True or any(clearance.get(k)!=v for k,v in expected.items()):raise ValueError('Separate exact stage/source/data clearance required')

def _vendor():
 root=Path(__file__).resolve().parents[1]/'vendor/supervision'
 expected={'supervise.py':'35c91a94456bb23339f971a20c898e0a51140904f720c73dfc4c808be8818429','evidence.py':'179877eae7a13c917f6de545e47b5ebeb090028de8f25cd45e954317c1f3463e'}
 for name,digest in expected.items():
  if sha(root/name)!=digest:raise ValueError('Reviewed supervisor dependency changed')
 sys.path.insert(0,str(root))
 import evidence,supervise
 if Path(evidence.__file__).resolve()!=root/'evidence.py' or Path(supervise.__file__).resolve()!=root/'supervise.py':raise ValueError('Unexpected supervisor module origin')
 return evidence,supervise

def reserve_phase(out):
 evidence,_=_vendor();return evidence.reserve(out)

def frozen_inputs(root,model):
 root=Path(root).resolve();model=Path(model).resolve()
 if Path(__file__).resolve().parents[1]!=root:raise ValueError('Executing module is not the reviewed source root')
 source=json.loads((root/'SOURCE_MANIFEST.json').read_text());verify(root,source)
 cfg=json.loads((root/'config.json').read_text())
 if cfg!=SETTINGS:raise ValueError('Scientific configuration differs')
 freeze=json.loads((root/'DATA_FREEZE.json').read_text())
 if freeze['source_sha256']!=sha(root/'SOURCE_MANIFEST.json') or freeze['rows_sha256']!=sha(root/'data/dataset.json'):raise ValueError('Frozen source/data binding mismatch')
 model_manifest=json.loads((root/'MODEL_MANIFEST.json').read_text());verify(model,model_manifest['files'])
 return dict(source_sha256=sha(root/'SOURCE_MANIFEST.json'),data_freeze_sha256=sha(root/'DATA_FREEZE.json'),config_sha256=sha(root/'config.json'),model_files=[dict(name=n,**v) for n,v in model_manifest['files'].items()])

def disk_size(paths):
 return sum(p.stat().st_size for root in paths for p in Path(root).rglob('*') if p.is_file())

def verify_upstream(metadata,baseline_metadata,baseline_manifest_sha256):
 if metadata.get('upstream_baseline_manifest_sha256')!=baseline_manifest_sha256:raise ValueError('Wrong upstream baseline manifest')
 for key in ('source_sha256','data_freeze_sha256','config_sha256','model_files'):
  if metadata.get(key)!=baseline_metadata.get(key):raise ValueError('Wrong upstream baseline binding: '+key)

def runtime_metadata():
 import platform,importlib.metadata
 cpu=platform.processor()
 try:
  for line in Path('/proc/cpuinfo').read_text().splitlines():
   if line.startswith('model name'):
    cpu=line.split(':',1)[1].strip();break
 except OSError:pass
 return dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),cpu=cpu,logical_cpu_count=os.cpu_count(),llama_source_sha256=sha(importlib.metadata.distribution('transformers').locate_file('transformers/models/llama/modeling_llama.py')),versions={k:importlib.metadata.version(k) for k in ('torch','transformers','numpy')})

def accepted_phase(root,baseline=None):
 root=Path(root);verify_acceptance(root)
 metadata=json.loads((root/'evidence/run_metadata.json').read_text())
 if metadata['stage']=='patch' and baseline is None:raise ValueError('Patch acceptance requires explicit baseline')
 worker_status=json.loads((root/'status.json').read_text())
 if worker_status.get('evidence_manifest_sha256')!=sha(root/'evidence/manifest.json'):raise ValueError('Worker evidence manifest mismatch')
 verify(root/'evidence',json.loads((root/'evidence/manifest.json').read_text()))
 summary=json.loads((root/'resource_summary.json').read_text())
 stage=metadata['stage'];prior=0
 if baseline is not None:
  baseline_meta=accepted_phase(baseline)
  verify_upstream(metadata,baseline_meta,sha(Path(baseline)/'evidence/manifest.json'))
  prior=json.loads((Path(baseline)/'resource_summary.json').read_text())['wall_seconds']
 wall=summary['wall_seconds']
 if not math.isfinite(wall) or wall<0 or wall>BUDGET[stage+'_wall_seconds'] or wall+prior>BUDGET['execution_wall_seconds']:raise ValueError('Phase/cumulative wall cap exceeded')
 if summary['final_rss_upper_bound_bytes']>BUDGET['max_rss_bytes']:raise ValueError('Final RSS bound exceeds cap')
 if disk_size([root]+([baseline] if baseline else []))>BUDGET['max_output_bytes']:raise ValueError('Aggregate disk cap exceeded')
 return metadata

def audit_phase(root,baseline=None,source_path=None,model_path=None):
 try:accepted_phase(root,baseline)
 except (ValueError,KeyError,OSError,TypeError) as error:
  return dict(schema_version=1,passed=False,execution_accepted=False,errors=[str(error)])
 from .audit import audit_archive
 result=audit_archive(Path(root)/'evidence',Path(baseline)/'evidence' if baseline else None,source_path,model_path)
 result['execution_accepted']=True
 return result

def disjoint_output(out,protected):
 out=Path(out).resolve()
 for item in protected:
  p=Path(item).resolve()
  if out==p or out in p.parents or p in out.parents:raise ValueError('Output overlaps frozen/model/prior evidence directory')

def launch(args):
 root=Path(args.root).resolve();model=Path(args.model_dir).resolve();out=Path(args.out).resolve()
 if args.baseline:args.baseline=str(Path(args.baseline).resolve())
 if args.stage=='baseline' and args.baseline:raise ValueError('Baseline cannot have an upstream baseline')
 disjoint_output(out,[root,model]+([args.baseline] if args.baseline else []))
 binding=frozen_inputs(root,model);expected=dict(stage=args.stage,source_sha256=binding['source_sha256'],data_freeze_sha256=binding['data_freeze_sha256'])
 baseline_wall=0
 if args.stage=='patch':
  if not args.baseline:raise ValueError('Patch requires explicit accepted baseline directory')
  base=Path(args.baseline).resolve();base_meta=accepted_phase(base)
  for k in ('source_sha256','data_freeze_sha256','config_sha256','model_files'):
   if base_meta[k]!=binding[k]:raise ValueError('Baseline freeze/model binding differs')
  if not json.loads((base/'evidence/gate.json').read_text())['passed']:raise ValueError('Baseline competence gate failed')
  expected['upstream_baseline_manifest_sha256']=sha(base/'evidence/manifest.json')
  baseline_wall=json.loads((base/'resource_summary.json').read_text())['wall_seconds']
 validate_clearance(json.loads(Path(args.clearance).read_text()),expected)
 cfg=limits(args.stage,baseline_wall);evidence,supervise=_vendor();evidence.reserve(out)
 env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root))
 command=[sys.executable,'-m','patchstudy.execution','worker',args.stage,'--root',str(root),'--model-dir',str(model),'--out',str(out),'--clearance',str(Path(args.clearance).resolve())]
 if args.baseline:command+=['--baseline',str(Path(args.baseline).resolve())]
 os.chdir(root)  # -m must resolve the verified package, not an unrelated current directory
 code=supervise.monitor(command,out,cfg,env=env)
 if code==0:
  accepted_phase(out,args.baseline)
  paths=[out]+([Path(args.baseline)] if args.baseline else [])
  if disk_size(paths)>BUDGET['max_output_bytes']:
   # Do not rewrite the supervisor's immutable acceptance evidence. The extra
   # phase acceptance layer below is required by downstream launch/audit.
   raise RuntimeError('Aggregate disk cap exceeded; phase rejected by integration')
 return code

def worker(args):
 started=time.monotonic();start_utc=utc();root=Path(args.root).resolve();out=Path(args.out).resolve()
 if json.loads((out/'status.json').read_text())!={'status':'running'} or (out/'evidence').exists():raise ValueError('Worker requires fresh supervisor reservation')
 binding=frozen_inputs(root,args.model_dir)
 expected=dict(stage=args.stage,source_sha256=binding['source_sha256'],data_freeze_sha256=binding['data_freeze_sha256'])
 if args.stage=='patch':
  accepted_phase(args.baseline);expected['upstream_baseline_manifest_sha256']=sha(Path(args.baseline)/'evidence/manifest.json')
 validate_clearance(json.loads(Path(args.clearance).read_text()),expected)
 # All imports/model loading occur after the external deadline starts.
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 from .model_adapter import Adapter,load_model
 from .runner import baseline_engine,patch_engine
 actual_runtime=runtime_metadata()
 if actual_runtime['llama_source_sha256']!=json.loads((root/'RUNTIME.json').read_text())['llama_source_sha256']:raise ValueError('Installed Llama implementation source differs from freeze')
 adapter=Adapter(load_model(args.model_dir))
 rows=json.loads((root/'data/dataset.json').read_text())['rows']
 metadata=dict(schema_version=1,stage=args.stage,**binding,model_repo='HuggingFaceTB/SmolLM2-135M-Instruct',revision=SETTINGS['model_revision'],start_utc=start_utc,runtime_versions=actual_runtime['versions'],hardware_description=actual_runtime,numerical_settings=dict(dtype='float32',device='cpu',batch_size=1,attention='eager',cache=False,intraop_threads=2,interop_threads=1),seed=20261022)
 if args.stage=='patch':metadata['upstream_baseline_manifest_sha256']=expected['upstream_baseline_manifest_sha256']
 prior_bytes=disk_size([Path(args.baseline)]) if args.baseline else 0
 from . import storage
 storage.PRIOR_OUTPUT_BYTES=prior_bytes
 storage.PHASE_OUTPUT_ROOT=out
 if args.stage=='baseline':baseline_engine(adapter,rows,out/'evidence',metadata,root)
 else:patch_engine(adapter,rows,out/'evidence',metadata,Path(args.baseline)/'evidence',root)
 # Manifest exists before completion marker. Supervisor publishes and binds its
 # own enclosing manifest after worker termination, not vice versa.
 paths=[out]+([Path(args.baseline)] if args.baseline else [])
 if disk_size(paths)>BUDGET['max_output_bytes']:raise RuntimeError('Aggregate disk cap exceeded')
 evidence,_=_vendor();evidence.atomic(out/'status.json',dict(status='complete',stage=args.stage,evidence_manifest_sha256=sha(out/'evidence/manifest.json'),worker_wall_seconds=time.monotonic()-started,end_utc=utc()))

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['launch','worker']);p.add_argument('stage',choices=['baseline','patch'])
 for name in ('root','model-dir','out','clearance'):p.add_argument('--'+name,required=True)
 p.add_argument('--baseline');a=p.parse_args()
 if a.mode=='worker':worker(a);return 0
 return launch(a)
if __name__=='__main__':sys.exit(main())
