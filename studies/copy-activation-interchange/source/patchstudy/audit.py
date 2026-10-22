"""Independent no-inference archive audit, deliberately does not import metrics."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

BLOCKS=(0,5,10,14,19,24,29)
CONDITIONS=('donor_word','donor_final','donor_prefix_negative','identity_word','identity_final','donor_all')

def _array(x):
    a=np.asarray(x)
    if a.dtype!=np.float32 or a.shape!=(49152,) or not np.isfinite(a).all(): raise ValueError('Invalid full logits')
    return a.astype('float64')

def independent_score(patch,baseline,donor,recipient_token_id,donor_token_id):
    p=_array(patch); a=_array(baseline); _array(donor); r=recipient_token_id; d=donor_token_id
    if r==d or not 0<=r<49152 or not 0<=d<49152: raise ValueError('Invalid target IDs')
    pshift=p-p.max(); ashift=a-a.max()
    logp=pshift-np.log(np.exp(pshift).sum()); loga=ashift-np.log(np.exp(ashift).sum())
    pp=np.exp(logp); pa=np.exp(loga); top=int(p.argmax()); old=int(a.argmax()); margin=float(p[d]-p[r])
    return dict(recipient_logit=float(p[r]),donor_logit=float(p[d]),recipient_probability=float(pp[r]),donor_probability=float(pp[d]),recipient_margin=-margin,donor_margin=margin,raw_transfer_delta=float((p[d]-p[r])-(a[d]-a[r])),argmax_token_id=top,argmax_logit=float(p[top]),greedy_recipient=top==r,greedy_donor=top==d,pairwise_donor_positive=margin>0,exact_pairwise_tie=margin==0,off_target=top not in {r,d},genuine_switch=old==r and top==d,off_target_move=old in {r,d} and top not in {r,d},kl_baseline_to_patch=float(np.dot(pa,loga-logp)))

def _read_json(p):
    def reject(x): raise ValueError('Nonfinite JSON value')
    return json.loads(p.read_text(),parse_constant=reject)

def _lines(p):
    return [_read_text(line) for line in p.read_text().splitlines() if line.strip()]

def _read_text(text):
    def reject(x): raise ValueError('Nonfinite JSON')
    return json.loads(text,parse_constant=reject)

def _sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def _safe(root,name):
    p=(root/name).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file(): raise ValueError('Missing or unsafe file: '+str(name))
    return p

def _load(root,rec):
    p=_safe(root,rec['logits_file'])
    if _sha(p)!=rec['logits_sha256']: raise ValueError('Logit hash mismatch: '+str(p))
    z=np.load(p,allow_pickle=False); _array(z)
    if rec.get('logits_dtype')!='float32' or rec.get('logits_shape')!=[49152]: raise ValueError('Logit declarations mismatch')
    return z

def audit_records(rows,baselines,patch_records):
    """Audit already-loaded vectors/scalars; never substitutes for archive bindings."""
    errors=[]; byid={r['row_id']:r for r in rows}; groups=defaultdict(list)
    if len(rows)!=64 or len(byid)!=64: errors.append('Expected 64 unique rows')
    for r in rows: groups[r['pair_id']].append(r)
    if len(groups)!=32: errors.append('Expected 32 lexical groups')
    if len({r.get('recipient_word',r['row_id']) for r in rows})!=len(rows): errors.append('Lexical groups overlap')
    for pair in groups.values():
        if len(pair)!=2 or {r['orientation'] for r in pair}!={0,1}: errors.append('Missing inverse orientation'); continue
        a,b=pair
        if a['donor_row_id']!=b['row_id'] or b['donor_row_id']!=a['row_id'] or a['recipient_token_id']!=b['donor_token_id'] or b['recipient_token_id']!=a['donor_token_id']: errors.append('Pair not exact inverse')
    expected={(r['row_id'],b,c) for r in rows for b in BLOCKS for c in CONDITIONS}; seen=set(); scores=[]; maxima={}; max_metric_error=0.
    for rec in patch_records:
        try:
            key=(rec['row_id'],rec['block'],rec['condition'])
            if key not in expected or key in seen: raise ValueError('Duplicate or unexpected cell')
            seen.add(key); r=byid[rec['row_id']]; a=baselines[r['row_id']]; b=baselines[r['donor_row_id']]; z=rec['logits']
            calc=independent_score(z,a,b,r['recipient_token_id'],r['donor_token_id'])
            for k,v in calc.items():
                if k not in rec: raise ValueError('Missing score '+k)
                if isinstance(v,bool):
                    if rec[k] is not v: raise ValueError('Boolean score mismatch '+k)
                else:
                    err=abs(float(rec[k])-v); max_metric_error=max(max_metric_error,err)
                    if not np.isfinite(err) or err>1e-10*max(1.,abs(v)): raise ValueError('Scalar score mismatch '+k)
            cond=rec['condition']; block=rec['block']; identity='none'
            if cond in ('identity_word','identity_final','donor_prefix_negative') or (block==29 and cond=='donor_word'): identity='recipient'
            if cond=='donor_all' or (block==29 and cond=='donor_final'): identity='donor'
            if rec.get('expected_identity')!=identity: raise ValueError('Incorrect expected identity')
            if identity!='none':
                error=float(np.max(np.abs(z.astype('float64')-(a if identity=='recipient' else b).astype('float64'))))
                maxima[f'{block}:{cond}']=max(error,maxima.get(f'{block}:{cond}',0.))
                if error>1e-6: raise ValueError('Full-vocabulary control identity failed')
                if rec.get('identity_max_abs_error') is None or abs(rec['identity_max_abs_error']-error)>1e-12: raise ValueError('Identity scalar mismatch')
            calc.update({name:rec.get(name) for name in ('residual_delta_norm','requested_residual_delta_norm','realized_residual_delta_norm','recipient_residual_norm','residual_norm_ratio')})
            calc.update(row_id=r['row_id'],pair_id=r['pair_id'],block=block,condition=cond); scores.append(calc)
        except (KeyError,ValueError,TypeError) as e: errors.append(str(e))
    missing=sorted(expected-seen)
    if missing: errors.append(f'Missing {len(missing)} patch vectors')
    greedy=[]; positive=[]; in_pair=[]; competent=[]; normalized=[]; denominators={}
    for rid,r in byid.items():
        try:
            a=_array(baselines[rid]); b=_array(baselines[r['donor_row_id']]); rt=r['recipient_token_id']; dt=r['donor_token_id']
            if a.argmax()==rt: greedy.append(rid)
            if a.argmax() in (rt,dt): in_pair.append(rid)
            if a[rt]>a[dt]: positive.append(rid)
            denominators[rid]=float((b[dt]-b[rt])-(a[dt]-a[rt]))
        except (KeyError,ValueError) as e: errors.append('Baseline: '+str(e))
    for pid,pair in groups.items():
        if len(pair)==2 and all(r['row_id'] in greedy and r['row_id'] in positive for r in pair):
            competent.append(pid)
            if all(denominators[r['row_id']]>=1 for r in pair): normalized.append(pid)
    return dict(schema_version=1,passed=not errors,errors=errors,missing_cells=[list(x) for x in missing],observed_patch_count=len(seen),expected_patch_count=2688,control_error_maxima=maxima,max_metric_error=max_metric_error,gate_reconstruction=dict(total_rows=len(rows),total_pairs=len(groups),greedy_correct_count=len(greedy),positive_margin_count=len(positive),jointly_competent_pair_count=len(competent),failed_row_ids=sorted(set(byid)-(set(greedy)&set(positive))),failed_pair_ids=sorted(set(groups)-set(competent))),normalized_eligible_pair_ids=sorted(normalized),normalized_denominators=denominators,baseline_greedy_row_ids=greedy,baseline_in_pair_row_ids=in_pair,reconstructed_scores=scores,sample_size_accounting={'rows':64,'lexical_groups':32,'resampling_unit':'unordered lexical pair'})

def _manifest(root):
    manifest=_read_json(root/'manifest.json'); entries=manifest.get('files',manifest)
    if isinstance(entries,list): entries={e.get('path',e.get('relative_name')):e for e in entries}
    if not isinstance(entries,dict): raise ValueError('Invalid manifest file mapping')
    listed=set()
    for name,desc in entries.items():
        if name=='schema_version': continue
        if name=='manifest.json': raise ValueError('Recursive manifest self hash')
        p=_safe(root,name); listed.add(name)
        if p.stat().st_size!=desc['bytes'] or _sha(p)!=desc['sha256']: raise ValueError('Manifest mismatch: '+name)
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name not in ('manifest.json','audit.json')}
    if actual!=listed: raise ValueError('Manifest coverage mismatch: '+str(sorted(actual^listed)))
    return manifest

def _attempts(root,stage):
    events=_lines(root/'attempts.jsonl'); current={}; planned=[]
    for e in events:
        idx=e['attempt_index']; status=e['status']
        if idx not in current:
            if status!='planned' or (planned and idx<=planned[-1]): raise ValueError('Invalid attempt planning sequence')
            planned.append(idx)
        else:
            if current[idx]['status']!='planned' or status not in ('completed','failed','uncertain'): raise ValueError('Invalid attempt transition')
            for k in ('stage','row_id','block','condition'):
                if e.get(k)!=current[idx].get(k): raise ValueError('Attempt identity changed')
        current[idx]=e
    cap=128 if stage=='baseline' else 2688
    low,high=(1,128) if stage=='baseline' else (129,2816)
    if len(current)>cap or any(not low<=i<=high for i in current): raise ValueError('Attempt budget exceeded')
    return current

def audit_archive(path,baseline_path=None,source_path=None,model_path=None):
    root=Path(path).resolve(); errors=[]; result={'schema_version':1,'passed':False,'errors':errors}; base=Path(baseline_path).resolve() if baseline_path else root
    try:
        _manifest(root)
        if base!=root: _manifest(base)
        metadata=_read_json(root/'run_metadata.json'); dataset=_read_json(root/'dataset.json'); rows=dataset['rows']; stage=metadata['stage']
        if stage not in ('baseline','patch'): raise ValueError('Unknown archive stage')
        for field in ('source_sha256','data_freeze_sha256','config_sha256','model_files'):
            if not metadata.get(field): raise ValueError('Missing provenance binding '+field)
        # The source/model files are external evidence: require explicit digest manifests,
        # never claim their current bytes were verified unless accessible paths are supplied.
        freeze=_read_json(root/'DATA_FREEZE.json')
        if _sha(root/'DATA_FREEZE.json')!=metadata['data_freeze_sha256']: raise ValueError('Data freeze hash mismatch')
        if freeze['source_sha256']!=metadata['source_sha256']: raise ValueError('Source freeze mismatch')
        if _sha(root/'dataset.json')!=freeze['rows_sha256']: raise ValueError('Dataset freeze mismatch')
        if _sha(root/'SOURCE_MANIFEST.json')!=metadata['source_sha256']: raise ValueError('Source manifest binding mismatch')
        external_verified=source_path is not None and model_path is not None
        if source_path is None or model_path is None:
            errors.append('Source and model byte verification requires explicit source_path and model_path')
        else:
            source_manifest=_read_json(root/'SOURCE_MANIFEST.json')
            for name,info in source_manifest.get('files',source_manifest).items():
                if name=='schema_version': continue
                file=_safe(Path(source_path),name)
                if _sha(file)!=info['sha256'] or file.stat().st_size!=info['bytes']: raise ValueError('External source hash mismatch')
            for info in metadata['model_files']:
                file=_safe(Path(model_path),info['name'])
                if _sha(file)!=info['sha256'] or file.stat().st_size!=info['bytes']: raise ValueError('External model hash mismatch')
        config_file=root/'config.json'
        if not config_file.is_file() or _sha(config_file)!=metadata['config_sha256']: raise ValueError('Config binding missing or mismatched')
        from .config import validate as validate_config
        validate_config(_read_json(config_file))
        if stage=='patch':
            if base==root: raise ValueError('Explicit baseline archive required')
            if metadata.get('upstream_baseline_manifest_sha256')!=_sha(base/'manifest.json'): raise ValueError('Baseline manifest binding mismatch')
            bm=_read_json(base/'run_metadata.json')
            for field in ('source_sha256','data_freeze_sha256','config_sha256','model_files'):
                if bm[field]!=metadata[field]: raise ValueError('Upstream provenance mismatch '+field)
            if _sha(base/'dataset.json')!=_sha(root/'dataset.json'): raise ValueError('Upstream dataset mismatch')
        attempts=_attempts(root,stage); base_attempts=_attempts(base,'baseline'); base_records=_lines(base/'outcomes.jsonl'); baselines={}; captures={}
        for rec in base_records:
            rid=rec['row_id']; cond=rec['condition']
            if cond=='untouched':
                if rid in baselines: raise ValueError('Duplicate baseline')
                baselines[rid]=_load(base,rec)
            elif cond=='capture_check':
                if rid in captures: raise ValueError('Duplicate capture')
                captures[rid]=rec
            else: raise ValueError('Unexpected baseline condition')
            if rec['attempt_index'] not in base_attempts or base_attempts[rec['attempt_index']]['status']!='completed': raise ValueError('Outcome without completed attempt')
        if len(baselines)!=64 or len(captures)!=64 or len(base_attempts)!=128: raise ValueError('Incomplete baseline coverage')
        if len({r['attempt_index'] for r in base_records})!=128: raise ValueError('Duplicate baseline outcome attempt')
        for rec in base_records:
            for key in ('row_id','pair_id','orientation','condition','block'):
                if rec.get(key)!=base_attempts[rec['attempt_index']].get(key): raise ValueError('Baseline attempt/outcome mismatch')
        for rec in base_records:
            r=next(row for row in rows if row['row_id']==rec['row_id'])
            calculated=independent_score(_load(base,rec),baselines[r['row_id']],baselines[r['donor_row_id']],r['recipient_token_id'],r['donor_token_id'])
            for name,value in calculated.items():
                if name not in rec or (isinstance(value,bool) and rec[name] is not value) or (not isinstance(value,bool) and not np.isclose(rec[name],value,rtol=1e-10,atol=1e-10)):
                    raise ValueError('Baseline scalar mismatch '+name)
        states={}; readonly=0
        for r in rows:
            rid=r['row_id']; rec=captures[rid]; z=_load(base,rec)
            if np.max(np.abs(z.astype('float64')-baselines[rid]))>1e-6: raise ValueError('Readonly full-vocabulary identity failed')
            readonly+=1; statefiles=rec['block_state_files']
            if set(statefiles)!=set(map(str,BLOCKS)): raise ValueError('State coverage mismatch')
            for block in BLOCKS:
                info=statefiles[str(block)]; p=_safe(base,info['file'])
                if _sha(p)!=info['sha256']: raise ValueError('State hash mismatch')
                a=np.load(p,allow_pickle=False)
                if a.dtype!=np.float32 or list(a.shape)!=[r['prefix_length'],576] or list(a.shape)!=info['shape'] or not np.isfinite(a).all(): raise ValueError('Invalid state')
                states[rid,block]=a
        for r in rows:
            for block in BLOCKS:
                a=states[r['row_id'],block][r['negative_position']].astype('float64'); b=states[r['donor_row_id'],block][r['negative_position']].astype('float64')
                if np.max(np.abs(a-b))>1e-6: raise ValueError('Causal prefix cache identity failed')
        records=[]
        if stage=='patch':
            for rec in _lines(root/'outcomes.jsonl'):
                if rec['attempt_index'] not in attempts or attempts[rec['attempt_index']]['status']!='completed': raise ValueError('Patch outcome without completed attempt')
                for key in ('row_id','pair_id','orientation','condition','block'):
                    if rec.get(key)!=attempts[rec['attempt_index']].get(key): raise ValueError('Patch attempt/outcome mismatch')
                rec=dict(rec); rec['logits']=_load(root,rec); records.append(rec)
            if len({r['attempt_index'] for r in records})!=len(records): raise ValueError('Repeated outcome attempt')
            if len(attempts)!=2688 or len(records)!=2688: raise ValueError('Incomplete patch attempt coverage')
        prior_errors=errors[:]
        result=audit_records(rows,baselines,records); errors=result['errors']; errors.extend(prior_errors)
        if stage=='baseline':
            errors[:]=[e for e in errors if not e.startswith('Missing ')]; result['missing_cells']=[]
        gate=result['gate_reconstruction']; gate['readonly_identity_pass_count']=readonly; gate['complete']=len(baselines)==64 and readonly==64
        gate['passed']=gate['complete'] and gate['greedy_correct_count']>=58 and gate['positive_margin_count']>=58 and gate['jointly_competent_pair_count']>=27
        stored_gate=_read_json(base/'gate.json')
        for k,v in gate.items():
            if stored_gate.get(k)!=v: errors.append('Gate mismatch '+k)
        if stage=='patch' and not gate['passed']: errors.append('Patching after failed gate')
        rowmap={r['row_id']:r for r in rows}
        for rec in records:
            r=rowmap[rec['row_id']]; block=rec['block']; cond=rec['condition']; original=states[r['row_id'],block]; donor=states[r['donor_row_id'],block]
            positions='all' if cond=='donor_all' else [r['negative_position'] if cond=='donor_prefix_negative' else r['final_position'] if cond.endswith('final') else r['source_position']]
            if rec.get('patched_positions')!=positions: errors.append('Patch position mismatch')
            select=slice(None) if positions=='all' else positions
            a=original[select].astype('float64'); b=(original if cond.startswith('identity') else donor)[select].astype('float64')
            delta=float(np.linalg.norm(b-a)); norm=float(np.linalg.norm(a)); ratio=delta/norm if norm else None
            for k,v in [('residual_delta_norm',delta),('requested_residual_delta_norm',delta),('recipient_residual_norm',norm),('residual_norm_ratio',ratio)]:
                actual=rec.get(k)
                if (v is None and actual is not None) or (v is not None and (actual is None or not np.isclose(actual,v,rtol=1e-10,atol=1e-10))): errors.append('Residual norm mismatch '+k)
            realized=rec.get('realized_residual_delta_norm')
            if realized is None or not np.isfinite(realized) or abs(realized-delta)>1e-6: errors.append('Realized residual norm mismatch')
            baseline_rec=next(x for x in base_records if x['row_id']==r['row_id'] and x['condition']=='untouched'); donor_rec=next(x for x in base_records if x['row_id']==r['donor_row_id'] and x['condition']=='untouched')
            if rec.get('baseline_logits_sha256')!=baseline_rec['logits_sha256'] or rec.get('donor_logits_sha256')!=donor_rec['logits_sha256']: errors.append('Patch vector provenance mismatch')
        result['attempt_accounting']={'stage_attempts':len(attempts),'baseline_attempts':len(base_attempts),'cumulative_attempts':len(base_attempts)+(len(attempts) if stage=='patch' else 0)}
        result['hash_checks']={'archive_manifest':True,'upstream_manifest':True,'data_freeze':True,'configuration':True,'external_source_model_bytes':external_verified}
        if stage=='patch': _audit_summary(root,result)
        result.pop('reconstructed_scores',None)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as e:
        errors.append(str(e))
    result['errors']=errors; result['passed']=not errors
    return result

def _audit_summary(root,result):
    """Independent pair aggregation, denominator, and bootstrap reconstruction."""
    summary=_read_json(root/'summary.json'); errors=result['errors']; scores=result['reconstructed_scores']
    if not summary.get('complete') or summary.get('observed_patch_count')!=2688 or summary.get('expected_cell_count')!=42: errors.append('Summary coverage mismatch')
    if summary.get('control_error_maxima')!=result['control_error_maxima']: errors.append('Summary control maxima mismatch')
    cells=summary.get('cells',[]); keys=[(c['block'],c['condition']) for c in cells]; expected=[(b,c) for b in BLOCKS for c in CONDITIONS]
    if keys!=expected: errors.append('Summary cell map mismatch'); return
    if summary.get('normalized_eligible_pair_ids')!=result['normalized_eligible_pair_ids'] or summary.get('normalized_denominators')!={str(k):v for k,v in result['normalized_denominators'].items()}: errors.append('Summary normalized baseline eligibility mismatch')
    pairids=sorted({s['pair_id'] for s in scores}); matrix=np.empty((32,42),dtype=np.float64)
    index=np.random.default_rng(20261022).integers(0,32,(10000,32))
    for i,cell in enumerate(cells):
        selected=[s for s in scores if (s['block'],s['condition'])==keys[i]]
        if len(selected)!=64: errors.append('Summary missing raw cells'); return
        for j,pid in enumerate(pairids):
            ds=[s for s in selected if s['pair_id']==pid]
            matrix[j,i]=sum(s['raw_transfer_delta'] for s in ds)/2
            for score in ds:
                score['normalized_transfer']=score['raw_transfer_delta']/result['normalized_denominators'][score['row_id']] if pid in result['normalized_eligible_pair_ids'] else None
            stored=[g for g in cell['pair_level_outcomes'] if g['pair_id']==pid]
            if len(stored)!=1 or not np.isclose(stored[0]['raw_transfer_delta'],matrix[j,i],atol=1e-12,rtol=1e-12): errors.append('Pair averaging mismatch')
            if len(stored)==1:
                group=stored[0]
                for field in ('raw_transfer_delta','normalized_transfer','donor_probability','recipient_probability','kl_baseline_to_patch','residual_delta_norm','requested_residual_delta_norm','realized_residual_delta_norm','recipient_residual_norm','residual_norm_ratio','greedy_donor','greedy_recipient','off_target','genuine_switch','off_target_move','pairwise_donor_positive','exact_pairwise_tie'):
                    vals=[v[field] for v in ds]; expected_value=sum(vals)/2 if len(vals)==2 and all(v is not None for v in vals) else None
                    if expected_value is None:
                        if group.get(field) is not None: errors.append('Ineligible pair statistic '+field)
                    elif not np.isclose(group.get(field,np.nan),expected_value,rtol=1e-10,atol=1e-10): errors.append('Pair statistic mismatch '+field)
                directions=group.get('directions',[])
                if len(directions)!=2: errors.append('Directional summary missing')
                for direction in directions:
                    matches=[v for v in ds if v['row_id']==direction.get('row_id')]
                    if len(matches)!=1: errors.append('Directional summary identity mismatch'); continue
                    for field,value in matches[0].items():
                        if field in ('block','condition'): continue
                        actual=direction.get(field)
                        if value is None:
                            if actual is not None: errors.append('Directional null mismatch '+field)
                        elif isinstance(value,(int,float,bool)):
                            if actual is None or not np.isclose(actual,value,rtol=1e-10,atol=1e-10): errors.append('Directional statistic mismatch '+field)
                        elif actual!=value: errors.append('Directional identity mismatch '+field)
        for key,value in [('mean_raw_transfer_delta',float(matrix[:,i].mean())),('median_raw_transfer_delta',float(np.median([s['raw_transfer_delta'] for s in selected])))]:
            if not np.isclose(cell.get(key,np.nan),value,rtol=1e-12,atol=1e-12): errors.append('Summary statistic mismatch '+key)
        for field,scorefield,denom in [('donor_replacement','greedy_donor',64),('genuine_switch','genuine_switch',result['gate_reconstruction']['greedy_correct_count']),('off_target','off_target',64),('off_target_move','off_target_move',len(result['baseline_in_pair_row_ids'])),('recipient_retention','greedy_recipient',64),('positive_donor_margin','pairwise_donor_positive',64),('exact_ties','exact_pairwise_tie',64)]:
            n=sum(s[scorefield] for s in selected); stored=cell.get(field,{})
            if stored.get('numerator')!=n or stored.get('denominator')!=denom or stored.get('rate')!=(n/denom if denom else None): errors.append('Summary denominator/rate mismatch '+field)
            nums=np.array([sum(v[scorefield] for v in selected if v['pair_id']==pid) for pid in pairids]); dens=np.full(32,2)
            if field in ('genuine_switch','off_target_move'):
                eligible=result['baseline_greedy_row_ids'] if field=='genuine_switch' else result['baseline_in_pair_row_ids']
                dens=np.array([sum(v['row_id'] in eligible for v in selected if v['pair_id']==pid) for pid in pairids])
            ns=nums[index].sum(1); dd=dens[index].sum(1); valid=dd>0; interval=np.quantile(ns[valid]/dd[valid],[.025,.975]) if valid.any() else None
            if interval is None:
                if stored.get('pointwise_95') is not None: errors.append('Undefined rate interval mismatch')
            elif not np.allclose(stored.get('pointwise_95',[]),interval,rtol=1e-10,atol=1e-10): errors.append('Rate bootstrap mismatch '+field)
        for field in ('donor_probability','recipient_probability','kl_baseline_to_patch','residual_delta_norm','requested_residual_delta_norm','realized_residual_delta_norm','recipient_residual_norm','residual_norm_ratio','normalized_transfer'):
            values=[]
            for pid in pairids:
                vals=[v[field] for v in selected if v['pair_id']==pid]
                values.append(sum(vals)/2 if len(vals)==2 and all(v is not None for v in vals) else np.nan)
            values=np.array(values); finite=np.isfinite(values); stored=cell.get('descriptive_metrics',{}).get(field,{})
            if not finite.any():
                if stored.get('mean') is not None or stored.get('pointwise_95') is not None: errors.append('Unavailable descriptive metric mismatch '+field)
                continue
            nums=np.where(finite,values,0.)[index].sum(1); dens=finite[index].sum(1); valid=dens>0
            interval=np.quantile(nums[valid]/dens[valid],[.025,.975])
            if not np.isclose(stored.get('mean',np.nan),values[finite].mean(),rtol=1e-10,atol=1e-10) or not np.allclose(stored.get('pointwise_95',[]),interval,rtol=1e-10,atol=1e-10): errors.append('Descriptive metric bootstrap mismatch '+field)
        if cell.get('raw_negative_count')!=sum(v['raw_transfer_delta']<0 for v in selected): errors.append('Raw negative count mismatch')
        for name,predicate in [('normalized_negative_count',lambda v:v<0),('normalized_overshoot_count',lambda v:v>1)]:
            if cell.get(name)!=sum(v['normalized_transfer'] is not None and predicate(v['normalized_transfer']) for v in selected): errors.append('Normalized count mismatch '+name)
    if summary.get('bootstrap_seed')!=20261022 or summary.get('bootstrap_replications')!=10000: errors.append('Bootstrap settings mismatch')
    indices=np.random.default_rng(20261022).integers(0,32,(10000,32)); means=matrix.mean(0); se=matrix.std(0,ddof=1)/np.sqrt(32)
    samples=np.empty((10000,42))
    for k in range(10000): samples[k]=matrix[indices[k]].mean(0)
    intervals=np.percentile(samples,[2.5,97.5],axis=0).T
    substantive=[i for i,(b,c) in enumerate(expected) if b!=29 and c in ('donor_word','donor_final')]; valid=[i for i in substantive if se[i]>0]
    critical=float(np.percentile(np.max(np.abs(samples[:,valid]-means[valid])/se[valid],axis=1),95)) if valid else 0.
    for i,cell in enumerate(cells):
        if not np.allclose(cell.get('raw_delta_pointwise_95',[]),intervals[i],rtol=1e-10,atol=1e-10): errors.append('Pointwise bootstrap mismatch')
        if cell.get('raw_delta_zero_variance')!=bool(se[i]==0): errors.append('Zero-variance label mismatch')
        if i in substantive and not np.allclose(cell.get('raw_delta_simultaneous_95',[]),[means[i]-critical*se[i],means[i]+critical*se[i]],rtol=1e-10,atol=1e-10): errors.append('Simultaneous bootstrap mismatch')
