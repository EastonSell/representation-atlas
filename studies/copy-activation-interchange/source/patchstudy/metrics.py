"""Frozen-panel descriptive scores. No model imports or inference."""
from collections import defaultdict
import numpy as np

BLOCKS=(0,5,10,14,19,24,29)
CONDITIONS=('donor_word','donor_final','donor_prefix_negative','identity_word','identity_final','donor_all')
CELLS=tuple((b,c) for b in BLOCKS for c in CONDITIONS)
SUBSTANTIVE=tuple((b,c) for b in BLOCKS[:-1] for c in ('donor_word','donor_final'))
SEED=20261022
REPLICATIONS=10000

def _vector(z):
    z=np.asarray(z)
    if z.shape!=(49152,) or z.dtype!=np.float32 or not np.isfinite(z).all():
        raise ValueError('Expected finite float32 logits with shape (49152,)')
    return z.astype(np.float64)

def _distribution(z):
    logp=z-(np.max(z)+np.log(np.exp(z-np.max(z)).sum(dtype=np.float64)))
    return logp,np.exp(logp)

def score_logits(patch, baseline, donor, recipient_token_id, donor_token_id):
    p,a,b=map(_vector,(patch,baseline,donor))
    r,d=recipient_token_id,donor_token_id
    if not isinstance(r,(int,np.integer)) or not isinstance(d,(int,np.integer)) or not (0<=r<49152 and 0<=d<49152) or r==d:
        raise ValueError('Distinct vocabulary token IDs required')
    lp,pp=_distribution(p); la,pa=_distribution(a)
    top=int(np.argmax(p)); old=int(np.argmax(a))
    margin=float(p[d]-p[r])
    return dict(recipient_logit=float(p[r]),donor_logit=float(p[d]),recipient_probability=float(pp[r]),donor_probability=float(pp[d]),recipient_margin=-margin,donor_margin=margin,raw_transfer_delta=float(margin-(a[d]-a[r])),argmax_token_id=top,argmax_logit=float(p[top]),greedy_recipient=top==r,greedy_donor=top==d,pairwise_donor_positive=margin>0,exact_pairwise_tie=margin==0,off_target=top not in (r,d),genuine_switch=old==r and top==d,off_target_move=old in (r,d) and top not in (r,d),kl_baseline_to_patch=float(np.sum(pa*(la-lp),dtype=np.float64)))

def normalized_eligibility(rows,baselines):
    groups=defaultdict(list); denominators={}; excluded=[]; eligible=[]
    for row in rows: groups[row['pair_id']].append(row)
    for pid, pair in sorted(groups.items()):
        if len(pair)!=2 or {r['orientation'] for r in pair}!={0,1}: raise ValueError('Each lexical pair requires both directions')
        reasons=[]
        left,right=pair
        if left['donor_row_id']!=right['row_id'] or right['donor_row_id']!=left['row_id'] or left['recipient_token_id']!=right['donor_token_id'] or right['recipient_token_id']!=left['donor_token_id']:
            raise ValueError('Pair orientations must be exact inverses')
        for row in pair:
            a=_vector(baselines[row['row_id']]); b=_vector(baselines[row['donor_row_id']]); r=row['recipient_token_id']; d=row['donor_token_id']
            denom=float((b[d]-b[r])-(a[d]-a[r])); denominators[row['row_id']]=denom
            if np.argmax(a)!=r or not a[r]>a[d]: reasons.append('baseline_pair_not_jointly_competent')
            if denom<1: reasons.append('baseline_denominator_below_one')
        if reasons: excluded.append({'pair_id':pid,'reasons':sorted(set(reasons))})
        else: eligible.append(pid)
    return dict(eligible_pair_ids=eligible,denominators=denominators,exclusions=excluded,excluded_pair_count=len(excluded))

def bootstrap_groups(values, simultaneous_indices=None):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=2 or x.shape[0]!=32 or not np.isfinite(x).all(): raise ValueError('Exactly 32 finite pair groups required')
    rng=np.random.default_rng(SEED); indices=rng.integers(0,32,size=(REPLICATIONS,32))
    # Same indices jointly retain every direction (already averaged) and cell.
    boot=np.empty((REPLICATIONS,x.shape[1]),dtype=np.float64)
    for start in range(0,REPLICATIONS,100): boot[start:start+100]=x[indices[start:start+100]].mean(axis=1)
    means=x.mean(axis=0); se=x.std(axis=0,ddof=1)/np.sqrt(32); zero=se==0
    result=dict(group_count=32,seed=SEED,replications=REPLICATIONS,mean=means.tolist(),standard_error=se.tolist(),zero_variance=zero.tolist(),pointwise_95=np.quantile(boot,[.025,.975],axis=0).T.tolist(),interval_label='descriptive pointwise percentile 95%')
    if simultaneous_indices is not None:
        chosen=np.asarray(simultaneous_indices,dtype=int); valid=chosen[~zero[chosen]]
        critical=float(np.quantile(np.max(np.abs(boot[:,valid]-means[valid])/se[valid],axis=1),.95)) if len(valid) else 0.
        result.update(simultaneous_critical_value=critical,simultaneous_95={str(int(i)):[float(means[i]-critical*se[i]),float(means[i]+critical*se[i])] for i in chosen},simultaneous_label='approximate descriptive 12-cell max-standardized bootstrap band')
    return result

def _ratio(n,d): return {'numerator':int(n),'denominator':int(d),'rate':float(n/d) if d else None,'null_reason':None if d else 'zero denominator'}

def summarize(rows,baselines,patch_records):
    if len(rows)!=64 or len({r['row_id'] for r in rows})!=64 or len({r['pair_id'] for r in rows})!=32: raise ValueError('Frozen panel must contain 64 rows in 32 disjoint lexical groups')
    if len({r.get('recipient_word',r['row_id']) for r in rows})!=64: raise ValueError('Lexical pair groups must be disjoint')
    lookup={r['row_id']:r for r in rows}; pairids=sorted({r['pair_id'] for r in rows}); elig=normalized_eligibility(rows,baselines)
    expected={(r['row_id'],b,c) for r in rows for b,c in CELLS}; seen={}; bycell=defaultdict(list); control_maxima={}
    for rec in patch_records:
        key=(rec['row_id'],rec['block'],rec['condition'])
        if key not in expected or key in seen: raise ValueError('Duplicate or unexpected patch cell')
        row=lookup[rec['row_id']]; score=score_logits(rec['logits'],baselines[row['row_id']],baselines[row['donor_row_id']],row['recipient_token_id'],row['donor_token_id'])
        score.update(row_id=row['row_id'],pair_id=row['pair_id'])
        block,condition=rec['block'],rec['condition']; reference=None
        if condition in ('identity_word','identity_final','donor_prefix_negative') or (block==29 and condition=='donor_word'): reference=baselines[row['row_id']]
        if condition=='donor_all' or (block==29 and condition=='donor_final'): reference=baselines[row['donor_row_id']]
        if reference is not None:
            error=float(np.max(np.abs(rec['logits'].astype('float64')-reference.astype('float64'))))
            label=f'{block}:{condition}'; control_maxima[label]=max(error,control_maxima.get(label,0.))
        for name in ('residual_delta_norm','requested_residual_delta_norm','realized_residual_delta_norm','recipient_residual_norm','residual_norm_ratio'):
            score[name]=rec.get(name)
            if score[name] is not None and (not np.isfinite(score[name]) or score[name]<0): raise ValueError('Invalid residual norm scalar')
        score['normalized_transfer']=score['raw_transfer_delta']/elig['denominators'][row['row_id']] if row['pair_id'] in elig['eligible_pair_ids'] else None
        seen[key]=score; bycell[(rec['block'],rec['condition'])].append(score)
    missing=sorted(expected-set(seen)); cells=[]; raw_group_matrix=[]
    baseline_correct=sum(int(np.argmax(baselines[r['row_id']]))==r['recipient_token_id'] for r in rows)
    baseline_in_pair=sum(int(np.argmax(baselines[r['row_id']])) in (r['recipient_token_id'],r['donor_token_id']) for r in rows)
    for block,condition in CELLS:
        scores=bycell[(block,condition)]; complete=len(scores)==64; groups=[]
        for pid in pairids:
            ds=[s for s in scores if s['pair_id']==pid]
            group={'pair_id':pid,'direction_count':len(ds),'directions':ds}
            for field in ('raw_transfer_delta','normalized_transfer','donor_probability','recipient_probability','kl_baseline_to_patch','residual_delta_norm','requested_residual_delta_norm','realized_residual_delta_norm','recipient_residual_norm','residual_norm_ratio','greedy_donor','greedy_recipient','off_target','genuine_switch','off_target_move','pairwise_donor_positive','exact_pairwise_tie'):
                vals=[s[field] for s in ds]
                group[field]=float(np.mean(vals)) if len(vals)==2 and all(v is not None for v in vals) else None
            groups.append(group)
        cell=dict(block=block,condition=condition,complete=complete,observed_count=len(scores),expected_count=64,pair_level_outcomes=groups)
        if complete:
            deltas=[s['raw_transfer_delta'] for s in scores]
            cell.update(mean_raw_transfer_delta=float(np.mean(deltas)),median_raw_transfer_delta=float(np.median(deltas)),donor_replacement=_ratio(sum(s['greedy_donor'] for s in scores),64),genuine_switch=_ratio(sum(s['genuine_switch'] for s in scores),baseline_correct),off_target=_ratio(sum(s['off_target'] for s in scores),64),off_target_move=_ratio(sum(s['off_target_move'] for s in scores),baseline_in_pair),off_target_move_count_over_64=_ratio(sum(s['off_target_move'] for s in scores),64),recipient_retention=_ratio(sum(s['greedy_recipient'] for s in scores),64),positive_donor_margin=_ratio(sum(s['pairwise_donor_positive'] for s in scores),64),exact_ties=_ratio(sum(s['exact_pairwise_tie'] for s in scores),64),raw_negative_count=sum(d<0 for d in deltas),normalized_negative_count=sum(s['normalized_transfer'] is not None and s['normalized_transfer']<0 for s in scores),normalized_overshoot_count=sum(s['normalized_transfer'] is not None and s['normalized_transfer']>1 for s in scores))
        if complete:
            index=np.random.default_rng(SEED).integers(0,32,(REPLICATIONS,32))
            descriptive={}
            for field in ('donor_probability','recipient_probability','kl_baseline_to_patch','residual_delta_norm','requested_residual_delta_norm','realized_residual_delta_norm','recipient_residual_norm','residual_norm_ratio','normalized_transfer'):
                values=np.asarray([np.nan if g[field] is None else g[field] for g in groups])
                available=np.isfinite(values)
                if not available.any():
                    descriptive[field]={'mean':None,'pointwise_95':None,'null_reason':'no eligible finite pair values','pair_count':0}; continue
                totals=np.where(available,values,0.)[index].sum(axis=1); counts=available[index].sum(axis=1)
                sample=totals[counts>0]/counts[counts>0]
                descriptive[field]={'mean':float(values[available].mean()),'pointwise_95':np.quantile(sample,[.025,.975]).tolist(),'pair_count':int(available.sum()),'undefined_bootstrap_count':int((counts==0).sum())}
            pair_rows={pid:[r for r in rows if r['pair_id']==pid] for pid in pairids}
            for field,scorefield in [('donor_replacement','greedy_donor'),('genuine_switch','genuine_switch'),('off_target','off_target'),('off_target_move','off_target_move'),('recipient_retention','greedy_recipient'),('positive_donor_margin','pairwise_donor_positive'),('exact_ties','exact_pairwise_tie')]:
                numerator=np.array([g[scorefield]*2 for g in groups]); denominator=np.full(32,2.)
                if field in ('genuine_switch','off_target_move'):
                    denominator=np.array([sum(int(np.argmax(baselines[r['row_id']]))==r['recipient_token_id'] if field=='genuine_switch' else int(np.argmax(baselines[r['row_id']])) in (r['recipient_token_id'],r['donor_token_id']) for r in pair_rows[pid]) for pid in pairids])
                ns=numerator[index].sum(1); ds=denominator[index].sum(1); valid=ds>0
                cell[field]['pointwise_95']=np.quantile(ns[valid]/ds[valid],[.025,.975]).tolist() if valid.any() else None
                cell[field]['undefined_bootstrap_count']=int((~valid).sum())
            cell['descriptive_metrics']=descriptive
        cells.append(cell); raw_group_matrix.append([g['raw_transfer_delta'] for g in groups])
    uncertainty=None
    if not missing:
        uncertainty=bootstrap_groups(np.asarray(raw_group_matrix).T,[CELLS.index(c) for c in SUBSTANTIVE])
        for i,c in enumerate(cells):
            c['raw_delta_pointwise_95']=uncertainty['pointwise_95'][i]; c['raw_delta_zero_variance']=uncertainty['zero_variance'][i]
            c['raw_delta_simultaneous_95']=uncertainty['simultaneous_95'].get(str(i))
    return dict(schema_version=1,complete=not missing,partial=bool(missing),expected_cell_count=42,observed_cell_count=sum(c['complete'] for c in cells),expected_patch_count=2688,observed_patch_count=len(seen),cells=cells,control_error_maxima=control_maxima,missing_cells=[dict(row_id=r,block=b,condition=c) for r,b,c in missing],normalized_eligible_pair_ids=elig['eligible_pair_ids'],normalized_denominators=elig['denominators'],normalized_exclusions=elig['exclusions'],normalized_excluded_pair_count=elig['excluded_pair_count'],bootstrap=uncertainty,bootstrap_seed=SEED,bootstrap_replications=REPLICATIONS,method_caveats=['32 disjoint investigator-selected lexical groups, not a random sample of English words.','Both directions averaged within pair before resampling; all cells share bootstrap indices.','Intervals are descriptive; zero variance does not imply population certainty.','Partial cells are not imputed or assigned complete-panel rates.'])
