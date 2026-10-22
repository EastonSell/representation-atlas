"""Extract display tables from a sealed summary after an accepted scientific audit."""
import argparse,csv,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 audit=json.loads(a.audit.read_text());assert audit['passed'] and audit['execution_accepted']
 s=json.loads(a.summary.read_text());assert s['complete'] and len(s['cells'])==42
 a.out.mkdir(exist_ok=True,parents=True)
 fields=['block','condition','role','observed_count','mean_raw_delta','pointwise_low','pointwise_high','simultaneous_low','simultaneous_high','mean_normalized_transfer','normalized_pair_count','donor_replacements','replacement_denominator','genuine_switches','switch_denominator','off_target_count','off_target_denominator']
 rows=[]
 for c in s['cells']:
  ci=c['raw_delta_simultaneous_95'];norm=c['descriptive_metrics']['normalized_transfer']
  rows.append(dict(zip(fields,[c['block'],c['condition'],'substantive' if c['block']!=29 and c['condition'] in ('donor_word','donor_final') else 'architectural identity',c['observed_count'],c['mean_raw_transfer_delta'],*c['raw_delta_pointwise_95'],*(ci or ['','']),norm['mean'],norm['pair_count'],c['donor_replacement']['numerator'],c['donor_replacement']['denominator'],c['genuine_switch']['numerator'],c['genuine_switch']['denominator'],c['off_target']['numerator'],c['off_target']['denominator']])))
 for name,subset in [('all-cells.csv',rows),('transfer-plot-input.csv',[r for r in rows if r['condition'] in ('donor_word','donor_final')])]:
  with (a.out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(subset)
 provenance={'operation':'deterministic display extraction, no new inference or statistics','summary_sha256':hashlib.sha256(a.summary.read_bytes()).hexdigest(),'audit_sha256':hashlib.sha256(a.audit.read_bytes()).hexdigest(),'resampling_units':'32 disjoint lexical pair groups, not randomly sampled independent English words','normalized_pairs':31,'normalized_exclusion_pair_ids':[29]}
 (a.out/'DERIVATION.json').write_text(json.dumps(provenance,indent=2)+'\n')
if __name__=='__main__':main()
