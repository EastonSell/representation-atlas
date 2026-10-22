"""Deterministic, scientific-only Markdown rendering; no model or network imports."""
import json

def render(summary,audit):
 valid=summary['complete'] and audit.get('passed',False) and audit.get('execution_accepted',False)
 lines=['# Copy-target interchange map','', 'Complete audited map.' if valid else 'Incomplete or unaudited evidence. No mechanistic conclusion is warranted.', '',f"Observed patch vectors: {summary['observed_patch_count']}/2688.",'','All cells, including failures and controls, follow. Missing data are not imputed.','','| Block | Condition | Role | Observed | Mean raw delta | Median raw delta |','|---|---|---|---:|---:|---:|']
 for cell in summary['cells']:
  role='architectural control' if cell['block']==29 or cell['condition'] not in ('donor_word','donor_final') else 'substantive'
  lines.append('| %s | %s | %s | %s | %s | %s |'%(cell['block'],cell['condition'],role,cell['observed_count'],cell.get('mean_raw_transfer_delta','missing'),cell.get('median_raw_transfer_delta','missing')))
 lines.extend(['','## Complete numerical record','','The machine-readable summary includes all pair-level directional observations, explicit switch/off-target denominators, normalized eligibility, percentile intervals and simultaneous raw-margin bands.','', 'The 32 disjoint investigator-selected lexical pair groups are the resampling units, not a random sample of English words. Both directions are averaged within group. Intervals are descriptive. Replacement effects do not establish necessity, exclusive storage, a unique circuit or brain correspondence.','','Missing cells: '+str(len(summary['missing_cells']))+'.',''])
 return '\n'.join(lines)
