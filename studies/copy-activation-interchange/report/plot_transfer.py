"""Plot a retained-data interchange map; never loads a model or computes new CIs."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--table',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 rows=list(csv.DictReader(a.table.open()))
 a.out.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 fig,(ax,zoom)=plt.subplots(1,2,figsize=(12,5.5),gridspec_kw={'width_ratios':[1.7,1]})
 colors={'donor_word':'#1764a5','donor_final':'#c45b13'}
 labels={'donor_word':'Source-word position','donor_final':'Final prompt position'}
 for cond in colors:
  sub=[r for r in rows if r['condition']==cond and r['role']=='substantive']
  x=[int(r['block']) for r in sub];y=[float(r['mean_raw_delta']) for r in sub]
  lo=[y[i]-float(r['simultaneous_low']) for i,r in enumerate(sub)];hi=[float(r['simultaneous_high'])-y[i] for i,r in enumerate(sub)]
  ax.errorbar(x,y,yerr=[lo,hi],fmt='o-',color=colors[cond],capsize=4,lw=1.7,label=labels[cond])
  end=next(r for r in rows if r['condition']==cond and r['role']=='architectural identity')
  ax.scatter([29],[float(end['mean_raw_delta'])],marker='D',s=64,facecolors='white',edgecolors=colors[cond],linewidths=2,zorder=5)
  if cond=='donor_final':
   zoom.errorbar(x[:-1],y[:-1],yerr=[lo[:-1],hi[:-1]],fmt='o-',color=colors[cond],capsize=4,lw=1.7)
 ax.axvspan(27.3,30.7,color='#eeeeee',zorder=0);ax.text(29,20.05,'Architectural\nidentities',ha='center',va='top',fontsize=9)
 ax.annotate('Exact donor output',(29,float(next(r['mean_raw_delta'] for r in rows if r['block']=='29' and r['condition']=='donor_final'))),xytext=(18.5,19.5),fontsize=8,arrowprops={'arrowstyle':'-','color':'#666'})
 ax.annotate('Exact recipient output',(29,0),xytext=(21,1.4),fontsize=8,arrowprops={'arrowstyle':'-','color':'#666'})
 ax.set(xlim=(-1,31),ylim=(-1,20.7),xticks=[0,5,10,14,19,24,29],xlabel='Raw decoder block output (zero-indexed)',ylabel='Change in donor-minus-recipient logit margin',title='A. Interchange sensitivity across sampled blocks')
 ax.legend(loc='center left',frameon=False)
 zoom.set(xlim=(-1,20),ylim=(-.065,.35),xticks=[0,5,10,14,19],xlabel='Raw decoder block output',ylabel='Same margin change, expanded scale',title='B. Early final-position effects (zoom)')
 for axis in (ax,zoom):axis.axhline(0,color='#888',lw=.8,ls=':');axis.grid(axis='y',alpha=.15)
 fig.suptitle('Copy-target transfer shifts from source-word to final-position sensitivity',fontsize=14,y=.97)
 fig.text(.055,.075,'Means over 32 disjoint lexical pair groups; both directions averaged within group.\nBars: descriptive simultaneous 95% bootstrap bands over the 12 substantive cells (10,000 resamples).',fontsize=9)
 fig.text(.055,.018,'Block 29 diamonds are architectural identities, not circuit discoveries. Lines are guides across sampled sites, not localization of untested blocks.',fontsize=8.5)
 fig.subplots_adjust(left=.075,right=.98,bottom=.25,top=.84,wspace=.28)
 for ext in ('png','svg','pdf'):fig.savefig(a.out/f'transfer-map.{ext}',dpi=180)
 (a.out/'PLOTTING_RUNTIME.json').write_text(json.dumps({'matplotlib':matplotlib.__version__,'inference_performed':False,'intervals':'read unchanged from frozen summary'},indent=2)+'\n')
if __name__=='__main__':main()
