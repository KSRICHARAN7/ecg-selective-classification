"""Scientific summaries of the followup, retaining undefined metrics as gaps."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'outputs/generalization'
COLORS={'raw_confidence':'#8172b3','confidence_adjusted':'#2866a5','quality_only':'#b07822','combined':'#14816b'}
LABELS={'raw_confidence':'Raw confidence','confidence_adjusted':'Confidence-adjusted','quality_only':'Quality only','combined':'Combined'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','figure.facecolor':'white','svg.fonttype':'none'})
def value(row,key):return float(row[key])if row[key]not in ['',None]else np.nan
def save(fig,name):
 for ext in ['png','svg']:fig.savefig(O/(name+'.'+ext),dpi=180,bbox_inches='tight')
 plt.close(fig)
def main():
 rows=list(csv.DictReader((O/'matched_coverage_metrics.csv').open()));original=list(csv.DictReader((ROOT/'outputs/final_experiment/matched_coverage_metrics.csv').open()));s=json.loads((O/'generalization_summary.json').read_text())
 fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
 for ax,family in zip(axes,['bw','ma','em']):
  levels=[24,18,12,9,6,3,0,-6]
  for m,color in COLORS.items():
   points=[]
   for snr in levels:
    pool=original if snr in [24,12,6,0]else rows;name=f'{family}_{snr}'if pool is original else f'internal_{family}_{snr}';points.append(next(r for r in pool if r['condition']==name and r['method']==m and float(r['target_coverage'])==.9))
   ax.plot(range(8),[100*value(r,'balanced_selective_error')for r in points],color=color,label=LABELS[m],lw=1.5)
   for i,r in enumerate(points):ax.scatter(i,100*value(r,'balanced_selective_error'),color=color,marker='s'if levels[i]in [18,9,3,-6]else'o',s=25)
  ax.set_xticks(range(8),[str(v)for v in levels]);ax.set_xlabel('Added-noise SNR (dB)');ax.set_title(family+(' (originally withheld)'if family=='em'else' (development source)'));ax.grid(alpha=.2)
 axes[0].set_ylabel('Balanced retained error at 90% coverage (%)');h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.05));fig.suptitle('Severity transfer • squares are Step10 levels; circles are Step9 levels');fig.tight_layout(rect=[0,.05,1,.95]);save(fig,'severity_transfer')
 fig,axes=plt.subplots(1,2,figsize=(10,4));names=['external_clean']+[f'external_em_{x}'for x in [24,12,6,0]]
 for m,color in COLORS.items():
  selected=[next(r for r in rows if r['condition']==n and r['method']==m and float(r['target_coverage'])==.9)for n in names]
  axes[0].plot(range(5),[100*value(r,'balanced_selective_error')for r in selected],color=color,marker='o',label=LABELS[m]);axes[1].plot(range(5),[100*value(r,'positive_retention')for r in selected],color=color,marker='o',label=LABELS[m])
 axes[1].axhline(80,color='#a94a40',ls='--');axes[1].set_ylim(0,105)
 for ax in axes:ax.set_xticks(range(5),['Clean','24dB','12dB','6dB','0dB']);ax.grid(alpha=.2)
 axes[0].set_ylabel('Balanced retained error (%)');axes[1].set_ylabel('AFIB-positive records retained (%)');h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.05));fig.suptitle(f'External sampled cohort • {s["external_records"]:,} eligible ECGs • frozen models');fig.tight_layout(rect=[0,.06,1,.95]);save(fig,'external_error_retention')
 fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True,sharey=True)
 for ax,name in zip(axes,['external_clean','external_em_0']):
  for m,color in COLORS.items():
   selected=[r for r in rows if r['condition']==name and r['method']==m];ax.plot([100*value(r,'coverage')for r in selected],[100*value(r,'balanced_selective_error')for r in selected],color=color,marker='o',label=LABELS[m])
  ax.set_title(name);ax.set_xlabel('Actual retained records (%)');ax.grid(alpha=.2)
 axes[0].set_ylabel('Balanced retained error (%)');h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.05));fig.suptitle('External error–coverage • undefined class metrics remain gaps');fig.tight_layout(rect=[0,.06,1,.95]);save(fig,'external_risk_coverage')
 fig,ax=plt.subplots(figsize=(8,3.8));names=['internal_new_em','external_clean','external_em'];ticks=['Internal new em levels','External clean','External em levels']
 for i,name in enumerate(names):
  t=s['summaries'][name];effect=t['effect_combined_minus_confidence_adjusted'];ci=t['inference']['percentile95_CI']
  if effect is not None:ax.plot(effect*100,i,'o',color='#14816b')
  if ci is not None:ax.plot(np.asarray(ci)*100,[i,i],color='#14816b',lw=2)
 ax.axvline(0,color='#777',ls='--');ax.set_yticks(range(3),ticks);ax.invert_yaxis();ax.set_xlabel('Combined − confidence-adjusted error (percentage points)');ax.set_title('Exploratory transfer effects • pointwise 95% participant/record intervals');fig.tight_layout();save(fig,'transfer_effect_intervals')
 print('Saved four Step10 PNG/SVG figure pairs')
if __name__=='__main__':main()
