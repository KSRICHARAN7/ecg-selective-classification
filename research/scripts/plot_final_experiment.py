"""Prespecified scientific plots from saved Step9 results; no tuning."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'outputs/final_experiment'
COLORS={'raw_confidence':'#8172b3','confidence_adjusted':'#2866a5','quality_only':'#b07822','combined':'#14816b'}
LABELS={'raw_confidence':'Raw confidence','confidence_adjusted':'Confidence-adjusted','quality_only':'Quality only','combined':'Combined'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','figure.facecolor':'white','svg.fonttype':'none'})
def save(fig,name):
 for ext in ['png','svg']:fig.savefig(O/(name+'.'+ext),dpi=180,bbox_inches='tight')
 plt.close(fig)
def main():
 rows=list(csv.DictReader((O/'matched_coverage_metrics.csv').open()));result=json.loads((O/'primary_result.json').read_text())
 fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
 for ax,snr in zip(axes.flat,[24,12,6,0]):
  for m,color in COLORS.items():
   selected=[r for r in rows if r['condition']==f'em_{snr}'and r['method']==m];ax.plot([100*float(r['coverage'])for r in selected],[100*float(r['balanced_selective_error'])for r in selected],label=LABELS[m],color=color,marker='o',ms=3)
  ax.set_title(f'Withheld electrode-motion source: {snr} dB');ax.grid(alpha=.2);ax.axvline(90,color='#999',ls=':',lw=1)
 axes[1,0].set_xlabel('Actual retained recordings (%)');axes[1,1].set_xlabel('Actual retained recordings (%)');axes[0,0].set_ylabel('Class-balanced retained error (%)');axes[1,0].set_ylabel('Class-balanced retained error (%)')
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.005));fig.suptitle('Matched review budgets • fixed classifier and rankers');fig.tight_layout(rect=[0,.05,1,.95]);save(fig,'risk_coverage_em')
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for m,color in COLORS.items():
  r=[result['primary_conditions'][str(s)][m]for s in [24,12,6,0]]
  axes[0].plot(range(4),[100*x['balanced_selective_error']for x in r],marker='o',label=LABELS[m],color=color)
  axes[1].plot(range(4),[100*x['positive_retention']for x in r],marker='o',label=LABELS[m],color=color)
 for ax in axes:ax.set_xticks(range(4),['24','12','6','0']);ax.set_xlabel('Added-noise AC-RMS SNR (dB)');ax.grid(alpha=.2)
 axes[0].set_ylabel('Class-balanced retained error (%)');axes[1].set_ylabel('AFIB-positive recordings retained (%)');axes[1].axhline(80,color='#a94a40',ls='--',label='80% research guardrail');axes[1].set_ylim(0,105);handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.09),fontsize=9);fig.suptitle('Primary condition results at 90% retained coverage');fig.tight_layout(rect=[0,.06,1,.98]);save(fig,'primary_error_retention')
 with np.load(O/'primary_bootstrap.npz')as z:effects=z['effects']*100
 fig,ax=plt.subplots(figsize=(7,4));ax.hist(effects[np.isfinite(effects)],bins=40,color='#14816b',alpha=.8);ax.axvline(0,color='black',ls='--',label='No difference');ci=result['inference']['percentile95_CI']
 if ci:
  for v in ci:ax.axvline(v*100,color='#a94a40',ls=':')
 ax.set_xlabel('Combined − confidence-adjusted error (percentage points)');ax.set_ylabel('Participant-bootstrap replicates');ax.set_title('Primary effect uncertainty • 2,000 paired resamples');ax.legend();fig.tight_layout();save(fig,'primary_bootstrap')
 cm=json.loads((O/'classification_metrics.json').read_text());fig,axes=plt.subplots(1,2,figsize=(9,4))
 for name,color in [('clean','#2866a5'),('em_0','#a94a40')]:
  b=[v for v in cm[name]['calibration']if v['N']];axes[0].plot([v['mean_probability']for v in b],[v['observed_fraction']for v in b],'o-',label=name,color=color);axes[1].bar(np.array([v['low']+.05 for v in b])+(-.015 if name=='clean'else .015),[v['N']for v in b],width=.03,label=name,color=color,alpha=.8)
 axes[0].plot([0,1],[0,1],ls='--',color='#888');axes[0].set_xlabel('Mean predicted AFIB probability');axes[0].set_ylabel('Observed AFIB fraction');axes[1].set_xlabel('Predicted probability bin');axes[1].set_ylabel('Recordings');axes[1].set_yscale('log');axes[0].legend();axes[1].legend();fig.suptitle('Descriptive calibration • clean and most severe withheld noise');fig.tight_layout();save(fig,'calibration')
 fixed=list(csv.DictReader((O/'fixed_policy_metrics.csv').open()));fig,ax=plt.subplots(figsize=(8,4))
 for m,color in COLORS.items():
  vals=[next(r for r in fixed if r['condition']==f'em_{s}'and r['method']==m)for s in [24,12,6,0]];ax.plot(range(4),[100*float(r['coverage'])for r in vals],marker='o',label=LABELS[m],color=color)
 ax.axhline(90,ls=':',color='#777');ax.set_xticks(range(4),['24','12','6','0']);ax.set_xlabel('Withheld added-noise AC-RMS SNR (dB)');ax.set_ylabel('Actual retained recordings (%)');ax.set_title('Coverage with fixed fold9 score cutoffs • no test adjustment');ax.legend();ax.grid(alpha=.2);fig.tight_layout();save(fig,'fixed_policy_coverage')
 print('Saved five figures in PNG and SVG')
if __name__=='__main__':main()
