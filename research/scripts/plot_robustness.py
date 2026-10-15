"""Scientific Step11 figures from saved sensitivity outputs."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cohort_loader import ROOT
O=ROOT/'outputs/robustness'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})
def save(fig,name):
 for ext in ['png','svg']:fig.savefig(O/(name+'.'+ext),dpi=180,bbox_inches='tight')
 plt.close(fig)
def forest(ax,effect,ci,i,color='#14816b'):
 if effect is not None:ax.plot(effect*100,i,'o',color=color)
 if ci is not None:ax.plot(np.array(ci)*100,[i,i],color=color,lw=2)
def main():
 s=json.loads((O/'robustness_summary.json').read_text());seeds=json.loads((O/'seeds/seed_summary.json').read_text());old=json.loads((ROOT/'outputs/final_experiment/primary_result.json').read_text());variants=[('20261015 original',old['effect_combined_minus_confidence_adjusted'],old['inference']['percentile95_CI'],[old['primary_conditions'][str(sn)]['combined']['positive_retention']for sn in [24,12,6,0]])]+[(str(t['seed']),t['primary_sensitivity']['effect'],t['primary_sensitivity']['percentile95_CI'],t['primary_sensitivity']['AF_retention'])for t in seeds if t['status']=='completed']
 fig,axes=plt.subplots(1,2,figsize=(10,3.5))
 for i,(name,e,ci,ret)in enumerate(variants):forest(axes[0],e,ci,i);axes[1].plot(np.array([min(ret),max(ret)])*100,[i,i],color='#2866a5',lw=2);axes[1].plot(np.array(ret)*100,np.full(4,i),'o',color='#2866a5',ms=4)
 for ax in axes:ax.set_yticks(range(len(variants)),[v[0]for v in variants]);ax.invert_yaxis()
 axes[0].axvline(0,ls='--',color='#888');axes[0].set_xlabel('Combined − adjusted error (pp)');axes[0].set_title('Pointwise 95% participant intervals');axes[1].axvline(80,ls='--',color='#a94a40');axes[1].set_xlim(0,105);axes[1].set_xlabel('AF cases retained across em severities (%)');axes[1].set_title('80% criterion must hold at every severity');fig.suptitle('Training-seed sensitivity • all three seeds');fig.tight_layout();save(fig,'seed_sensitivity')
 fig,axes=plt.subplots(1,2,figsize=(11,5));names=list(s['ablations'])
 for i,name in enumerate(names):
  t=s['ablations'][name]['comparison'];forest(axes[0],t['effect'],t['percentile95_CI'],i);axes[1].plot(min(t['AF_retention'])*100,i,'o',color='#2866a5')
 for ax in axes:ax.set_yticks(range(len(names)),[n.replace('_',' ')for n in names]);ax.invert_yaxis()
 axes[0].axvline(0,color='#888',ls='--');axes[0].set_xlabel('Ablation − corresponding full ranker error (pp)');axes[0].set_title('Pointwise 95% intervals • 90% budget');axes[1].axvline(80,color='#a94a40',ls='--');axes[1].set_xlim(0,105);axes[1].set_xlabel('Lowest AF retention across four em levels (%)');axes[1].set_title('Minimum AF retention across the four conditions');fig.tight_layout();save(fig,'feature_ablations')
 groups=s['group_results'];fig,ax=plt.subplots(figsize=(9,5));names=list(groups)
 for i,name in enumerate(names):
  t=groups[name]
  if t['status']=='completed':forest(ax,t['comparison']['effect'],t['comparison']['percentile95_CI'],i)
  else:ax.text(.98,i,'insufficient class participants',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8,color='#777')
 ax.set_yticks(range(len(names)),[n.replace('_',' ')for n in names]);ax.invert_yaxis();ax.axvline(0,color='#888',ls='--');ax.set_xlabel('Combined − adjusted error (pp)');ax.set_title('Exploratory cohort and subgroup sensitivities');fig.tight_layout();save(fig,'subgroup_sensitivities')
 flags=list(csv.DictReader((O/'positive_case_flagging.csv').open()));fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
 for ax,cohort,prefix in zip(axes,['internal','external'],['','external_']):
  for snr,color in zip([24,12,6,0],['#14816b','#2866a5','#8172b3','#b07822']):
   vals=[next(r for r in flags if r['cohort']==cohort and r['condition']==f'{prefix}em_{snr}'and r['method']=='combined'and r['policy']=='matched_budget'and float(r['target_coverage'])==c)for c in [.8,.9,.95]];ax.plot([80,90,95],[100*float(r['positive_retention'])for r in vals],'o-',color=color,label=f'{snr} dB')
  ax.axhline(80,color='#a94a40',ls='--');ax.set_ylim(0,105);ax.set_xlabel('Retained-record budget (%)');ax.set_title(cohort+' cohort');ax.grid(alpha=.2)
 axes[0].set_ylabel('AFIB-positive records retained (%)');axes[1].legend(title='Added em SNR');fig.suptitle('Combined ranker • budget sensitivity without retraining');fig.tight_layout();save(fig,'review_budget_sensitivity')
 fig,axes=plt.subplots(1,2,figsize=(11,4));methods=['raw_confidence','confidence_adjusted','quality_only','combined'];labels=['Raw','Adjusted','Quality','Combined'];names=['Retained TP','Retained FN','Deferred TP','Deferred FN'];colors=['#14816b','#c49736','#7696b6','#b45d54']
 for ax,cohort,condition in zip(axes,['internal','external'],['em_24','external_clean']):
  vals=[next(r for r in flags if r['cohort']==cohort and r['condition']==condition and r['method']==m and r['policy']=='matched_budget'and float(r['target_coverage'])==.9)for m in methods];bottom=np.zeros(4);values=[[float(r['TP'])for r in vals],[float(r['FN'])for r in vals],[float(r['true_positives_deferred'])for r in vals],[float(r['AF_records_deferred'])-float(r['true_positives_deferred'])for r in vals]]
  for name,color,v in zip(names,colors,values):ax.bar(range(4),v,bottom=bottom,label=name,color=color);bottom+=np.array(v)
  ax.set_xticks(range(4),labels);ax.set_title(cohort+': '+condition);ax.set_ylabel('Original AFIB-positive recordings')
 h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.04));fig.suptitle('Where AF cases go at 90% overall retention');fig.tight_layout(rect=[0,.06,1,.95]);save(fig,'positive_case_deferral')
 print('Saved five Step11 PNG/SVG figure pairs')
if __name__=='__main__':main()
