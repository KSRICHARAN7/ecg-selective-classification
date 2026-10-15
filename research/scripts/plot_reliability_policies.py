"""Policy coverage only. No validation target or performance metric used."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cohort_loader import ROOT


def main():
    out=ROOT/'outputs/reliability_development';d=json.loads((out/'policy_thresholds.json').read_text())
    names=list(d);conditions=['clean']+[f'{f}_{s}'for f in ['bw','ma']for s in [24,12,6,0]]
    colors=['#636363','#276b92','#a55b2c','#48845b']
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for i,name in enumerate(names):
        values=[d[name]['coverage_by_condition'][c]for c in conditions]
        axes[0].plot(np.arange(len(conditions)),values,marker='o',lw=1.3,color=colors[i],label=name.replace('_',' '))
    axes[0].axhline(.9,color='0.65',ls='--',lw=1)
    axes[0].set_xticks(np.arange(len(conditions)),conditions,rotation=45,ha='right')
    axes[0].set(ylabel='Retained fraction',ylim=(0,1.03),title='A. Coverage by source condition')
    axes[0].legend(frameon=False,fontsize=8,loc='lower left')
    x=np.arange(len(names));width=.34
    axes[1].bar(x-width/2,[d[n]['actual_weighted_coverage']for n in names],width,label='Specified mixture',color='#276b92')
    axes[1].bar(x+width/2,[d[n]['actual_unweighted_coverage']for n in names],width,label='Equal9-condition mixture',color='#a55b2c')
    axes[1].axhline(.9,color='0.65',ls='--',lw=1)
    axes[1].set_xticks(x,[n.replace('_','\n')for n in names]);axes[1].set(ylabel='Retained fraction',ylim=(0,1.03),title='B. Achieved policy coverage');axes[1].legend(frameon=True,facecolor='white',framealpha=1,loc='lower left',fontsize=8)
    fig.suptitle('Fold9 unlabeled policy selection — coverage is not classification accuracy',fontsize=12)
    fig.savefig(out/'policy_coverage.png',dpi=170);fig.savefig(out/'policy_coverage.svg');plt.close(fig)
    print('Saved policy coverage PNG and SVG')


if __name__=='__main__':main()
