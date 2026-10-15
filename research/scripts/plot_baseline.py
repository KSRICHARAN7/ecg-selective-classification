"""Development-only baseline figures; no held-out risk experiment results."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve
from cohort_loader import ROOT


def main():
    out=ROOT/'outputs/baseline'
    summary=json.loads((out/'baseline_summary.json').read_text())
    history=json.loads((out/'epoch_history.json').read_text())
    rows=list(csv.DictReader((out/'checkpoint_predictions.csv').open()))
    y=np.asarray([int(r['reference_AFIB'])for r in rows]);p=np.asarray([float(r['p_AFIB'])for r in rows])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,3,figsize=(13,4.1),layout='constrained')
    epochs=[h['epoch']for h in history]
    axes[0].plot(epochs,[h['train_BCE']for h in history],label='Training',color='#276891')
    axes[0].plot(epochs,[h['checkpoint_BCE']for h in history],label='Checkpoint',color='#b65030')
    axes[0].axvline(summary['best_epoch'],color='0.4',ls=':',label='Selected epoch')
    axes[0].set(xlabel='Epoch',ylabel='Binary cross entropy',title='A. Training and selection');axes[0].legend(frameon=False)
    fpr,tpr,_=roc_curve(y,p)
    axes[1].plot(fpr,tpr,color='#276891',label=f"CNN AUROC {summary['CNN_checkpoint_metrics']['AUROC']:.3f}")
    axes[1].plot([0,1],[0,1],ls='--',color='0.65',lw=1)
    m=summary['CNN_checkpoint_metrics'];axes[1].scatter([1-m['specificity']],[m['sensitivity']],color='#b65030',s=25,label='Selected threshold')
    axes[1].set(xlabel='False-positive rate',ylabel='Sensitivity',title='B. Checkpoint ROC',xlim=(-.02,1.02),ylim=(-.02,1.02));axes[1].legend(frameon=False,loc='lower right')
    precision,recall,_=precision_recall_curve(y,p)
    axes[2].plot(recall,precision,color='#276891',label=f"CNN AP {m['AUPRC']:.3f}")
    axes[2].axhline(y.mean(),ls='--',color='0.65',label=f'Positive fraction {y.mean():.3f}')
    axes[2].set(xlabel='Recall',ylabel='Precision',title='C. Checkpoint precision–recall',xlim=(-.02,1.02),ylim=(-.02,1.02));axes[2].legend(frameon=False,loc='lower left',bbox_to_anchor=(0,.12))
    fig.suptitle('Clean lead-II baseline — fold7 development estimates, not independent validation',fontsize=12)
    fig.savefig(out/'baseline_development.png',dpi=180);fig.savefig(out/'baseline_development.svg');plt.close(fig)
    print('Saved baseline development PNG and SVG')


if __name__=='__main__':main()
