"""Preview predeclared pilot ECGs, without choosing examples on model outcomes."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cohort_loader import ROOT


def main():
    out=ROOT/'outputs/fault_library'
    with np.load(out/'paired_training_previews.npz',allow_pickle=False)as z:
        arrays={key:z[key].copy()for key in z.files}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    time=np.arange(1000)/100
    for identifier in ['12394','8491']:
        fig,axes=plt.subplots(2,4,figsize=(14,5),sharex=True,layout='constrained')
        for row,family in enumerate(['bw','ma']):
            indices=np.flatnonzero((arrays['ids']==identifier)&(arrays['families']==family))
            low=min(arrays['clean'][indices].min(),arrays['corrupted'][indices].min());high=max(arrays['clean'][indices].max(),arrays['corrupted'][indices].max())
            pad=.08*(high-low)
            for col,snr in enumerate([24,12,6,0]):
                matches=np.flatnonzero((arrays['ids']==identifier)&(arrays['families']==family)&(arrays['snr_db']==snr))
                if len(matches)!=1:raise ValueError('Preview pair mismatch')
                index=matches[0];ax=axes[row,col]
                ax.plot(time,arrays['corrupted'][index],color='#bd5a30',lw=.7,label='ECG + noise')
                ax.plot(time,arrays['clean'][index],color='#246b93',lw=.65,alpha=.9,label='Original ECG')
                ax.set(title=f'{family.upper()} | {snr} dB',ylim=(low-pad,high+pad),xlim=(0,10))
                if col==0:ax.set_ylabel('Lead II (mV)')
                if row==1:ax.set_xlabel('Time (s)')
        axes[0,0].legend(frameon=False,loc='upper right',fontsize=8)
        fig.suptitle(f'ECG {identifier}: deterministic development noise pairs — no clinical performance claim',fontsize=12)
        fig.savefig(out/f'fault_preview_ECG{identifier}.png',dpi=160);fig.savefig(out/f'fault_preview_ECG{identifier}.svg');plt.close(fig)
    print('Saved two waveform preview figures (PNG and SVG)')


if __name__=='__main__':main()
