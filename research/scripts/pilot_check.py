"""Development-only waveform, numerical and runtime feasibility check.

Uses random synthetic targets for timing. No clinical model fit is saved and
no classification-performance metric is computed.
"""
import csv,hashlib,json,platform,time,statistics,resource
from pathlib import Path
import importlib.metadata as md
import numpy as np
import wfdb
from signal_tools import add_noise,quality_features,resample_noise

ROOT=Path(__file__).resolve().parents[1]


def main():
    selected=json.loads((ROOT/'config/pilot_selection.json').read_text())
    cohort={r['ecg_id']:r for r in csv.DictReader((ROOT/'outputs/development_cohort_manifest.csv').open())}
    signals=[];checks=[]
    for row in selected:
        actual=cohort[row['ecg_id']]
        assert int(actual['strat_fold'])<=6 and row['filename_lr']==actual['filename_lr']
        base=str(ROOT/'data/pilot'/row['filename_lr'])
        digital=wfdb.rdrecord(base,physical=False)
        physical=wfdb.rdrecord(base,physical=True)
        assert physical.fs==100 and physical.p_signal.shape==(1000,12)
        assert physical.sig_name.count('II')==1
        idx=physical.sig_name.index('II')
        assert physical.units[idx]=='mV'
        assert np.isfinite(physical.p_signal).all()
        calculated=np.sum(digital.d_signal.astype(np.int64),axis=0)%65536
        expected=np.asarray(digital.checksum,dtype=np.int64)%65536
        assert np.array_equal(calculated,expected),'WFDB signal checksum mismatch'
        assert np.array_equal(digital.d_signal[0],digital.init_value),'First sample mismatch'
        mapped=(digital.d_signal[:,idx]-digital.baseline[idx])/digital.adc_gain[idx]
        assert np.allclose(mapped,physical.p_signal[:,idx],rtol=0,atol=1e-12)
        x=physical.p_signal[:,idx];signals.append(x)
        checks.append({'ecg_id':row['ecg_id'],'fold':int(row['strat_fold']),'lead_name':'II','lead_index':idx,'samples':len(x),'fs':physical.fs,'unit':'mV','signal_checksum_verified':True,'finite':True})
    noise={};noise_checks=[]
    for name in ['bw','ma']:
        base=str(ROOT/'data/pilot/noise'/name)
        header=wfdb.rdheader(base)
        r=wfdb.rdrecord(base,sampfrom=0,sampto=int(header.fs*10),physical=True)
        assert np.isfinite(r.p_signal).all()
        y=resample_noise(r.p_signal[:,0],header.fs)
        assert y.shape==(1000,)
        noise[name]=y
        noise_checks.append({'record':name,'source_hz':header.fs,'source_total_samples':header.sig_len,'pilot_source_seconds':[0,10],'selected_channel':0,'resampled_samples':len(y),'resample_filter':'scipy.signal.resample_poly Kaiser beta5, constant padding'})
    rows=[];max_snr_error=0
    for meta,x in zip(selected,signals):
        cases=[('original',None,x)]
        for name,n in noise.items():
            for snr in [24,12,6,0]:
                y,added=add_noise(x,n,snr)
                measured=10*np.log10(np.mean((x-x.mean())**2)/np.mean(added**2))
                max_snr_error=max(max_snr_error,abs(measured-snr))
                cases.append((name,snr,y))
        for kind,snr,y in cases:
            f=quality_features(y)
            assert not np.isinf(f).any()
            rows.append({'ecg_id':meta['ecg_id'],'condition':kind,'snr_db':snr,**{f'quality_{i}':None if np.isnan(v) else float(v) for i,v in enumerate(f)}})
    with (ROOT/'outputs/pilot_quality_features.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import torch
    from model_definition import ECGClassifier
    torch.manual_seed(20261015);torch.set_num_threads(2)
    model=ECGClassifier();model.train()
    batch=torch.from_numpy(np.stack(signals).astype('float32')).unsqueeze(1).repeat(8,1,1)
    batch=batch-batch.mean(dim=2,keepdim=True)
    batch=batch/batch.std(unbiased=False)
    target=torch.randint(0,2,(len(batch),)).float()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    times=[]
    for i in range(15):
        begin=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(batch),target)
        loss.backward();optimizer.step()
        if i>=5:times.append(time.perf_counter()-begin)
    model.eval();inference=[]
    with torch.no_grad():
        for _ in range(20):
            begin=time.perf_counter();model(batch);inference.append(time.perf_counter()-begin)
    packages=['numpy','scipy','pandas','scikit-learn','matplotlib','torch','wfdb']
    env={p:md.version(p) for p in packages}
    summary={'stage':'Step4 development-only pilot','clinical_performance_computed':False,'saved_clinical_model':False,'clinical_labels_used_as_training_targets':False,'records_checked':len(checks),'waveform_checks':checks,'noise_checks':noise_checks,'withheld_em_requested':False,'feature_rows':len(rows),'maximum_absolute_snr_error_db':max_snr_error,'parameter_count':sum(p.numel() for p in model.parameters()),'runtime':{'batch_size':len(batch),'cpu_threads':torch.get_num_threads(),'device':'cpu','warmup_batches':5,'measured_training_batches':10,'median_forward_backward_update_seconds':statistics.median(times),'median_forward_batch_seconds':statistics.median(inference),'timing_targets':'random synthetic binary targets; weights discarded','rough_60_epoch_seconds':statistics.median(times)*int(np.ceil(8301/64))*60,'projection_limit':'not a full training benchmark; excludes loading, validation, logging and setup'},'process_peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'environment':{'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),'packages':env},'scope_limit':'eight selected training-fold recordings; not a cohort quality audit or model-performance result'}
    (ROOT/'outputs/pilot_check.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['waveform_checks','noise_checks']},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True)
    t=np.arange(1000)/100
    for ax,index in zip(axes,[0,4]):
        x=signals[index];y,_=add_noise(x,noise['ma'],6)
        ax.plot(t,x-x.mean(),label='Original recording',lw=.8)
        ax.plot(t,y,label='Added muscle noise at6dB',lw=.6,alpha=.65)
        ax.set_ylabel('mV');ax.set_title('Development-only waveform check | ECG '+selected[index]['ecg_id'],fontsize=10);ax.legend(fontsize=8,loc='upper right')
    axes[-1].set_xlabel('Seconds');fig.tight_layout();fig.savefig(ROOT/'outputs/pilot_waveforms.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
