"""Step7 numerical verification across all classifier-training ECGs.

No clinical predictions, classifier tuning or ranker fitting are performed.
"""
import csv,hashlib,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from cohort_loader import ROOT,manifest
from fault_library import DevelopmentFaultLibrary,FAMILIES,SNR_LEVELS


def main():
    begin=time.perf_counter();out=ROOT/'outputs/fault_library';out.mkdir(exist_ok=True)
    specification=json.loads((ROOT/'config/fault_library_specification.json').read_text())
    source_path=ROOT/'data/derived/train_leadII.npz'
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=specification['training_array_sha256']:raise ValueError('Training array changed after specification')
    rows=manifest('train');ids=[r['ecg_id']for r in rows]
    with np.load(source_path,allow_pickle=False)as z:
        if z['ids'].tolist()!=ids:raise ValueError('Array role mismatch')
        signals=z['x'].copy()  # Reference labels are not used by this audit.
    if signals.shape!=(8301,1000)or not np.isfinite(signals).all():raise ValueError('Invalid complete training array')
    library=DevelopmentFaultLibrary();usage={name:Counter()for name in FAMILIES}
    audit=out/'clean_corrupted_pair_manifest.csv';previews=[];preview_ids={r['ecg_id']for r in json.loads((ROOT/'config/pilot_selection.json').read_text())}
    fields=['ecg_id','family','role','channel','slot','source_start_second','source_stop_second','seed','requested_snr_db','measured_snr_db','absolute_snr_error_db','clean_AC_RMS_mV','injected_AC_RMS_mV','clean_sha256','corrupted_sha256','injected_sha256']
    max_error=0.;pairs=0;max_reconstruction_error=0.
    with audit.open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for n,(identifier,signal)in enumerate(zip(ids,signals),1):
            for family in FAMILIES:
                for severity in SNR_LEVELS:
                    clean,corrupted,injected,meta=library.make_pair(signal,identifier,family,severity)
                    if not np.isfinite(corrupted).all()or not np.isfinite(injected).all():raise ValueError('Nonfinite generated pair')
                    measured=float(20*np.log10(np.sqrt(np.mean(clean**2))/np.sqrt(np.mean(injected**2))))
                    error=abs(measured-severity);max_error=max(max_error,error)
                    reconstruction=float(np.max(np.abs((corrupted-clean)-injected)));max_reconstruction_error=max(max_reconstruction_error,reconstruction)
                    if error>1e-9 or reconstruction>1e-10:raise ValueError('Generated corruption failed numerical tolerance')
                    writer.writerow({**meta,'measured_snr_db':measured,'absolute_snr_error_db':error,'clean_AC_RMS_mV':float(np.sqrt(np.mean(clean**2))),'injected_AC_RMS_mV':float(np.sqrt(np.mean(injected**2))),'clean_sha256':hashlib.sha256(clean.tobytes()).hexdigest(),'corrupted_sha256':hashlib.sha256(corrupted.tobytes()).hexdigest(),'injected_sha256':hashlib.sha256(injected.tobytes()).hexdigest()})
                    pairs+=1
                    if severity==SNR_LEVELS[0]:usage[family][(meta['channel'],meta['slot'])]+=1
                    if identifier in preview_ids:previews.append((identifier,family,severity,clean,corrupted,injected))
            if n%1000==0:print(f'Verified all8 fault conditions for {n}/{len(ids)} training ECGs',flush=True)
    # Keep actual numerical waveform pairs for the pre-existing8-record pilot;
    # every other pair can be regenerated from the recorded source assignment.
    np.savez_compressed(out/'paired_training_previews.npz',ids=np.asarray([p[0]for p in previews]),families=np.asarray([p[1]for p in previews]),snr_db=np.asarray([p[2]for p in previews]),clean=np.stack([p[3]for p in previews]),corrupted=np.stack([p[4]for p in previews]),injected=np.stack([p[5]for p in previews]))
    (out/'source_segment_audit.json').write_text(json.dumps(library.segment_audit,indent=2)+'\n')
    collisions={family:{'possible_channel_window_pairs':116,'distinct_pairs_used':len(counts),'recordings_beyond_first_use':len(ids)-len(counts),'maximum_recordings_sharing_one_pair':max(counts.values()),'usage':[{'channel':key[0],'slot':key[1],'recordings':value}for key,value in sorted(counts.items())]}for family,counts in usage.items()}
    (out/'noise_reuse_audit.json').write_text(json.dumps(collisions,indent=2)+'\n')
    summary={'stage':'Step7 verified development fault library','training_records':len(ids),'corrupted_pairs':pairs,'preview_pairs_saved':len(previews),'families':list(FAMILIES),'SNR_dB':list(SNR_LEVELS),'source_region_decoded_seconds':[0,580],'source_segments_verified':len(library.segment_audit),'maximum_absolute_SNR_error_dB':max_error,'maximum_additive_reconstruction_error_mV':max_reconstruction_error,'all_pairs_finite':True,'no_record_or_segment_dropped':True,'withheld_em_requested_or_decoded':False,'validation_or_test_region_decoded':False,'clinical_predictions_computed':False,'classifier_retrained':False,'ranker_fitted':False,'recorded_noise_reuse_not_independent_donors':True,'source_hashes':library.source_hashes,'pair_manifest_sha256':hashlib.sha256(audit.read_bytes()).hexdigest(),'seconds':time.perf_counter()-begin,'scope':'Controlled added recorded-noise verification; no clinical or equipment-failure validity claim'}
    (out/'verification_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
