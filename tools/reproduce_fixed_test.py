"""Rerun frozen-model primary predictions using separately obtained data.

New public adapter; original scientific source is retained under research/.
Outputs aggregate tables only. Does not refit models, claim a new untouched
test cohort, or recreate the original development/freeze chronology.
"""
import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/scripts'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(path):
    result = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        result[name.strip().lstrip('./*')] = digest
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ptbxl-root', type=Path, required=True, help='Contains records100/ from PTB-XL 1.0.3')
    parser.add_argument('--metadata-dir', type=Path, help='Contains ptbxl_database.csv and scp_statements.csv; defaults to --ptbxl-root')
    parser.add_argument('--ptbxl-checksums', type=Path, required=True, help='Official PTB-XL 1.0.3 SHA256SUMS.txt')
    parser.add_argument('--nstdb-root', type=Path, required=True, help='Contains em.hea and em.dat from NSTDB 1.0.0')
    parser.add_argument('--output', type=Path, required=True, help='New directory; refuses an existing path')
    parser.add_argument('--bootstrap', action='store_true', help='Rerun the 2,000 paired participant resamples')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must be a new directory')
    from verify_release import main as verify
    verify()
    import numpy as np
    import torch
    import wfdb
    from audit_metadata import classify
    from build_partitions import check_integrity
    from model_definition import ECGClassifier
    from run_reliability_development import predict
    from reliability_methods import raw_confidence, score_saved
    from signal_tools import add_noise, quality_features, resample_noise
    from test_evaluation import METHODS, COVERAGES, metrics, selection_weights, bootstrap_primary

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    metadata = args.metadata_dir or args.ptbxl_root
    task = json.loads((ROOT/'research/config/task_definition.json').read_text())
    for name, digest in task['source_sha256'].items():
        if sha(metadata/name) != digest:
            raise ValueError(f'Wrong metadata version or modified file: {name}')
    source = json.loads((ROOT/'SOURCE_DATA.json').read_text())
    if sha(args.ptbxl_checksums) != source['ptbxl_inventory_sha256']:
        raise ValueError('Wrong PTB-XL publisher inventory')
    checks = inventory(args.ptbxl_checksums)
    noise_checks = inventory(ROOT/'provider-checksums/NSTDB_SHA256SUMS.txt')
    records = list(csv.DictReader((metadata/'ptbxl_database.csv').open()))
    check_integrity(records)
    rhythm = {r[''] for r in csv.DictReader((metadata/'scp_statements.csv').open()) if r['rhythm'] == '1.0'}
    selected = [r for r in records if int(r['strat_fold']) == 10 and classify(r, rhythm)[0] == 'eligible']
    if len(selected) != 2085 or len({r['patient_id'] for r in selected}) != 1812:
        raise ValueError('Unexpected final test cohort; no silent exclusions')
    waves = []
    for r in selected:
        base = args.ptbxl_root/r['filename_lr']
        for extension in ['.hea', '.dat']:
            path = base.with_suffix(extension)
            if sha(path) != checks[r['filename_lr']+extension]:
                raise ValueError('Source ECG checksum failed')
        d = wfdb.rdrecord(str(base), physical=False)
        if d.fs != 100 or d.d_signal.shape != (1000,12) or d.sig_name.count('II') != 1:
            raise ValueError('Unexpected source layout')
        k = d.sig_name.index('II')
        if d.units[k] != 'mV' or d.adc_gain[k] <= 0 or np.any(d.d_signal[:,k] == -32768):
            raise ValueError('Invalid lead II')
        if not np.array_equal(np.sum(d.d_signal.astype(np.int64),axis=0)%65536,np.asarray(d.checksum,dtype=np.int64)%65536):
            raise ValueError('WFDB signal checksum mismatch')
        waves.append((d.d_signal[:,k].astype(float)-d.baseline[k])/d.adc_gain[k])
    x = np.stack(waves).astype('float32')
    if not np.isfinite(x).all() or (x.std(axis=1) <= 1e-12).any():
        raise ValueError('Nonfinite or zero-power signal')
    ids = np.array([r['ecg_id'] for r in selected])
    people = np.array([r['patient_id'] for r in selected])
    y = np.array([int('AFIB' in ast.literal_eval(r['scp_codes'])) for r in selected])
    if y.sum() != 151:
        raise ValueError('Unexpected AF-positive count')
    for extension in ['.dat', '.hea']:
        if sha(args.nstdb_root/('em'+extension)) != noise_checks['em'+extension]:
            raise ValueError('Noise source checksum failed')
    header = wfdb.rdheader(str(args.nstdb_root/'em'))
    if header.fs != 360 or header.n_sig != 2 or header.sig_len < 1780*360:
        raise ValueError('Unexpected noise source layout')
    source_noise = wfdb.rdrecord(str(args.nstdb_root/'em'),sampfrom=1200*360,sampto=1780*360,physical=True).p_signal
    bank = np.empty((2,58,1000))
    for channel in range(2):
        for slot in range(58):
            bank[channel,slot] = resample_noise(source_noise[slot*3600:(slot+1)*3600,channel],360)
    checkpoint = torch.load(ROOT/'research/models/cnn_seed20261015.pt',map_location='cpu',weights_only=True)
    model = ECGClassifier(); model.load_state_dict(checkpoint['state_dict']); model.eval()
    threshold = json.loads((ROOT/'research/config/step8_specification.json').read_text())['classification_threshold']
    policies = json.loads((ROOT/'research/outputs/reliability_development/policy_thresholds.json').read_text())
    matched, fixed, conditions = [], [], []
    for name, snr in [('clean',None),('em_24',24),('em_12',12),('em_6',6),('em_0',0)]:
        if snr is None:
            inputs = x.astype(float)-x.mean(axis=1,keepdims=True,dtype=float)
        else:
            corrupted = []
            for signal, identifier in zip(x, ids):
                digest = hashlib.sha256(f'20261015|{identifier}|em'.encode()).digest()
                channel = int.from_bytes(digest[:8],'big')%2; slot = int.from_bytes(digest[8:16],'big')%58
                noisy, injected = add_noise(signal, bank[channel,slot], snr)
                centered = signal.astype(float)-signal.astype(float).mean()
                measured = 20*np.log10(np.sqrt(np.mean(centered**2))/np.sqrt(np.mean(injected**2)))
                if abs(measured-snr)>1e-10: raise ValueError('Incorrect SNR')
                corrupted.append(noisy)
            inputs = np.stack(corrupted)
        q = np.stack([quality_features(w) for w in inputs])
        p = predict(model,checkpoint['training_scale_mV'],inputs.astype('float32'))
        pred = (p>=threshold).astype(int)
        scores = {'raw_confidence':raw_confidence(p),**{m:score_saved(ROOT/f'research/models/reliability/{m}.json',p,q) for m in METHODS if m!='raw_confidence'}}
        for method in METHODS:
            for coverage in COVERAGES:
                matched.append(dict(condition=name,method=method,target_coverage=coverage,**metrics(y,pred,selection_weights(scores[method],ids,coverage))))
            fixed.append(dict(condition=name,method=method,threshold=policies[method]['threshold'],**metrics(y,pred,(scores[method]<=policies[method]['threshold']).astype(int))))
        if snr is not None: conditions.append(dict(ids=ids,participant=people,y=y,pred=pred,**scores))
        print(f'Computed {name}',flush=True)
    comparisons = []
    for filename, actual in [('matched_coverage_metrics.csv',matched),('fixed_policy_metrics.csv',fixed)]:
        expected = list(csv.DictReader((ROOT/'research/outputs/final_experiment'/filename).open()))
        for row in actual:
            key = 'target_coverage' if 'target_coverage' in row else 'threshold'
            reference = next(r for r in expected if r['condition']==row['condition'] and r['method']==row['method'] and float(r[key])==row[key])
            for k,v in row.items():
                if isinstance(v,(int,float)) and not np.isclose(v,float(reference[k]),atol=1e-10,rtol=1e-10):
                    comparisons.append({'condition':row['condition'],'method':row['method'],'metric':k,'expected':float(reference[k]),'actual':v})
    report = {'matched_rows':len(matched),'fixed_policy_rows':len(fixed),'differences':comparisons,'all_aggregate_cells_match':not comparisons,'scope':'fixed primary model/recorded em noise replay; not retraining, external replication, or original cross-role waveform-overlap audit'}
    if args.bootstrap:
        inference,_,_ = bootstrap_primary(conditions)
        report['recomputed_inference'] = inference
        saved = json.loads((ROOT/'research/outputs/final_experiment/primary_result.json').read_text())['inference']
        report['bootstrap_interval_matches'] = bool(np.allclose(inference['percentile95_CI'],saved['percentile95_CI'],atol=1e-10,rtol=1e-10))
    args.output.mkdir(parents=True)
    for filename, rows in [('matched_coverage_metrics.csv',matched),('fixed_policy_metrics.csv',fixed)]:
        with (args.output/filename).open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if comparisons or report.get('bootstrap_interval_matches') is False:
        raise SystemExit('Numerical reproduction differs; inspect comparison.json rather than replacing saved evidence.')


if __name__ == '__main__':
    main()
