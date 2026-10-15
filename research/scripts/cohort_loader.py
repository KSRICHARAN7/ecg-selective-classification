"""Role-aware waveform loading and fail-closed final-test access guard.

This prevents accidental use through the pipeline, not access to the provider's
raw metadata. It is not encryption or an institutional data-access boundary.
"""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import wfdb

ROOT=Path(__file__).resolve().parents[1]


def require_protocol_lock(root=ROOT):
    path=root/'config/protocol_lock.json'
    if not path.exists():raise PermissionError('Final-test waveform access requires a recorded protocol lock')
    lock=json.loads(path.read_text());protocol=root/'config/analysis_protocol.json'
    if lock.get('sha256')!=hashlib.sha256(protocol.read_bytes()).hexdigest():raise PermissionError('Protocol lock hash mismatch')
    if json.loads(protocol.read_text()).get('status')!='frozen':raise PermissionError('Protocol is not frozen')


def manifest(role,root=ROOT):
    audit=json.loads((root/'outputs/partition_audit.json').read_text())
    path=root/'outputs/partitions'/(role+'.csv')
    expected=audit['role_manifest_sha256'].get(path.relative_to(root).as_posix())
    if expected!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('Role manifest modified')
    with path.open()as f:return list(csv.DictReader(f))


def cached_base(row,root=ROOT):
    for directory in ['data/cohort','data/pilot']:
        path=root/directory/row['filename_lr']
        if path.with_suffix('.hea').exists() and path.with_suffix('.dat').exists():return path
    return None


def load_records(role,ids,root=ROOT):
    if role=='test':require_protocol_lock(root)
    rows={r['ecg_id']:r for r in manifest(role,root)}
    ids=[str(i)for i in ids]
    if len(set(ids))!=len(ids) or any(i not in rows for i in ids):raise ValueError('IDs must be unique members of the requested role')
    signals=[]
    for identifier in ids:
        row=rows[str(identifier)];base=cached_base(row,root)
        if base is None:raise FileNotFoundError(f'Missing waveform for ECG{identifier}; no silent cohort reduction')
        digital=wfdb.rdrecord(str(base),physical=False)
        if digital.fs!=100 or digital.d_signal.shape!=(1000,12)or digital.sig_name.count('II')!=1:raise ValueError('Unexpected waveform layout')
        if not np.array_equal(np.sum(digital.d_signal.astype(np.int64),axis=0)%65536,np.asarray(digital.checksum,dtype=np.int64)%65536):raise ValueError('Signal checksum mismatch')
        index=digital.sig_name.index('II')
        if digital.units[index]!='mV' or digital.adc_gain[index]<=0:raise ValueError('Unexpected units or gain')
        x=(digital.d_signal[:,index].astype(float)-digital.baseline[index])/digital.adc_gain[index]
        if not np.isfinite(x).all() or np.any(digital.d_signal[:,index]==-32768):raise ValueError('Missing or nonfinite lead II samples')
        signals.append(x)
    return np.stack(signals)


def load_complete_role(role,root=ROOT):
    rows=manifest(role,root)
    return load_records(role,[r['ecg_id']for r in rows],root)


def inventory(root=ROOT):
    report={}
    for role in ['train','checkpoint','ranker','validation','test']:
        rows=manifest(role,root);available=sum(cached_base(r,root)is not None for r in rows)
        report[role]={'expected_recordings':len(rows),'cached_header_waveform_pairs':available,'missing_pairs':len(rows)-available,'full_role_ready':available==len(rows)}
    return report


if __name__=='__main__':
    result=inventory();selected=json.loads((ROOT/'config/pilot_selection.json').read_text())
    x=load_records('train',[r['ecg_id']for r in selected])
    out={'role_inventory':result,'pilot_loader_shape':list(x.shape),'pilot_source':'eight previously declared training records','full_dataset_loaded':False,'test_waveforms_accessed':False}
    (ROOT/'outputs/cohort_loader_check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
