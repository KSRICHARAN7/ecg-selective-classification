"""Build patient-separated role manifests without final-test label summaries."""
import csv,hashlib,json
from collections import Counter
from pathlib import Path,PurePosixPath
from audit_metadata import classify,read_csv

ROOT=Path(__file__).resolve().parents[1]
ROLE_FOLDS={'train':set(range(1,7)),'checkpoint':{7},'ranker':{8},'validation':{9},'test':{10}}


def check_integrity(rows):
    ids=set();patients={};paths=set()
    for row in rows:
        if not row['patient_id'] or row['ecg_id'] in ids:raise ValueError('Missing patient or duplicate ECG ID')
        ids.add(row['ecg_id']);fold=int(row['strat_fold'])
        if fold not in range(1,11):raise ValueError('Invalid official fold')
        if row['patient_id'] in patients and patients[row['patient_id']]!=fold:
            raise ValueError('Participant crosses official folds')
        patients[row['patient_id']]=fold
        path=PurePosixPath(row['filename_lr'])
        if path.is_absolute() or '..' in path.parts or not str(path).startswith('records100/'):
            raise ValueError('Unexpected waveform source path')
        if str(path) in paths:raise ValueError('Waveform path shared by multiple ECG IDs')
        paths.add(str(path))
    return patients


def main():
    config=json.loads((ROOT/'config/task_definition.json').read_text())
    for name,digest in config['source_sha256'].items():
        if hashlib.sha256((ROOT/'data/source'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source changed')
    rows=read_csv(ROOT/'data/source/ptbxl_database.csv');check_integrity(rows)
    dictionary=read_csv(ROOT/'data/source/scp_statements.csv')
    rhythm={r[''] for r in dictionary if r['rhythm']=='1.0'}
    roles={role:[] for role in ROLE_FOLDS};excluded=Counter()
    for row in rows:
        role=next(role for role,folds in ROLE_FOLDS.items() if int(row['strat_fold']) in folds)
        reason,label,codes=classify(row,rhythm)
        if reason!='eligible':excluded[(role,reason)]+=1;continue
        # Eligibility necessarily reads reference statements. Held-out target
        # values are intentionally omitted from exported role manifests.
        roles[role].append({'ecg_id':row['ecg_id'],'patient_id':row['patient_id'],'official_fold':row['strat_fold'],'role':role,'filename_lr':row['filename_lr'],'target_afib_annotation':label if int(row['strat_fold'])<=8 else ''})
    folder=ROOT/'outputs/partitions';folder.mkdir(exist_ok=True)
    groups={role:{r['patient_id'] for r in items} for role,items in roles.items()}
    overlap={a:{b:len(groups[a]&groups[b]) if a!=b else len(groups[a]) for b in roles}for a in roles}
    if any(overlap[a][b] for a in roles for b in roles if a!=b):raise ValueError('Participant role leakage')
    summary={};digests={}
    for role,items in roles.items():
        path=folder/(role+'.csv')
        with path.open('w',newline='')as stream:
            writer=csv.DictWriter(stream,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
        digests[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        summary[role]={'recordings':len(items),'participants':len(groups[role]),'folds':sorted(ROLE_FOLDS[role]),'target_exported':role not in ['validation','test']}
        if role not in ['validation','test']:summary[role]['positive_recordings']=sum(int(r['target_afib_annotation']) for r in items)
    report={'status':'metadata partitions verified; full waveform audit not yet complete','role_counts':summary,'participant_overlap_matrix':overlap,'eligible_recordings':sum(len(r)for r in roles.values()),'eligible_participants':len(set.union(*groups.values())),'sequential_exclusion_counts':{role:{reason:n for (r,reason),n in excluded.items()if r==role}for role in roles},'role_manifest_sha256':digests,'heldout_note':'Reference statements are read for eligibility; validation/test labels are not exported or summarized. No performance evaluation has occurred.'}
    (ROOT/'outputs/partition_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
