"""Acquire only ranker and policy-validation roles; verify provider hashes."""
import concurrent.futures,hashlib,json,threading,time
from pathlib import Path
import requests
from cohort_loader import ROOT,manifest


def main():
    expected={}
    for line in (ROOT/'data/source/PTBXL_SHA256SUMS.txt').read_text().splitlines():
        digest,name=line.split(maxsplit=1);expected[name.strip().lstrip('./*')]=digest
    tasks=[(role,row)for role in ['ranker','validation']for row in manifest(role)]
    local=threading.local()
    def fetch(task):
        role,row=task;session=getattr(local,'session',None)
        if session is None:local.session=requests.Session();session=local.session
        files=[]
        for suffix in ['.hea','.dat']:
            rel=row['filename_lr']+suffix;path=ROOT/'data/cohort'/rel;digest=expected[rel]
            if not path.exists()or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                for attempt in range(6):
                    try:
                        url='https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/'+rel
                        response=session.get(url,timeout=(15,45));response.raise_for_status();data=response.content
                        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Provider SHA256 mismatch: '+rel)
                        path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(suffix+'.part');temporary.write_bytes(data);temporary.replace(path);break
                    except Exception:
                        if attempt==5:raise
                        time.sleep(min(20,2**attempt))
            files.append({'path':rel,'sha256':digest,'bytes':path.stat().st_size})
        return {'role':role,'ecg_id':row['ecg_id'],'files':files,'provider_sha256_verified':True}
    print(f'Fetching {len(tasks)} ranker/policy-validation ECGs; no final-test source requests',flush=True)
    with (ROOT/'outputs/step8_acquisition.jsonl').open('w')as stream,concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:
        for n,result in enumerate(pool.map(fetch,tasks),1):
            stream.write(json.dumps(result)+'\n');stream.flush()
            if n%100==0 or n==len(tasks):print(f'Verified {n}/{len(tasks)} ECG pairs',flush=True)
    (ROOT/'outputs/step8_acquisition_summary.json').write_text(json.dumps({'roles':['ranker','validation'],'recordings':len(tasks),'source_files':2*len(tasks),'all_provider_sha256_verified':True,'final_test_acquired':False,'source_base':'https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/','resume_note':'Initial files acquired from official PhysioNet HTTP source; verified cached files reused when switching to the documented official public AWS mirror'},indent=2)+'\n')


if __name__=='__main__':main()
