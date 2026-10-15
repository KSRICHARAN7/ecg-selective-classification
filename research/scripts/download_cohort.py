"""Resumable role-specific source download. Default produces a plan only."""
import argparse,concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
from cohort_loader import ROOT,manifest,cached_base,require_protocol_lock


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--role',required=True,choices=['train','checkpoint','ranker','validation','test'])
    parser.add_argument('--fetch',action='store_true',help='Without this flag, write a plan without network access')
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if not 1<=args.workers<=8:raise ValueError('Use between1 and8 download workers')
    rows=manifest(args.role)
    plan={'role':args.role,'expected_records':len(rows),'cached_pairs':sum(cached_base(r)is not None for r in rows),'network_requested':args.fetch,'estimated_dat_bytes_uncompressed':len(rows)*1000*12*2,'estimate_excludes':'headers and HTTP overhead; assumes verified pilot format16'}
    (ROOT/'outputs'/('download_plan_'+args.role+'.json')).write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2),flush=True)
    if not args.fetch:return
    if args.role=='test':require_protocol_lock()
    log=ROOT/'outputs'/('download_log_'+args.role+'.jsonl')
    def download(row):
        if cached_base(row)is not None:return {'ecg_id':row['ecg_id'],'status':'cached_pair'}
        files=[]
        try:
            for suffix in ['.hea','.dat']:
                rel=row['filename_lr']+suffix;path=ROOT/'data/cohort'/rel
                path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():
                    url='https://physionet.org/files/ptb-xl/1.0.3/'+rel
                    with urllib.request.urlopen(url,timeout=45)as r:data=r.read()
                    if not data or data[:20].lower().startswith(b'<!doctype html'):raise ValueError('Empty or HTML source response')
                    temp=path.with_suffix(suffix+'.part');temp.write_bytes(data);temp.replace(path)
                data=path.read_bytes();files.append({'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
            return {'ecg_id':row['ecg_id'],'status':'downloaded','files':files}
        except Exception as e:return {'ecg_id':row['ecg_id'],'status':'failed','error':type(e).__name__+': '+str(e)}
    failures=0
    with log.open('a')as stream,concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)as pool:
        for n,result in enumerate(pool.map(download,rows),1):
            stream.write(json.dumps(result)+'\n');stream.flush()
            failures+=result['status']=='failed'
            if n%100==0:print(f'Completed{n}/{len(rows)}; failures{failures}',flush=True)
    if failures:raise RuntimeError(f'{failures} recordings failed; rerun to resume, do not train on a silent subset')


if __name__=='__main__':main()
