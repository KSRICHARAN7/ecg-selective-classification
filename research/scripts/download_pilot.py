"""Download only the declared eight training-fold records and bw/ma noise.

The withheld em waveform is intentionally not requested. Network failures are
recorded, never replaced by synthetic ECGs.
"""
import concurrent.futures
import csv
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    selection=json.loads((ROOT/'config/pilot_selection.json').read_text())
    cohort={r['ecg_id']:r for r in csv.DictReader((ROOT/'outputs/development_cohort_manifest.csv').open())}
    jobs=[]
    for row in selection:
        actual=cohort[row['ecg_id']]
        if int(actual['strat_fold'])>6 or row['filename_lr']!=actual['filename_lr']:
            raise ValueError('Pilot must use verified classifier-training records only')
        for suffix in ['.hea','.dat']:
            rel=row['filename_lr']+suffix
            jobs.append(('https://physionet.org/files/ptb-xl/1.0.3/'+rel,ROOT/'data/pilot'/rel))
    for name in ['bw','ma']:
        for suffix in ['.hea','.dat']:
            jobs.append(('https://physionet.org/files/nstdb/1.0.0/'+name+suffix,ROOT/'data/pilot/noise'/(name+suffix)))
    def fetch(job):
        url,path=job
        try:
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                with urllib.request.urlopen(url,timeout=45) as response:
                    data=response.read()
                if not data or data[:20].lower().startswith(b'<!doctype html'):
                    raise ValueError('Empty or HTML response')
                temp=path.with_suffix(path.suffix+'.part');temp.write_bytes(data);temp.replace(path)
            data=path.read_bytes()
            return {'url':url,'path':path.relative_to(ROOT).as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'status':'downloaded'}
        except Exception as error:
            return {'url':url,'path':path.relative_to(ROOT).as_posix(),'status':'failed','error':type(error).__name__+': '+str(error)}
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(fetch,jobs):
            results.append(result)
            print(result['status'],result['path'],flush=True)
            (ROOT/'outputs/pilot_download_manifest.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(r['status']!='downloaded' for r in results):
        raise RuntimeError('Pilot download incomplete; inspect manifest')


if __name__=='__main__':main()
