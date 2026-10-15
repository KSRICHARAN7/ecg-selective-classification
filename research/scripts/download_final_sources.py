"""Post-lock acquisition, exact fold 10 inventory and provider SHA256 checks."""
import concurrent.futures,hashlib,json,threading,time,datetime
import requests
from cohort_loader import ROOT,manifest,require_protocol_lock

def main():
 require_protocol_lock();lock=json.loads((ROOT/'config/protocol_lock.json').read_text())
 expected={}
 for line in (ROOT/'data/source/PTBXL_SHA256SUMS.txt').read_text().splitlines():
  h,p=line.split(maxsplit=1);expected[p.strip().lstrip('./*')]=h
 local=threading.local()
 def session():
  if not hasattr(local,'session'):local.session=requests.Session()
  return local.session
 def retrieve(url,path,digest=None):
  if not path.exists() or (digest and hashlib.sha256(path.read_bytes()).hexdigest()!=digest):
   for attempt in range(6):
    try:
     r=session().get(url,timeout=(15,45));r.raise_for_status();data=r.content
     if not data or data.startswith(b'<!DOCTYPE'):raise ValueError('Invalid response')
     if digest and hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Provider hash mismatch')
     path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.part');tmp.write_bytes(data);tmp.replace(path);break
    except Exception:
     if attempt==5:raise
     time.sleep(min(16,2**attempt))
  return dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,url=url,provider_sha256_verified=digest is not None)
 def fetch(row):
  files=[]
  for ext in ['.hea','.dat']:
   rel=row['filename_lr']+ext;files.append(retrieve('https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/'+rel,ROOT/'data/cohort'/rel,expected[rel]))
  return dict(ecg_id=row['ecg_id'],role='test',files=files)
 rows=manifest('test');results=[]
 with (ROOT/'outputs/step9_acquisition.jsonl').open('w') as stream,concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for i,r in enumerate(pool.map(fetch,rows),1):
   results.append(r);stream.write(json.dumps(r)+'\n');stream.flush()
   if i%100==0 or i==len(rows):print(f'Final-test pairs verified {i}/{len(rows)}',flush=True)
 # Noise publisher inventory is downloaded after lock, before any em decoder.
 checksum=ROOT/'data/source/NSTDB_SHA256SUMS.txt';retrieve('https://physionet.org/files/nstdb/1.0.0/SHA256SUMS.txt',checksum)
 noise_expected={}
 for line in checksum.read_text().splitlines():
  h,p=line.split(maxsplit=1);noise_expected[p.strip().lstrip('./*')]=h
 noise=[]
 for family in ['bw','ma','em']:
  for ext in ['.hea','.dat']:
   rel=family+ext;noise.append(retrieve('https://physionet.org/files/nstdb/1.0.0/'+rel,ROOT/'data/pilot/noise'/rel,noise_expected[rel]))
 summary=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_locked_utc=lock['recorded_utc'],recordings=len(results),source_files=2*len(results),all_provider_sha256_verified=True,noise_files=noise)
 (ROOT/'outputs/step9_acquisition_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('All final sources verified',flush=True)
if __name__=='__main__':main()
