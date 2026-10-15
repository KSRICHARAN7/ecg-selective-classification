"""Acquire a fixed, unused external header sample; all clinical files stay private."""
import concurrent.futures,hashlib,json,sys,threading,time
from pathlib import Path
from collections import Counter
import requests,numpy as np
HERE=Path(__file__).resolve().parent
SOURCE=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE.parents[1]/'research'
sys.path.insert(0,str(SOURCE/'scripts'))
from external_ecg import header_metadata,physical_lead_II
from cohort_loader import load_records,manifest
plan=json.loads((HERE/'PLAN.json').read_text());selection=json.loads((HERE/'PRIVATE_selection.json').read_text());assert hashlib.sha256('\n'.join(selection['selected']).encode()).hexdigest()==plan['external']['selected_paths_sha256']
expected={}
for line in (SOURCE/'data/external/source/SHA256SUMS.txt').read_text().splitlines():
 h,n=line.split(maxsplit=1);expected[n.strip().lstrip('./*')]=h
out=HERE/'PRIVATE_external';out.mkdir(exist_ok=True);local=threading.local()
def fetch(rel):
 path=out/rel;digest=expected[rel]
 if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:return path
 if not hasattr(local,'session'):local.session=requests.Session()
 for attempt in range(3):
  try:
   r=local.session.get('https://physionet-open.s3.amazonaws.com/ecg-arrhythmia/1.0.0/'+rel,timeout=(10,30));r.raise_for_status()
   if hashlib.sha256(r.content).hexdigest()!=digest:raise ValueError('Publisher checksum mismatch')
   path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.partial');temp.write_bytes(r.content);temp.replace(path);return path
  except Exception:
   if attempt==2:raise
   time.sleep(1+attempt)
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:
 for i,path in enumerate(pool.map(fetch,selection['selected']),1):
  try: row=header_metadata(path)
  except ValueError as error:
   if 'does not match format' not in str(error) or '%d/%m/%Y' not in str(error): raise
   row={'record_id':path.stem,'eligible':False,'exclusion_reasons':['unparseable_provider_date'],'reference_AFIB':None}
  row['relative_header']=str(path.relative_to(out));rows.append(row)
  if i%100==0:print(f'New external headers verified {i}/1000',flush=True)
(HERE/'PRIVATE_headers.json').write_text(json.dumps(rows,indent=2)+'\n')
eligible=[r for r in rows if r['eligible']]
with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:
 for i,_ in enumerate(pool.map(fetch,[str(Path(r['relative_header']).with_suffix('.mat'))for r in eligible]),1):
  if i%100==0 or i==len(eligible):print(f'New external signals verified {i}/{len(eligible)}',flush=True)
x=np.stack([physical_lead_II(out/Path(r['relative_header']).with_suffix(''))[0]for r in eligible]);ids=np.array(['replication-external:'+r['record_id']for r in eligible]);y=np.array([r['reference_AFIB']for r in eligible]);hashes={hashlib.sha256(w.tobytes()).hexdigest()for w in x}
if len(hashes)!=len(x):raise ValueError('Within-sample duplicate: stop, no replacement')
overlap={}
old=[r for r in json.loads((SOURCE/'outputs/generalization/external_header_manifest.json').read_text())if r['eligible']]
old_hashes={hashlib.sha256(physical_lead_II(SOURCE/Path(r['source_header']).with_suffix(''))[0].tobytes()).hexdigest()for r in old}
overlap['original_external']=len(hashes&old_hashes)
for role in ['train','checkpoint','ranker','validation']:
 with np.load(SOURCE/f'data/derived/{role}_leadII.npz')as z:overlap[role]=len(hashes&{hashlib.sha256(w.tobytes()).hexdigest()for w in z['x']})
test=load_records('test',[r['ecg_id']for r in manifest('test')]).astype('float32');overlap['internal_test']=len(hashes&{hashlib.sha256(w.tobytes()).hexdigest()for w in test})
if any(overlap.values()):raise ValueError('Cross-sample exact waveform overlap: stop, no removal')
np.savez_compressed(HERE/'PRIVATE_external_arrays.npz',x=x,ids=ids,y=y)
summary={'sampled_headers':1000,'eligible_records':len(y),'AF_positive_records':int(y.sum()),'non_AF_records':int((1-y).sum()),'exclusion_reasons':dict(Counter(reason for r in rows for reason in r['exclusion_reasons'])),'all_provider_hashes_verified':True,'exact_waveform_overlap':overlap,'within_sample_duplicate_waveforms':0,'numerical_class_gate_passed':bool(min(y.sum(),(1-y).sum())>=30),'patient_linkage_unavailable':True,'bootstrap_unit':'record','selection_sha256':selection['sha256']}
(HERE/'external_acquisition_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
