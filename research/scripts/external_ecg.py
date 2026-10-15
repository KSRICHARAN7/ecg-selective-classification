"""Fixed external sample acquisition and header-driven physical lead-II decoding."""
import concurrent.futures,csv,hashlib,json,threading,time
from pathlib import Path
import numpy as np
import requests,wfdb
from scipy.io import loadmat
from scipy.signal import resample_poly
from cohort_loader import ROOT

RHYTHM_ACRONYMS=('SB','SR','AFIB','ST','AF','SA','SVT','AT','AVNRT','AVRT','SAAWR')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dictionary(root=ROOT):
 with (root/'data/external/source/ConditionNames_SNOMED-CT.csv').open(encoding='utf-8-sig')as stream:rows=list(csv.DictReader(stream))
 selected={r['Acronym Name']:str(r['Snomed_CT'])for r in rows if r['Acronym Name']in RHYTHM_ACRONYMS}
 if set(selected)!=set(RHYTHM_ACRONYMS)or selected['AFIB']!='164889003' or selected['AF']!='164890007':raise ValueError('Rhythm dictionary mismatch')
 return selected

def header_metadata(path,root=ROOT):
 h=wfdb.rdheader(str(Path(path).with_suffix('')));comments={s.split(':',1)[0].strip():s.split(':',1)[1].strip()for s in h.comments if ':'in s}
 try:age=float(comments.get('Age','nan'))
 except ValueError:age=float('nan')
 codes={x.strip()for x in comments.get('Dx','').split(',')};rhythms=dictionary(root);reasons=[]
 if not np.isfinite(age)or age<18:reasons.append('age_missing_or_under18')
 if not codes&set(rhythms.values()):reasons.append('no_explicit_eligible_rhythm_code')
 if h.fs!=500 or h.sig_len!=5000 or h.n_sig!=12 or h.sig_name.count('II')!=1:reasons.append('incompatible_signal_layout')
 return dict(record_id=Path(path).stem,age=age if np.isfinite(age)else None,sex=comments.get('Sex'),codes=sorted(codes),rhythm_acronyms=sorted(k for k,v in rhythms.items()if v in codes),reference_AFIB=int(rhythms['AFIB']in codes),eligible=not reasons,exclusion_reasons=reasons)

def physical_lead_II(base):
 base=Path(base);h=wfdb.rdheader(str(base));raw=loadmat(base.with_suffix('.mat'))['val']
 if h.fs!=500 or h.sig_len!=5000 or h.n_sig!=12 or raw.shape!=(12,5000) or h.sig_name.count('II')!=1:raise ValueError('Unexpected external format')
 i=h.sig_name.index('II');unit=h.units[i];gain=h.adc_gain[i];baseline=h.baseline[i]
 if unit not in ['mV','uV','µV']or not np.isfinite(gain)or gain<=0:raise ValueError('Unrecognized physical calibration')
 if not np.isfinite(raw).all()or np.any(raw[i]==-32768):raise ValueError('Invalid external waveform samples')
 # Keep provider SHA checks authoritative; check WFDB checksum as an additional audit.
 checks=np.asarray(h.checksum,dtype=np.int64)%65536
 if not np.array_equal(raw.astype(np.int64).sum(axis=1)%65536,checks):raise ValueError('Native signal/header checksum mismatch')
 x=(raw[i].astype(float)-baseline)/gain
 if unit in ['uV','µV']:x=x/1000
 y=resample_poly(x,1,5,window=('kaiser',5.),padtype='constant')
 if y.shape!=(1000,)or not np.isfinite(y).all()or y.std()<=1e-12:raise ValueError('Invalid resampled external leadII; no omission')
 return y.astype('float32'),dict(native_rate=500,target_rate=100,native_unit=unit,adc_gain=float(gain),baseline=float(baseline),lead='II',resampling='resample_poly1/5 Kaiser5 constant padding',native_checksum_verified=True)

def main():
 spec=json.loads((ROOT/'config/step10_specification.json').read_text());expected={}
 for line in (ROOT/'data/external/source/SHA256SUMS.txt').read_text().splitlines():
  h,p=line.split(maxsplit=1);expected[p.strip().lstrip('./*')]=h
 for name in ['ConditionNames_SNOMED-CT.csv','LICENSE.txt']:
  if digest(ROOT/'data/external/source'/name)!=expected[name]:raise ValueError('External metadata hash mismatch')
 all_headers=sorted(p for p in expected if p.endswith('.hea')and p.startswith('WFDBRecords/'))
 if len(all_headers)!=45152:raise ValueError('External inventory size differs')
 selected=sorted(all_headers,key=lambda p:hashlib.sha256(f"{spec['external']['sample_seed']}|{p}".encode()).digest())[:spec['external']['sample_records']]
 (ROOT/'outputs/step10_external_selection.json').write_text(json.dumps({'sample_before_headers':True,'eligible_header_universe':len(all_headers),'seed':spec['external']['sample_seed'],'selected':selected,'selection_sha256':hashlib.sha256('\n'.join(selected).encode()).hexdigest()},indent=2)+'\n')
 local=threading.local()
 def fetch(rel):
  path=ROOT/'data/external/records'/rel
  if not path.exists()or digest(path)!=expected[rel]:
   if not hasattr(local,'session'):local.session=requests.Session()
   for attempt in range(6):
    try:
     r=local.session.get('https://physionet-open.s3.amazonaws.com/ecg-arrhythmia/1.0.0/'+rel,timeout=(15,45));r.raise_for_status();data=r.content
     if hashlib.sha256(data).hexdigest()!=expected[rel]:raise ValueError('External provider hash mismatch')
     path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.part');temp.write_bytes(data);temp.replace(path);break
    except Exception:
     if attempt==5:raise
     time.sleep(min(16,2**attempt))
  return {'path':path.relative_to(ROOT).as_posix(),'provider_sha256':expected[rel],'bytes':path.stat().st_size,'provider_verified':True}
 audit=[];metadata=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:
  for i,r in enumerate(pool.map(fetch,selected),1):
   audit.append(r);m=header_metadata(ROOT/r['path']);m['source_header']=r['path'];metadata.append(m)
   if i%200==0:print(f'External headers verified {i}/{len(selected)}',flush=True)
 eligible=[m for m in metadata if m['eligible']];data_paths=[str(Path(m['source_header']).relative_to('data/external/records').with_suffix('.mat'))for m in eligible]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:
  for i,r in enumerate(pool.map(fetch,data_paths),1):
   audit.append(r)
   if i%200==0 or i==len(data_paths):print(f'External signals verified {i}/{len(data_paths)}',flush=True)
 out=ROOT/'outputs/generalization';out.mkdir(exist_ok=True)
 (out/'external_header_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n');(out/'external_acquisition.json').write_text(json.dumps(audit,indent=2)+'\n')
 summary=dict(selected_records=len(metadata),eligible_records=len(eligible),excluded_records=len(metadata)-len(eligible),eligible_AFIB=sum(m['reference_AFIB']for m in eligible),all_provider_sha256_verified=True,selection_is_label_independent=True,source_inventory_sha256=digest(ROOT/'data/external/source/SHA256SUMS.txt'))
 (out/'external_acquisition_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
