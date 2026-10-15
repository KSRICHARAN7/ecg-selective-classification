"""Step 9 locked independent test. No fitting or threshold selection."""
import ast,csv,datetime,hashlib,json,platform,resource,time
from collections import Counter
import numpy as np
import torch,wfdb
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from cohort_loader import ROOT,manifest,load_records,require_protocol_lock
from signal_tools import resample_noise,add_noise,quality_features
from run_reliability_development import predict
from model_definition import ECGClassifier
from reliability_methods import raw_confidence,score_saved
from test_evaluation import METHODS,COVERAGES,selection_weights,metrics,bootstrap_primary

class FinalNoise:
 def __init__(self):
  require_protocol_lock();self.bank={};self.audit=[]
  acquisition=json.loads((ROOT/'outputs/step9_acquisition_summary.json').read_text());expected={x['path']:x['sha256']for x in acquisition['noise_files']}
  for family in ['bw','ma','em']:
   base=ROOT/'data/pilot/noise'/family
   for ext in ['.dat','.hea']:
    p=base.with_suffix(ext)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected[p.relative_to(ROOT).as_posix()]:raise ValueError('Changed noise')
   h=wfdb.rdheader(str(base))
   if h.fs!=360 or h.n_sig!=2 or h.sig_len<1780*360:raise ValueError('Unexpected noise layout')
   source=wfdb.rdrecord(str(base),sampfrom=1200*360,sampto=1780*360,physical=True).p_signal
   if source.shape!=(580*360,2) or not np.isfinite(source).all():raise ValueError('Invalid final noise')
   bank=np.empty((2,58,1000))
   for channel in range(2):
    for slot in range(58):
     x=resample_noise(source[slot*3600:(slot+1)*3600,channel],360)
     if x.shape!=(1000,) or not np.isfinite(x).all() or x.std()<=1e-12:raise ValueError('Invalid final noise segment; stop without skipping')
     bank[channel,slot]=x;self.audit.append(dict(family=family,channel=channel,slot=slot,start_second=1200+10*slot,sha256=hashlib.sha256(x.tobytes()).hexdigest()))
   self.bank[family]=bank
 def pair(self,x,identifier,family,snr):
  digest=hashlib.sha256(f'20261015|{identifier}|{family}'.encode()).digest();channel=int.from_bytes(digest[:8],'big')%2;slot=int.from_bytes(digest[8:16],'big')%58
  corrupt,injected=add_noise(x,self.bank[family][channel,slot],snr);clean=np.asarray(x,dtype=float);clean=clean-clean.mean()
  actual=20*np.log10(np.sqrt(np.mean(clean**2))/np.sqrt(np.mean(injected**2)))
  if not np.isfinite(corrupt).all() or abs(actual-snr)>1e-10:raise ValueError('Corruption numerical failure')
  return corrupt,dict(ecg_id=str(identifier),family=family,channel=channel,slot=slot,start_second=1200+10*slot,snr_db=snr,measured_snr_db=float(actual),wave_sha256=hashlib.sha256(corrupt.tobytes()).hexdigest())

def validate_locks():
 require_protocol_lock();lock=json.loads((ROOT/'config/protocol_lock.json').read_text())
 for name,h in lock['files'].items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('Frozen file changed: '+name)
 return lock

def score_metrics(y,p,edges):
 return dict(AUROC=float(roc_auc_score(y,p)) if len(set(y))==2 else None,AUPRC=float(average_precision_score(y,p)) if y.sum() else None,Brier=float(brier_score_loss(y,p)),calibration=[dict(low=float(a),high=float(b),N=int(m.sum()),mean_probability=float(p[m].mean()) if m.any() else None,observed_fraction=float(y[m].mean()) if m.any() else None)for a,b in zip(edges[:-1],edges[1:])for m in [(p>=a)&((p<b) if b<1 else (p<=b))]],incorrect_confidence_mean=None)

def main():
 begin=time.perf_counter();lock=validate_locks();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 out=ROOT/'outputs/final_experiment';out.mkdir(exist_ok=True)
 if (out/'primary_result.json').exists():raise FileExistsError('Final result exists; explicitly choose a separate audit copy to rerun')
 acquisition=json.loads((ROOT/'outputs/step9_acquisition_summary.json').read_text())
 rows=manifest('test')
 if acquisition['recordings']!=len(rows) or not acquisition['all_provider_sha256_verified']:raise ValueError('Incomplete acquisition')
 ids=np.array([r['ecg_id']for r in rows]);people=np.array([r['patient_id']for r in rows]);chunks=[]
 for i in range(0,len(rows),256):chunks.append(load_records('test',ids[i:i+256]).astype('float32'))
 x=np.concatenate(chunks)
 if not np.isfinite(x).all() or (x.std(axis=1)<=1e-12).any():raise ValueError('Invalid/zero-power test; no silent dropping')
 test_hash={hashlib.sha256(w.tobytes()).hexdigest()for w in x};overlap={}
 for role in ['train','checkpoint','ranker','validation']:
  with np.load(ROOT/f'data/derived/{role}_leadII.npz',allow_pickle=False)as z:overlap[role]=len(test_hash&{hashlib.sha256(w.tobytes()).hexdigest()for w in z['x']})
 if any(overlap.values()):raise ValueError('Cross-role identical waveform; stop before test inference')
 # Reference outcome access begins only after recorded protocol lock and QC.
 metadata={r['ecg_id']:r for r in csv.DictReader((ROOT/'data/source/ptbxl_database.csv').open())};y=np.array([int('AFIB' in ast.literal_eval(metadata[i]['scp_codes']))for i in ids])
 checkpoint=torch.load(ROOT/'models/cnn_seed20261015.pt',map_location='cpu',weights_only=True);model=ECGClassifier();model.load_state_dict(checkpoint['state_dict']);model.eval();scale=checkpoint['training_scale_mV'];threshold=json.loads((ROOT/'config/step8_specification.json').read_text())['classification_threshold']
 policies=json.loads((ROOT/'outputs/reliability_development/policy_thresholds.json').read_text());noise=FinalNoise();conditions={};all_rows=[];fixed_rows=[];classification={};assignments=[];timings=[];latencies=[]
 configurations=[('clean',None,None)]+[(f'{family}_{snr}',family,snr)for family in ['bw','ma','em']for snr in [24,12,6,0]]
 for name,family,snr in configurations:
  print('Evaluating locked condition '+name,flush=True);t=time.perf_counter()
  if family:
   pairs=[noise.pair(w,i,family,snr)for w,i in zip(x,ids)];waves=np.stack([v[0]for v in pairs]);assignments.extend([v[1]for v in pairs])
  else:waves=x.astype(float)-x.mean(axis=1,keepdims=True,dtype=float)
  gen=time.perf_counter()-t;t=time.perf_counter();q=np.stack([quality_features(w)for w in waves]);qt=time.perf_counter()-t
  t=time.perf_counter();p=predict(model,scale,waves.astype('float32'));ct=time.perf_counter()-t;pred=(p>=threshold).astype(int)
  t=time.perf_counter();scores={'raw_confidence':raw_confidence(p),**{m:score_saved(ROOT/f'models/reliability/{m}.json',p,q)for m in METHODS if m!='raw_confidence'}};st=time.perf_counter()-t
  d=dict(ids=ids,participant=people,y=y,pred=pred,p=p,quality=q,**scores);conditions[name]=d
  temporary=out/(name+'.pending.npz');np.savez_compressed(temporary,**d)
  with np.load(temporary,allow_pickle=False)as z:
   for key,v in d.items():
    if not np.array_equal(z[key],v,equal_nan=v.dtype.kind in 'fc'):raise ValueError('Saved test features differ')
  temporary.replace(out/(name+'.npz'))
  for method in METHODS:
   for coverage in COVERAGES:all_rows.append(dict(condition=name,method=method,target_coverage=coverage,**metrics(y,pred,selection_weights(scores[method],ids,coverage))))
   fixed_rows.append(dict(condition=name,method=method,threshold=policies[method]['threshold'],**metrics(y,pred,(scores[method]<=policies[method]['threshold']).astype(int))))
  cm=score_metrics(y,p,np.linspace(0,1,11));error=pred!=y;cm['incorrect_confidence_mean']=float(np.maximum(p,1-p)[error].mean()) if error.any() else None;cm['all_record_metrics']=metrics(y,pred,np.ones(len(y),dtype=int));classification[name]=cm
  timings.append(dict(condition=name,N=len(y),noise_generation_seconds=gen,quality_seconds=qt,cnn_seconds=ct,all_four_rankers_seconds=st,batch_size=128,torch_threads=2))
 # Fixed latency benchmark: first 100 ECG IDs in manifest order, clean; warm-up10; no selection by outcomes.
 waves=x[:100];predict(model,scale,waves.astype('float32'))
 for w in waves:
  t=time.perf_counter();pp=predict(model,scale,w[None,:].astype('float32'));cnn_ms=(time.perf_counter()-t)*1000;t=time.perf_counter();qq=quality_features(w)[None,:];quality_ms=(time.perf_counter()-t)*1000;t=time.perf_counter();score_saved(ROOT/'models/reliability/combined.json',pp,qq);rank_ms=(time.perf_counter()-t)*1000
  latencies.append([cnn_ms,quality_ms,rank_ms])
 primary_conditions=[conditions[f'em_{snr}']for snr in [24,12,6,0]];inference,effects,snr_effects=bootstrap_primary(primary_conditions)
 values={}
 for snr in [24,12,6,0]:values[str(snr)]={m:next(r for r in all_rows if r['condition']==f'em_{snr}'and r['method']==m and r['target_coverage']==.9)for m in METHODS}
 diffs=[values[str(s)]['combined']['balanced_selective_error']-values[str(s)]['confidence_adjusted']['balanced_selective_error']for s in [24,12,6,0]]
 effect=float(np.mean(diffs));guardrail=all(values[str(s)]['combined']['positive_retention']>=.8 for s in [24,12,6,0]);ci=inference['percentile95_CI'];superior=bool(ci and ci[1]<0 and guardrail)
 result=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_locked_utc=lock['recorded_utc'],N=len(y),participants=len(set(people)),positive_records=int(y.sum()),positive_participants=len(set(people[y==1])),negative_records=int((1-y).sum()),effect_combined_minus_confidence_adjusted=effect,per_snr_differences=diffs,inference=inference,AF_retention_guardrail_passed=guardrail,qualified_superiority_supported=superior,primary_conditions=values,interpretation='qualified superiority supported in this controlled benchmark' if superior else 'qualified superiority not supported; report effect, interval and AF retention separately',cross_role_exact_hash_overlap=overlap,missing_quality_values={k:np.isnan(v['quality']).sum(axis=0).tolist()for k,v in conditions.items()},scope='one fixed classifier and noise library; no external or clinical validation',elapsed_seconds=time.perf_counter()-begin)
 np.savez_compressed(out/'primary_bootstrap.npz',effects=effects,per_snr_effects=snr_effects)
 for name,data in [('matched_coverage_metrics',all_rows),('fixed_policy_metrics',fixed_rows)]:
  with (out/(name+'.csv')).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 (out/'classification_metrics.json').write_text(json.dumps(classification,indent=2)+'\n');(out/'noise_assignments.json').write_text(json.dumps(assignments)+'\n');(out/'noise_segment_audit.json').write_text(json.dumps(noise.audit,indent=2)+'\n')
 lat=np.asarray(latencies);hardware=dict(platform=platform.platform(),processor=platform.processor(),cpu_model=next((s.strip()for s in open('/proc/cpuinfo')if s.startswith('model name')),'unknown'),torch_version=torch.__version__,torch_threads=2,condition_timings=timings,single_record_samples=100,single_record_warmup='one 100-record batch',single_record_columns=['cnn_ms','quality_ms','combined_JSON_load_and_score_ms'],single_record_median_ms=np.median(lat,axis=0).tolist(),single_record_p95_ms=np.percentile(lat,95,axis=0).tolist(),process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,scope='host CPU benchmark including Python overhead; process peak memory includes data and analysis, not isolated deployment memory')
 (out/'compute.json').write_text(json.dumps(hardware,indent=2)+'\n');(out/'primary_result.json').write_text(json.dumps(result,indent=2)+'\n');validate_locks();print(json.dumps({k:v for k,v in result.items()if k not in ['primary_conditions','missing_quality_values']},indent=2),flush=True)
if __name__=='__main__':main()
