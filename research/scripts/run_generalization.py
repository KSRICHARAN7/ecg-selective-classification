"""Step10 exploratory fixed-model severity and sampled-external transfer tests."""
import hashlib,json,csv,time,datetime
from pathlib import Path
import numpy as np
import torch
from cohort_loader import ROOT,manifest,load_records
from model_definition import ECGClassifier
from run_reliability_development import predict
from run_final_experiment import FinalNoise,validate_locks,score_metrics
from external_ecg import physical_lead_II
from signal_tools import quality_features
from reliability_methods import raw_confidence,score_saved
from test_evaluation import METHODS,COVERAGES,selection_weights,metrics,bootstrap_primary

def check_step10():
 validate_locks();spec=json.loads((ROOT/'config/step10_specification.json').read_text())
 for name,h in spec['files'].items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('Step10 immutable input changed: '+name)
 return spec

def load_external():
 out=ROOT/'outputs/generalization';a=json.loads((out/'external_acquisition_summary.json').read_text());rows=[m for m in json.loads((out/'external_header_manifest.json').read_text())if m['eligible']]
 if len(rows)!=a['eligible_records']or not a['all_provider_sha256_verified']:raise ValueError('Incomplete external acquisition')
 expected={m['path']:m['provider_sha256']for m in json.loads((out/'external_acquisition.json').read_text())};waves=[];audit=[]
 for m in rows:
  base=ROOT/Path(m['source_header']).with_suffix('')
  for ext in ['.hea','.mat']:
   p=base.with_suffix(ext)
   if hashlib.sha256(p.read_bytes()).hexdigest()!=expected[p.relative_to(ROOT).as_posix()]:raise ValueError('External source changed')
  wave,meta=physical_lead_II(base);waves.append(wave);audit.append(dict(record_id=m['record_id'],**meta))
 x=np.stack(waves);ids=np.array(['external:'+m['record_id']for m in rows]);y=np.array([m['reference_AFIB']for m in rows]);hashes={hashlib.sha256(w.tobytes()).hexdigest()for w in x};overlaps={}
 for role in ['train','checkpoint','ranker','validation']:
  with np.load(ROOT/f'data/derived/{role}_leadII.npz')as z:overlaps[role]=len(hashes&{hashlib.sha256(w.tobytes()).hexdigest()for w in z['x']})
 with np.load(ROOT/'outputs/final_experiment/clean.npz')as z:
  # Original test's prediction array does not store waveforms. Reload only fixed test role.
  test_x=load_records('test',[r['ecg_id']for r in manifest('test')]).astype('float32');overlaps['test']=len(hashes&{hashlib.sha256(w.tobytes()).hexdigest()for w in test_x})
 if any(overlaps.values()):raise ValueError('Exact external/development waveform overlap: stop without outcome-based removal')
 (out/'external_signal_audit.json').write_text(json.dumps({'records':len(rows),'physical_layouts':audit,'exact_resampled_cross_cohort_overlap':overlaps,'duplicate_external_resampled_hashes':len(x)-len(hashes),'independent_patient_ID_unavailable':True},indent=2)+'\n')
 if len(x)!=len(hashes):raise ValueError('Duplicate external waveform: protocol amendment required; no silent removal')
 return x,ids,y

def evaluate(waves,ids,people,y,model,scale,threshold):
 q=np.stack([quality_features(w)for w in waves]);p=predict(model,scale,waves.astype('float32'));pred=(p>=threshold).astype(int);scores={'raw_confidence':raw_confidence(p),**{name:score_saved(ROOT/f'models/reliability/{name}.json',p,q)for name in METHODS if name!='raw_confidence'}}
 return dict(ids=ids,participant=people,y=y,pred=pred,p=p,quality=q,**scores)

def write_array(path,data):
 temp=path.with_name(path.stem+'.pending.npz');np.savez_compressed(temp,**data)
 with np.load(temp,allow_pickle=False)as z:
  for k,v in data.items():
   if not np.array_equal(z[k],v,equal_nan=v.dtype.kind in 'fc'):raise ValueError('Step10 saved array differs')
 temp.replace(path)

def describe(name,ds,seed=20261018):
 inference,effects,per=bootstrap_primary(ds,2000,seed)
 differences=[];retentions=[]
 for d in ds:
  m={k:metrics(d['y'],d['pred'],selection_weights(d[k],d['ids'],.9))for k in ['combined','confidence_adjusted']};a=m['combined']['balanced_selective_error'];b=m['confidence_adjusted']['balanced_selective_error'];differences.append(a-b if a is not None and b is not None else None);retentions.append(m['combined']['positive_retention'])
 effect=float(np.mean(differences))if None not in differences else None
 summary=dict(effect_combined_minus_confidence_adjusted=effect,condition_effects=differences,inference=inference,combined_AF_retention=retentions,AF_retention_guardrail_passed=all(v is not None and v>=.8 for v in retentions),interpretation='exploratory transfer summary; not a new confirmatory superiority test')
 write_array(ROOT/f'outputs/generalization/{name}_bootstrap.npz',{'effects':effects,'per_condition':per});return summary

def main():
 start=time.perf_counter();spec=check_step10();o=ROOT/'outputs/generalization';o.mkdir(exist_ok=True)
 if (o/'generalization_summary.json').exists():raise FileExistsError('Completed Step10 outputs exist; use a separate audit copy')
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);ck=torch.load(ROOT/'models/cnn_seed20261015.pt',map_location='cpu',weights_only=True);model=ECGClassifier();model.load_state_dict(ck['state_dict']);model.eval();scale=ck['training_scale_mV'];threshold=json.loads((ROOT/'config/step8_specification.json').read_text())['classification_threshold'];noise=FinalNoise()
 with np.load(ROOT/'outputs/final_experiment/clean.npz',allow_pickle=False)as z:test={k:z[k]for k in z.files}
 rows=manifest('test');test_x=load_records('test',[r['ecg_id']for r in rows]).astype('float32')
 if not np.array_equal(test['ids'],[r['ecg_id']for r in rows]):raise ValueError('Test source order changed')
 external_x,external_ids,external_y=load_external();conditions={};assignments=[];numerical_max=0.;tasks=[]
 for family in ['bw','ma','em']:
  for snr in spec['internal']['new_severities_db']:tasks.append((f'internal_{family}_{snr}',test_x,test['ids'],test['participant'],test['y'],family,snr))
 tasks.append(('external_clean',external_x,external_ids,external_ids,external_y,None,None))
 for snr in [24,12,6,0]:tasks.append((f'external_em_{snr}',external_x,external_ids,external_ids,external_y,'em',snr))
 metric_rows=[];fixed_rows=[];classification={};policies=json.loads((ROOT/'outputs/reliability_development/policy_thresholds.json').read_text())
 for name,x,ids,people,y,family,snr in tasks:
  print('Evaluating fixed followup '+name,flush=True)
  if family:
   pairs=[noise.pair(w,i,family,snr)for w,i in zip(x,ids)];waves=np.stack([p[0]for p in pairs]);meta=[dict(cohort='external'if name.startswith('external')else'internal',condition=name,**p[1])for p in pairs];assignments.extend(meta);numerical_max=max(numerical_max,max(abs(m['measured_snr_db']-snr)for m in meta))
  else:waves=x.astype(float)-x.mean(axis=1,keepdims=True,dtype=float)
  d=evaluate(waves,ids,people,y,model,scale,threshold);conditions[name]=d;write_array(o/(name+'.npz'),d)
  for method in METHODS:
   for coverage in COVERAGES:metric_rows.append(dict(condition=name,method=method,target_coverage=coverage,**metrics(y,d['pred'],selection_weights(d[method],ids,coverage))))
   fixed_rows.append(dict(condition=name,method=method,threshold=policies[method]['threshold'],**metrics(y,d['pred'],(d[method]<=policies[method]['threshold']).astype(int))))
  classification[name]=score_metrics(y,d['p'],np.linspace(0,1,11));classification[name]['all_record_metrics']=metrics(y,d['pred'],np.ones(len(y),dtype=int))
 summaries={'internal_new_em':describe('internal_new_em',[conditions[f'internal_em_{s}']for s in spec['internal']['new_severities_db']]),'external_em':describe('external_em',[conditions[f'external_em_{s}']for s in [24,12,6,0]]),'external_clean':describe('external_clean',[conditions['external_clean']])}
 for name,data in [('matched_coverage_metrics',metric_rows),('fixed_policy_metrics',fixed_rows)]:
  with (o/(name+'.csv')).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 (o/'classification_metrics.json').write_text(json.dumps(classification,indent=2)+'\n');(o/'noise_assignments.json').write_text(json.dumps(assignments)+'\n')
 result=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),followup_specification_utc=spec['recorded_utc'],scope='Exploratory fixed-model severity transfer and sampled external validation; original Step9 result unchanged',internal_test_records=len(test_x),external_records=len(external_y),external_positive_records=int(external_y.sum()),external_negative_records=int((1-external_y).sum()),external_class_count_warning=bool(min(external_y.sum(),(1-external_y).sum())<30),new_condition_count=len(conditions),matched_metric_rows=len(metric_rows),fixed_policy_rows=len(fixed_rows),summaries=summaries,max_snr_error_db=numerical_max,quality_missing={k:np.isnan(v['quality']).sum(axis=0).tolist()for k,v in conditions.items()},external_dataset='ecg-arrhythmia1.0.0',external_sampling='2000 header paths selected before labels; adult/rhythm eligibility then applied without replacement',external_patient_linkage_assumption='one patient ECG per record per provider; no independent linkage key',unseen={'severity':'18,9,3,-6 not used to fit methods or policies','withheld_source':'em excluded from original development but already evaluated in Step9; Step10 is not a fresh source holdout','external':'new Chinese hospital ECG source not used in classifier/ranker fitting or policy selection','not_independent':'internal participants and NSTDB noise library reused; external noise uses same library'},models_or_policies_refitted=False,elapsed_seconds=time.perf_counter()-start)
 (o/'generalization_summary.json').write_text(json.dumps(result,indent=2)+'\n');check_step10();print(json.dumps({k:v for k,v in result.items()if k!='quality_missing'},indent=2),flush=True)
if __name__=='__main__':main()
