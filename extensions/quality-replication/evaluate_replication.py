"""Evaluate locked seeds and a separately selected external sample; save aggregates."""
import csv,datetime,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent
SOURCE=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE.parents[1]/'research'
sys.path.insert(0,str(SOURCE/'scripts'))
from cohort_loader import load_records,manifest
from model_definition import ECGClassifier
from run_reliability_development import predict
from run_final_experiment import FinalNoise
from robustness_tools import load,score_ablation
from reliability_methods import raw_confidence,score_saved
from signal_tools import quality_features
from test_evaluation import metrics,selection_weights,bootstrap_primary
plan=json.loads((HERE/'PLAN.json').read_text())
for path,digest in plan['preserved_inputs_sha256'].items():assert hashlib.sha256((SOURCE/path).read_bytes()).hexdigest()==digest
acquisition=json.loads((HERE/'external_acquisition_summary.json').read_text());assert acquisition['numerical_class_gate_passed']
developments={r['seed']:r for r in json.loads((HERE/'seed_development_summary.json').read_text())};assert set(developments)==set(plan['fresh_training_seeds'])
noise=FinalNoise();original=load(SOURCE/'outputs/final_experiment/clean.npz');ix=load_records('test',original['ids']).astype('float32')
external=load(HERE/'PRIVATE_external_arrays.npz');cohorts={'internal':dict(x=ix,ids=original['ids'],participant=original['participant'],y=original['y']),'fresh_external':dict(x=external['x'],ids=external['ids'],participant=external['ids'],y=external['y'])}
prepared={}
for cohort,base in cohorts.items():
 inputs={}
 for name in plan['conditions']:
  if name=='clean':waves=base['x'].astype(float)-base['x'].mean(axis=1,keepdims=True,dtype=float)
  else:snr=int(name.split('_')[1]);waves=np.stack([noise.pair(w,i,'em',snr)[0]for w,i in zip(base['x'],base['ids'])])
  quality=np.stack([quality_features(w)for w in waves])if cohort=='fresh_external'else load(SOURCE/f'outputs/final_experiment/{name}.npz')['quality']
  inputs[name]={'waves':waves,'quality':quality}
 prepared[cohort]=inputs
methods=['raw_confidence','confidence_adjusted','quality_only','combined','quality_no_class'];matched=[];fixed=[];summaries=[];failed=[]
for seed in [plan['original_seed_reference']]+plan['fresh_training_seeds']:
 if seed!=plan['original_seed_reference'] and developments[seed]['status']!='development_locked_before_test_or_external_predictions':failed.append(developments[seed]);continue
 if seed==plan['original_seed_reference']:
  directory=SOURCE/'models';checkpoint=directory/'cnn_seed20261015.pt';models=directory/'reliability';ablation=SOURCE/'outputs/robustness/ablations/quality_no_class.json';threshold=json.loads((SOURCE/'config/step8_specification.json').read_text())['classification_threshold'];policies={k:v['0.9']for k,v in json.loads((SOURCE/'outputs/robustness/review_policies.json').read_text()).items()}
 else:
  directory=HERE/'PRIVATE_seeds'/str(seed);checkpoint=directory/'cnn.pt';models=directory;ablation=directory/'quality_no_class.json';threshold=developments[seed]['classification_threshold'];policies=json.loads((directory/'policies.json').read_text())
  for name,digest in developments[seed]['files_sha256'].items():assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
 assert 'predicted_AFIB' not in json.loads(ablation.read_text())['feature_names']
 ck=torch.load(checkpoint,map_location='cpu',weights_only=True);model=ECGClassifier();model.load_state_dict(ck['state_dict']);model.eval();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 for cohort,base in cohorts.items():
  private=HERE/'PRIVATE_predictions'/cohort/str(seed);private.mkdir(parents=True,exist_ok=True);conditions=[];allrows=[]
  for name in plan['conditions']:
   data=prepared[cohort][name];quality=data['quality']
   probability=load(SOURCE/f'outputs/final_experiment/{name}.npz')['p']if cohort=='internal'and seed==plan['original_seed_reference']else predict(model,ck['training_scale_mV'],data['waves'].astype('float32'))
   pred=(probability>=threshold).astype(int)
   scores={'raw_confidence':raw_confidence(probability),**{method:score_saved(models/(method+'.json'),probability,quality)for method in methods if method not in ['raw_confidence','quality_no_class']},'quality_no_class':score_ablation(ablation,probability,quality)}
   d={'ids':base['ids'],'participant':base['participant'],'y':base['y'],'pred':pred,'p':probability,'quality':quality,**scores};np.savez_compressed(private/(name+'.npz'),**d)
   for method in methods:
    for coverage in plan['matched_coverages']:
     r=dict(cohort=cohort,seed=seed,condition=name,method=method,target_coverage=coverage,**metrics(d['y'],d['pred'],selection_weights(scores[method],d['ids'],coverage)));matched.append(r);allrows.append(r)
    fixed.append(dict(cohort=cohort,seed=seed,condition=name,method=method,validation_target=.9,threshold=policies[method],**metrics(d['y'],d['pred'],(scores[method]<=policies[method]).astype(int))))
   baseline=dict(cohort=cohort,seed=seed,condition=name,method='unfiltered',target_coverage=1.,**metrics(d['y'],d['pred'],np.ones(len(d['y']),dtype=int)));matched.append(baseline);allrows.append(baseline)
   if name!='clean':conditions.append(dict(ids=d['ids'],participant=d['participant'],y=d['y'],pred=d['pred'],combined=scores['quality_no_class'],confidence_adjusted=scores['quality_only']))
   print(f'Evaluated {cohort} seed{seed} {name}',flush=True)
  pos_units=len(set(base['participant'][base['y']==1]));neg_units=len(set(base['participant'][base['y']==0]));gate=min(pos_units,neg_units)>=30
  inference,effects,per=bootstrap_primary(conditions,replicates=2000,seed=20261030)if gate else ({'percentile95_CI':None},np.array([]),np.array([]))
  differences=[];retention=[];method_means={}
  for method in methods+['unfiltered']:
   chosen=[r for r in allrows if r['method']==method and r['condition']!='clean'and r['target_coverage']==(1 if method=='unfiltered'else .9)]
   values=[r['balanced_selective_error']for r in chosen];method_means[method]={'mean_balanced_retained_error':float(np.mean(values))if None not in values else None,'AF_retention':[r['positive_retention']for r in chosen],'mean_overall_error':float(np.mean([r['selective_error']for r in chosen]))}
  for condition in ['em_24','em_12','em_6','em_0']:
   a=next(r for r in allrows if r['condition']==condition and r['method']=='quality_no_class'and r['target_coverage']==.9);b=next(r for r in allrows if r['condition']==condition and r['method']=='quality_only'and r['target_coverage']==.9);differences.append(a['balanced_selective_error']-b['balanced_selective_error']if None not in [a['balanced_selective_error'],b['balanced_selective_error']]else None);retention.append(a['positive_retention'])
  ci=inference.get('percentile95_CI');passed=bool(gate and ci and ci[1]<0 and all(v is not None and v>=.8 for v in retention));inference.pop('participants',None)
  summary={'cohort':cohort,'seed':seed,'N':len(base['y']),'AF_positive_records':int(base['y'].sum()),'resampling_unit':'participant'if cohort=='internal'else'record','resampling_units':len(set(base['participant'])),'class_gate_passed':gate,'candidate_minus_quality_with_class_effect':float(np.mean(differences))if None not in differences else None,'CI95':ci,'candidate_AF_retention':retention,'AF_retention_requirement_passed':all(v is not None and v>=.8 for v in retention),'comparison_requirement_passed':passed,'required_for_new_replication_rule':cohort=='fresh_external'or seed!=plan['original_seed_reference'],'method_summaries':method_means,'inference':{**inference,'scope':'conditional pointwise interval; paired '+('participant'if cohort=='internal'else'record')+' resampling; external patient linkage unavailable'}};summaries.append(summary)
  np.savez_compressed(HERE/f'aggregate_bootstrap_{cohort}_{seed}.npz',effects=effects,per_snr=per)
for name,rows in [('matched_metrics',matched),('fixed_policy_metrics',fixed)]:
 with (HERE/(name+'.csv')).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
required=[s for s in summaries if s['required_for_new_replication_rule']];passed=len(required)==5 and not failed and all(s['comparison_requirement_passed']for s in required)
report={'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':hashlib.sha256((HERE/'PLAN.json').read_bytes()).hexdigest(),'external_acquisition':acquisition,'seed_development_failures':failed,'comparisons':summaries,'global_replication_requirement_passed':passed,'required_comparisons':5,'completed_required_comparisons':len(required),'matched_metric_rows':len(matched),'fixed_policy_rows':len(fixed),'scope':'locked posthoc-finding followup; internal test participants and recorded noise reused; new external records from previously studied source with no patient-linkage key; original primary remains unchanged','interpretation':'The fixed replication requirement was met in this experiment; clinical or independent human validation is not established.'if passed else'The favorable original ablation did not meet the full locked replication requirement; report seed-specific error and AF retention without promoting the method.'}
(HERE/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'global_replication_requirement_passed':passed,'comparisons':[{k:s[k]for k in ['cohort','seed','candidate_minus_quality_with_class_effect','CI95','candidate_AF_retention','comparison_requirement_passed']}for s in summaries]},indent=2),flush=True)
