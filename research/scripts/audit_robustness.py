"""Independent metric, saved model, bootstrap and primary-preservation checks."""
import json,csv,math
from pathlib import Path
import numpy as np
import torch
from cohort_loader import ROOT,load_records
from robustness_tools import check,load,subset,score_ablation,ABLATIONS
from reliability_methods import raw_confidence,score_saved,weighted_quantile,policy_weights
from metrics_reference import retained_indices,selective_metrics
from train_baseline import threshold_at_sensitivity
from run_reliability_development import predict
from run_final_experiment import FinalNoise
from model_definition import ECGClassifier

def check_metrics(row,d,idx):
 m=selective_metrics(d['y'].tolist(),d['pred'].tolist(),list(idx))
 for k,v in m.items():
  if v is None:assert row[k]in [None,'']
  else:assert abs(float(row[k])-v)<1e-12,(k,row[k],v)

def expanded_bootstrap(ds,candidate,reference,path):
 people=ds[0]['participant'];unique=np.unique(people);rng=np.random.default_rng(20261018);effects=[]
 for _ in range(10):
  draw=rng.integers(0,len(unique),len(unique));indices=np.concatenate([np.flatnonzero(people==unique[j])for j in draw]);diffs=[]
  for d in ds:
   vals=[]
   for name in [candidate,reference]:
    order=np.lexsort((d['ids'][indices].astype(str),d[name][indices]));ret=indices[order[:math.ceil(.9*len(indices))]];y=d['y'][ret];pred=d['pred'][ret];vals.append(.5*(np.mean(pred[y==1]==0)+np.mean(pred[y==0]==1))if len(set(y))==2 else np.nan)
   diffs.append(vals[0]-vals[1])
  effects.append(np.mean(diffs))
 with np.load(path)as z:np.testing.assert_allclose(z['effects'][:10],effects,atol=1e-12,rtol=0,equal_nan=True)
 return 10

def main():
 spec=check();o=ROOT/'outputs/robustness';summary=json.loads((o/'robustness_summary.json').read_text());val=load(ROOT/'outputs/reliability_development/validation_features_predictions.npz');vscore=load(o/'validation_scores.npz');policies=json.loads((o/'review_policies.json').read_text());weight=policy_weights(val['condition']);maxdiff=0.
 assert not any(k in vscore for k in ['y','reference_AFIB','classification_error'])
 for name in policies:
  s=score_ablation(o/'ablations'/(name+'.json'),val['p_AFIB'],val['quality'])if name in ABLATIONS else raw_confidence(val['p_AFIB'])if name=='raw_confidence'else score_saved(ROOT/f'models/reliability/{name}.json',val['p_AFIB'],val['quality'])
  np.testing.assert_allclose(s,vscore[name],atol=1e-12,rtol=0)
  for c,cut in policies[name].items():assert abs(weighted_quantile(s,weight,float(c))-cut)<1e-12
 conditions={name:load(o/(name+'_ablation_scores.npz'))for name in ['clean','em_24','em_12','em_6','em_0']}
 for condition,d in conditions.items():
  original=load(ROOT/f'outputs/final_experiment/{condition}.npz')
  for key,v in original.items():np.testing.assert_array_equal(d[key],v)
  for name in ABLATIONS:
   diff=float(np.max(np.abs(d[name]-score_ablation(o/'ablations'/(name+'.json'),d['p'],d['quality']))));assert diff<1e-12;maxdiff=max(maxdiff,diff)
 matched=list(csv.DictReader((o/'ablation_matched_metrics.csv').open()));fixed=list(csv.DictReader((o/'ablation_fixed_policy_metrics.csv').open()))
 for row in matched:
  d=conditions[row['condition']];idx=retained_indices(d[row['method']].tolist(),d['ids'].tolist(),float(row['target_coverage']));check_metrics(row,d,idx)
 for row in fixed:
  d=conditions[row['condition']];idx=np.flatnonzero(d[row['method']]<=float(row['threshold']));check_metrics(row,d,idx)
 group_rows=list(csv.DictReader((o/'group_metrics.csv').open()))
 for row in group_rows:
  mask=load(o/'subgroup_masks'/(row['group']+'.npz'))['included'];d=subset(conditions[row['condition']],mask);idx=retained_indices(d[row['method']].tolist(),d['ids'].tolist(),.9);check_metrics(row,d,idx)
 flags=list(csv.DictReader((o/'positive_case_flagging.csv').open()));cache={}
 for row in flags:
  condition=row['condition']
  if condition not in cache:cache[condition]=load(ROOT/(f'outputs/generalization/{condition}.npz'if row['cohort']=='external'else f'outputs/final_experiment/{condition}.npz'))
  d=cache[condition];scores=d[row['method']]
  idx=retained_indices(scores.tolist(),d['ids'].tolist(),float(row['target_coverage']))if row['policy']=='matched_budget'else np.flatnonzero(scores<=policies[row['method']][row['target_coverage']]).tolist();check_metrics(row,d,idx);ret=set(idx);deferred=set(range(len(d['y'])))-ret
  af_def=sum(d['y'][i]==1 for i in deferred);tp_def=sum(d['y'][i]==1 and d['pred'][i]==1 for i in deferred);fn_ret=sum(d['y'][i]==1 and d['pred'][i]==0 for i in ret)
  assert int(row['AF_records_deferred'])==af_def;assert int(row['true_positives_deferred'])==tp_def;assert int(row['false_negatives_retained'])==fn_ret
 boot={}
 for name,(reference,_)in ABLATIONS.items():boot['ablation_'+name]=expanded_bootstrap([conditions[f'em_{s}']for s in [24,12,6,0]],name,reference,o/('ablation_'+name+'_bootstrap.npz'))
 for name,g in summary['group_results'].items():
  if g['status']!='completed':continue
  mask=load(o/'subgroup_masks'/(name+'.npz'))['included'];boot['group_'+name]=expanded_bootstrap([subset(conditions[f'em_{s}'],mask)for s in [24,12,6,0]],'combined','confidence_adjusted',o/('group_'+name+'_bootstrap.npz'))
 seeds=json.loads((o/'seeds/seed_summary.json').read_text());seed_audits={};torch.set_num_threads(2);torch.use_deterministic_algorithms(True);noise=FinalNoise();x=load_records('test',conditions['clean']['ids']).astype('float32')
 for seed in seeds:
  if seed['status']!='completed':seed_audits[str(seed['seed'])]={'status':seed['status']};continue
  out=o/'seeds'/str(seed['seed']);dev=json.loads((out/'development_lock.json').read_text());import hashlib
  for name,h in dev['models_and_policies_sha256'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h
  ck=torch.load(out/'cnn.pt',map_location='cpu',weights_only=True);model=ECGClassifier();model.load_state_dict(ck['state_dict']);model.eval();cp=load(out/'checkpoint_predictions.npz');history=json.loads((out/'epoch_history.json').read_text());assert ck['best_epoch']==min(history,key=lambda h:h['checkpoint_BCE'])['epoch'];assert threshold_at_sensitivity(cp['y'],cp['p'])==seed['classification_threshold']
  vp=load(out/'validation_scores.npz');sp=json.loads((out/'policies.json').read_text());assert not any(k in vp for k in ['y','reference_AFIB','classification_error'])
  for name in sp:assert abs(weighted_quantile(vp[name],weight,.9)-sp[name])<1e-12
  sd={};max_prediction_difference=0
  for name in ['clean','em_24','em_12','em_6','em_0']:
   d=load(out/(name+'.npz'));sd[name]=d
   wave=x.astype(float)-x.mean(axis=1,keepdims=True,dtype=float)if name=='clean'else np.stack([noise.pair(w,i,'em',int(name.split('_')[1]))[0]for w,i in zip(x,d['ids'])]);p=predict(model,ck['training_scale_mV'],wave.astype('float32'));diff=float(np.max(np.abs(p-d['p'])));assert diff==0;max_prediction_difference=max(max_prediction_difference,diff)
   for method in ['raw_confidence','confidence_adjusted','quality_only','combined']:
    score=raw_confidence(p)if method=='raw_confidence'else score_saved(out/(method+'.json'),p,d['quality']);np.testing.assert_allclose(score,d[method],atol=1e-12,rtol=0)
    row=next(r for r in seed['matched_metrics']if r['condition']==name and r['method']==method);idx=retained_indices(score.tolist(),d['ids'].tolist(),.9);check_metrics(row,d,idx)
  boot['seed'+str(seed['seed'])]=expanded_bootstrap([sd[f'em_{s}']for s in [24,12,6,0]],'combined','confidence_adjusted',out/'primary_bootstrap.npz');seed_audits[str(seed['seed'])]={'status':'passed','saved_test_prediction_max_difference':max_prediction_difference,'checkpoint_selection_verified':True,'development_lock_verified':True}
 check();report=dict(status='passed',matched_rows=len(matched),fixed_rows=len(fixed),group_rows=len(group_rows),flagging_rows=len(flags),original_result_files_unchanged=len(spec['preserved_files']),original_policies_reproduced=True,validation_arrays_have_no_labels=True,ablation_score_max_difference=maxdiff,expanded_bootstrap_checks=boot,seed_model_audits=seed_audits)
 (o/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
