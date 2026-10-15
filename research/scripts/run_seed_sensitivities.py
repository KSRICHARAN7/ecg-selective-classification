"""Replicate the full primary method-development workflow for two fixed seeds."""
import json,random,time,hashlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from cohort_loader import ROOT,manifest,load_records
from model_definition import ECGClassifier
from train_baseline import arrays,threshold_at_sensitivity,metrics as classification_metrics
from run_reliability_development import predict,condition_arrays,PolicyValidationNoise
from fault_library import DevelopmentFaultLibrary
from run_final_experiment import FinalNoise
from reliability_methods import ErrorRanker,raw_confidence,policy_weights,weighted_quantile
from test_evaluation import METHODS,metrics,selection_weights
from robustness_tools import check,load,write_array,interval

def train(seed,out):
 torch.manual_seed(seed);random.seed(seed);np.random.seed(seed);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 x,y,ids=arrays('train');v,vy,vids=arrays('checkpoint');x=x-x.mean(axis=1,keepdims=True);v=v-v.mean(axis=1,keepdims=True);scale=float(np.std(x,dtype=np.float64));xt=torch.from_numpy(x/scale).unsqueeze(1);vt=torch.from_numpy(v/scale).unsqueeze(1)
 loader=DataLoader(TensorDataset(xt,torch.from_numpy(y.astype('float32'))),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed));model=ECGClassifier();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);best=float('inf');ref=float('inf');stale=0;history=[];start=time.perf_counter();path=out/'cnn.pt'
 for epoch in range(1,61):
  model.train();total=0
  for xb,yb in loader:
   opt.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(model(xb),yb)
   if not torch.isfinite(loss):raise ValueError('Nonfinite seed loss')
   loss.backward();opt.step();total+=float(loss.detach())*len(xb)
  model.eval()
  with torch.no_grad():logits=torch.cat([model(vt[i:i+128])for i in range(0,len(vt),128)])
  bce=float(nn.functional.binary_cross_entropy_with_logits(logits,torch.from_numpy(vy.astype('float32'))));history.append(dict(epoch=epoch,train_BCE=total/len(x),checkpoint_BCE=bce))
  if bce<best:best=bce;best_epoch=epoch;torch.save(dict(state_dict=model.state_dict(),training_scale_mV=scale,seed=seed,best_epoch=epoch),path)
  if bce<ref-.0001:ref=bce;stale=0
  else:stale+=1
  (out/'epoch_history.json').write_text(json.dumps(history,indent=2)+'\n');print(f'Seed{seed} epoch{epoch} checkpoint BCE {bce:.6f}',flush=True)
  if stale>=8:break
 ck=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);model.eval()
 with torch.no_grad():p=torch.cat([model(vt[i:i+128])for i in range(0,len(vt),128)]).sigmoid().numpy()
 threshold=threshold_at_sensitivity(vy,p);cm=classification_metrics(vy,p,threshold);write_array(out/'checkpoint_predictions.npz',dict(ids=np.array(vids),y=vy,p=p));summary=dict(seed=seed,best_epoch=best_epoch,epochs_run=len(history),training_seconds=time.perf_counter()-start,checkpoint=cm,checkpoint_gate_passed=cm['AUROC']>=.8 and cm['specificity']>=.5)
 (out/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n');return model,scale,threshold,summary

def main():
 spec=check();o=ROOT/'outputs/robustness/seeds';o.mkdir(parents=True,exist_ok=True)
 if (o/'seed_summary.json').exists():raise FileExistsError('Seed sensitivity already complete')
 # Feature quality and variant identities must agree with Step8 saved arrays.
 rank_cache=load(ROOT/'outputs/reliability_development/ranker_features_predictions.npz');val_cache=load(ROOT/'outputs/reliability_development/validation_features_predictions.npz')
 waves={}
 for role,library,cache in [('ranker',DevelopmentFaultLibrary(),rank_cache),('validation',PolicyValidationNoise(),val_cache)]:
  rows=manifest(role)
  with np.load(ROOT/f'data/derived/{role}_leadII.npz')as z:x=z['x']
  data=condition_arrays(rows,x,library,role)
  if not np.array_equal(data['ids'],cache['ids'])or not np.array_equal(data['quality'],cache['quality'],equal_nan=True):raise ValueError('Seed variant identity/quality differs from original')
  waves[role]=data['waves']
 test=load(ROOT/'outputs/final_experiment/clean.npz');test_x=load_records('test',test['ids']).astype('float32');noise=FinalNoise();testwaves={'clean':test_x.astype(float)-test_x.mean(axis=1,keepdims=True,dtype=float)}
 for snr in [24,12,6,0]:testwaves[f'em_{snr}']=np.stack([noise.pair(w,i,'em',snr)[0]for w,i in zip(test_x,test['ids'])])
 summaries=[]
 for seed in spec['seeds']:
  out=o/str(seed);out.mkdir(exist_ok=True);model,scale,threshold,summary=train(seed,out)
  if not summary['checkpoint_gate_passed']:summary['status']='checkpoint_gate_failed_not_evaluated';summaries.append(summary);continue
  rp=predict(model,scale,waves['ranker']);error=((rp>=threshold)!=rank_cache['reference_AFIB']).astype(int);people=rank_cache['participant'];gate={'error_participants':len(set(people[error==1])),'correct_participants':len(set(people[error==0]))};summary['ranker_gate']=gate
  if min(gate.values())<30:summary['status']='ranker_case_gate_failed_not_evaluated';summaries.append(summary);continue
  rankers={}
  for name in ['confidence_adjusted','quality_only','combined']:
   r=ErrorRanker(name,threshold).fit(rp,rank_cache['quality'],error,np.full(len(rp),1/9));r.save(out/(name+'.json'));rankers[name]=r
  vp=predict(model,scale,waves['validation']);val_scores={'raw_confidence':raw_confidence(vp),**{name:r.score(vp,val_cache['quality'])for name,r in rankers.items()}};weights=policy_weights(val_cache['condition']);policies={name:weighted_quantile(s,weights,.9)for name,s in val_scores.items()};(out/'policies.json').write_text(json.dumps(policies,indent=2)+'\n');write_array(out/'validation_scores.npz',dict(ids=val_cache['ids'],condition=val_cache['condition'],p=vp,**val_scores))
  # Record development outputs before final predictions; no validation labels.
  development={'classification_threshold':threshold,'models_and_policies_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in out.glob('*')if p.is_file()},'validation_labels_used':False,'status':'development_complete_before_seed_test_predictions'};(out/'development_lock.json').write_text(json.dumps(development,indent=2)+'\n')
  conditions={};rows=[];fixed=[]
  for name,wave in testwaves.items():
   old=load(ROOT/f'outputs/final_experiment/{name}.npz');p=predict(model,scale,wave.astype('float32'));scores={'raw_confidence':raw_confidence(p),**{n:r.score(p,old['quality'])for n,r in rankers.items()}};d=dict(ids=old['ids'],participant=old['participant'],y=old['y'],pred=(p>=threshold).astype(int),p=p,quality=old['quality'],**scores);conditions[name]=d;write_array(out/(name+'.npz'),d)
   for method in METHODS:
    rows.append(dict(condition=name,method=method,**metrics(d['y'],d['pred'],selection_weights(d[method],d['ids'],.9))));fixed.append(dict(condition=name,method=method,threshold=policies[method],**metrics(d['y'],d['pred'],(d[method]<=policies[method]).astype(int))))
  summary.update(status='completed',primary_sensitivity=interval('primary',[conditions[f'em_{snr}']for snr in [24,12,6,0]],out=out),classification_threshold=threshold,matched_metrics=rows,fixed_policy_metrics=fixed,classification_threshold_refit_role='clean fold7 only',ranker_refit_role='fold8 only',policies_role='unlabeled fold9 only',original_model_replaced=False)
  (out/'seed_result.json').write_text(json.dumps(summary,indent=2)+'\n');summaries.append(summary)
 (o/'seed_summary.json').write_text(json.dumps(summaries,indent=2)+'\n');check();print(json.dumps([{k:s.get(k)for k in ['seed','status','best_epoch','primary_sensitivity']}for s in summaries],indent=2),flush=True)
if __name__=='__main__':main()
