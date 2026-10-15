"""Fit every locked fresh seed using original development roles only."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SOURCE=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE.parents[1]/'research'
sys.path.insert(0,str(SOURCE/'scripts'))
from cohort_loader import manifest
from run_seed_sensitivities import train
from run_reliability_development import condition_arrays,PolicyValidationNoise,predict
from fault_library import DevelopmentFaultLibrary
from robustness_tools import load,AblationRanker
from reliability_methods import ErrorRanker,raw_confidence,policy_weights,weighted_quantile
plan=json.loads((HERE/'PLAN.json').read_text())
for name,digest in plan['preserved_inputs_sha256'].items():assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==digest
rank=load(SOURCE/'outputs/reliability_development/ranker_features_predictions.npz');val=load(SOURCE/'outputs/reliability_development/validation_features_predictions.npz')
assert not {'y','reference_AFIB','classification_error'}&val.keys()
waves={}
for role,library,cache in [('ranker',DevelopmentFaultLibrary(),rank),('validation',PolicyValidationNoise(),val)]:
 with np.load(SOURCE/f'data/derived/{role}_leadII.npz')as z:x=z['x']
 data=condition_arrays(manifest(role),x,library,role)
 assert np.array_equal(data['ids'],cache['ids']) and np.array_equal(data['quality'],cache['quality'],equal_nan=True)
 waves[role]=data['waves']
summaries=[]
for seed in plan['fresh_training_seeds']:
 out=HERE/'PRIVATE_seeds'/str(seed);out.mkdir(parents=True,exist_ok=True)
 if (out/'development_lock.json').exists():
  summaries.append(json.loads((out/'development_lock.json').read_text()));continue
 model,scale,threshold,summary=train(seed,out)
 if not summary['checkpoint_gate_passed']:
  summary['status']='checkpoint_gate_failed';summaries.append(summary);continue
 rp=predict(model,scale,waves['ranker']);error=((rp>=threshold)!=rank['reference_AFIB']).astype(int);people=rank['participant'];gate={'wrong_participants':len(set(people[error==1])),'correct_participants':len(set(people[error==0]))};summary['ranker_gate']=gate
 if min(gate.values())<30:summary['status']='ranker_gate_failed';summaries.append(summary);continue
 fitted={}
 for method in ['confidence_adjusted','quality_only','combined','quality_no_class']:
  ranker=AblationRanker(method,threshold)if method=='quality_no_class'else ErrorRanker(method,threshold)
  ranker.fit(rp,rank['quality'],error,np.full(len(rp),1/9));ranker.save(out/(method+'.json'));fitted[method]=ranker
 vp=predict(model,scale,waves['validation']);scores={'raw_confidence':raw_confidence(vp),**{k:r.score(vp,val['quality'])for k,r in fitted.items()}};weights=policy_weights(val['condition']);policies={name:weighted_quantile(v,weights,.9)for name,v in scores.items()};(out/'policies.json').write_text(json.dumps(policies,indent=2)+'\n')
 summary.update(status='development_locked_before_test_or_external_predictions',classification_threshold=threshold,validation_labels_used=False,files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in out.iterdir()if p.is_file()and p.suffix in ['.json','.pt']})
 (out/'development_lock.json').write_text(json.dumps(summary,indent=2)+'\n');summaries.append(summary);print('Development locked for '+str(seed),flush=True)
(HERE/'seed_development_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
