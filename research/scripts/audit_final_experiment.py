"""Independent saved-artifact check, with explicit bootstrap expansion."""
import hashlib,json,math,csv
from collections import Counter
from pathlib import Path
import numpy as np
from metrics_reference import retained_indices,selective_metrics
from reliability_methods import raw_confidence,score_saved
from run_final_experiment import validate_locks
from cohort_loader import ROOT

def main():
 validate_locks();o=ROOT/'outputs/final_experiment';result=json.loads((o/'primary_result.json').read_text());rows=list(csv.DictReader((o/'matched_coverage_metrics.csv').open()));fixed=list(csv.DictReader((o/'fixed_policy_metrics.csv').open()));policies=json.loads((ROOT/'outputs/reliability_development/policy_thresholds.json').read_text());max_score_diff=0.;conditions={}
 for path in sorted(o.glob('*.npz')):
  if path.name=='primary_bootstrap.npz':continue
  with np.load(path,allow_pickle=False)as z:d={k:z[k]for k in z.files}
  conditions[path.stem]=d
  for method in ['raw_confidence','confidence_adjusted','quality_only','combined']:
   scores=raw_confidence(d['p'])if method=='raw_confidence'else score_saved(ROOT/f'models/reliability/{method}.json',d['p'],d['quality']);diff=float(np.max(np.abs(d[method]-scores)));max_score_diff=max(max_score_diff,diff);assert diff<1e-12
   for row in [v for v in rows if v['condition']==path.stem and v['method']==method]:
    idx=retained_indices(scores.tolist(),d['ids'].tolist(),float(row['target_coverage']));m=selective_metrics(d['y'].tolist(),d['pred'].tolist(),idx)
    for key in m:
     if m[key]is not None:assert abs(float(row[key])-m[key])<1e-12,(path,method,key)
   row=next(v for v in fixed if v['condition']==path.stem and v['method']==method);idx=np.flatnonzero(scores<=policies[method]['threshold']).tolist();m=selective_metrics(d['y'].tolist(),d['pred'].tolist(),idx)
   for key in m:
    if m[key]is not None:assert abs(float(row[key])-m[key])<1e-12,(path,method,key)
 ds=[conditions[f'em_{s}']for s in [24,12,6,0]];people=ds[0]['participant'];unique=np.unique(people);rng=np.random.default_rng(20261018);explicit=[]
 for b in range(25):
  draw=rng.integers(0,len(unique),len(unique));indices=np.concatenate([np.flatnonzero(people==unique[j])for j in draw]);effects=[]
  for d in ds:
   values=[]
   for name in ['combined','confidence_adjusted']:
    order=np.lexsort((d['ids'][indices].astype(str),d[name][indices]));ret=indices[order[:math.ceil(.9*len(indices))]];y=d['y'][ret];p=d['pred'][ret];values.append(.5*(np.mean(p[y==1]==0)+np.mean(p[y==0]==1)))
   effects.append(values[0]-values[1])
  explicit.append(np.mean(effects))
 with np.load(o/'primary_bootstrap.npz')as z:
  max_boot_diff=float(np.max(np.abs(z['effects'][:25]-explicit)));assert max_boot_diff<1e-12
  ci=np.percentile(z['effects'],[2.5,97.5]);assert np.max(np.abs(ci-result['inference']['percentile95_CI']))<1e-12
 with np.load(ROOT/'outputs/final_experiment_interrupted/clean.npz',allow_pickle=False)as z:
  for key,value in conditions['clean'].items():assert np.array_equal(z[key],value,equal_nan=value.dtype.kind in 'fc')
 noise=json.loads((o/'noise_assignments.json').read_text());snr_error=max(abs(r['measured_snr_db']-r['snr_db'])for r in noise);assert snr_error<1e-10
 reuse={}
 for family in ['bw','ma','em']:
  # One assignment per original recording, not per severity.
  c=Counter((r['channel'],r['slot'])for r in noise if r['family']==family and r['snr_db']==24);reuse[family]={'used_channel_windows':len(c),'max_ECGs_per_window':max(c.values()),'uses_beyond_first':sum(c.values())-len(c),'independent_noise_donors':1}
 report=dict(status='passed',interrupted_clean_arrays_reproduced_exactly=True,saved_conditions=len(conditions),matched_metric_rows=len(rows),fixed_policy_rows=len(fixed),score_reproduction_max_difference=max_score_diff,explicit_bootstrap_replicates_checked=25,bootstrap_multiplicity_vs_expansion_max_difference=max_boot_diff,max_absolute_snr_error_db=snr_error,noise_source_reuse=reuse,protocol_and_model_hashes_unchanged=True)
 (o/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
