"""Saved-artifact checks independent of Step10 runner metric calculations."""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np
from cohort_loader import ROOT
from run_generalization import check_step10
from reliability_methods import raw_confidence,score_saved
from metrics_reference import retained_indices,selective_metrics

def main():
 check_step10();o=ROOT/'outputs/generalization';summary=json.loads((o/'generalization_summary.json').read_text());rows=list(csv.DictReader((o/'matched_coverage_metrics.csv').open()));fixed=list(csv.DictReader((o/'fixed_policy_metrics.csv').open()));policies=json.loads((ROOT/'outputs/reliability_development/policy_thresholds.json').read_text());arrays={};max_diff=0
 for p in o.glob('*.npz'):
  if p.stem.endswith('_bootstrap'):continue
  with np.load(p,allow_pickle=False)as z:d={k:z[k]for k in z.files}
  arrays[p.stem]=d
  for method in ['raw_confidence','confidence_adjusted','quality_only','combined']:
   score=raw_confidence(d['p'])if method=='raw_confidence'else score_saved(ROOT/f'models/reliability/{method}.json',d['p'],d['quality']);diff=float(np.max(np.abs(score-d[method])));max_diff=max(max_diff,diff);assert diff<1e-12
   selected=[v for v in rows if v['condition']==p.stem and v['method']==method]
   for row in selected:
    idx=retained_indices(score.tolist(),d['ids'].tolist(),float(row['target_coverage']));m=selective_metrics(d['y'].tolist(),d['pred'].tolist(),idx)
    for key,v in m.items():
     if v is not None:assert abs(float(row[key])-v)<1e-12
   row=next(v for v in fixed if v['condition']==p.stem and v['method']==method);idx=np.flatnonzero(score<=policies[method]['threshold']).tolist();m=selective_metrics(d['y'].tolist(),d['pred'].tolist(),idx)
   for key,v in m.items():
    if v is not None:assert abs(float(row[key])-v)<1e-12
 explicit_checks={}
 for name,names in [('internal_new_em',[f'internal_em_{s}'for s in [18,9,3,-6]]),('external_em',[f'external_em_{s}'for s in [24,12,6,0]]),('external_clean',['external_clean'])]:
  ds=[arrays[n]for n in names];people=ds[0]['participant'];unique=np.unique(people);rng=np.random.default_rng(20261018);effects=[]
  for _ in range(25):
   draw=rng.integers(0,len(unique),len(unique));indices=np.concatenate([np.flatnonzero(people==unique[j])for j in draw]);diffs=[]
   for d in ds:
    values=[]
    for method in ['combined','confidence_adjusted']:
     order=np.lexsort((d['ids'][indices].astype(str),d[method][indices]));ret=indices[order[:math.ceil(.9*len(indices))]];y=d['y'][ret];pred=d['pred'][ret];values.append(.5*(np.mean(pred[y==1]==0)+np.mean(pred[y==0]==1)))
    diffs.append(values[0]-values[1])
   effects.append(np.mean(diffs))
  with np.load(o/(name+'_bootstrap.npz'))as z:
   np.testing.assert_allclose(z['effects'][:25],effects,rtol=0,atol=1e-12,equal_nan=True);delta=np.abs(z['effects'][:25]-effects);diff=float(np.nanmax(delta))if np.isfinite(delta).any()else 0.;interval=summary['summaries'][name]['inference']['percentile95_CI']
   if interval is not None:assert np.max(np.abs(np.nanpercentile(z['effects'],[2.5,97.5])-interval))<1e-12
   else:assert np.isnan(z['effects']).sum()>20
  explicit_checks[name]=dict(expanded_replicates_checked=25,max_difference=diff)
 # Old inventory remains available until the new archive is generated.
 original=json.loads((ROOT/'MANIFEST.json').read_text());primary_files={p:h for p,h in original['files'].items()if p.startswith('outputs/final_experiment/')}
 for p,h in primary_files.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 report=dict(status='passed',saved_condition_arrays=len(arrays),matched_metric_rows=len(rows),fixed_policy_rows=len(fixed),score_reproduction_max_difference=max_diff,bootstrap_expansion_checks=explicit_checks,original_step9_files_checked=len(primary_files),original_step9_files_unchanged=True,all_model_and_policy_locks_passed=True)
 (o/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
