"""Independent scalar metric/expanded-bootstrap audit of private saved predictions."""
import csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SOURCE=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE.parents[1]/'research'
sys.path.insert(0,str(SOURCE/'scripts'))
from metrics_reference import retained_indices,selective_metrics
from reliability_methods import score_saved,raw_confidence,QUALITY_NAMES
from robustness_tools import score_ablation
from test_evaluation import bootstrap_primary
plan=json.loads((HERE/'PLAN.json').read_text());result=json.loads((HERE/'RESULTS.json').read_text())
assert result['plan_sha256']==hashlib.sha256((HERE/'PLAN.json').read_bytes()).hexdigest()
for name,digest in plan['preserved_inputs_sha256'].items():assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==digest
matched=list(csv.DictReader((HERE/'matched_metrics.csv').open()));fixed=list(csv.DictReader((HERE/'fixed_policy_metrics.csv').open()));checked=0;explicit_checks=0;max_score_difference=0.
for summary in result['comparisons']:
 seed=summary['seed'];cohort=summary['cohort'];private=HERE/'PRIVATE_predictions'/cohort/str(seed);conditions={}
 directory=SOURCE/'models/reliability'if seed==20261015 else HERE/'PRIVATE_seeds'/str(seed)
 ablation=SOURCE/'outputs/robustness/ablations/quality_no_class.json'if seed==20261015 else directory/'quality_no_class.json'
 assert json.loads(ablation.read_text())['feature_names']==QUALITY_NAMES+['missing_'+n for n in QUALITY_NAMES]
 for path in sorted(private.glob('*.npz')):
  with np.load(path)as z:d={k:z[k]for k in z.files}
  conditions[path.stem]=d
  for method in ['raw_confidence','confidence_adjusted','quality_only','combined','quality_no_class']:
   score=raw_confidence(d['p'])if method=='raw_confidence'else score_ablation(ablation,d['p'],d['quality'])if method=='quality_no_class'else score_saved(directory/(method+'.json'),d['p'],d['quality'])
   difference=float(np.max(np.abs(score-d[method])));max_score_difference=max(max_score_difference,difference);assert difference<1e-12
  for mode,rows in [('matched',matched),('fixed',fixed)]:
   for row in [r for r in rows if int(r['seed'])==seed and r['cohort']==cohort and r['condition']==path.stem]:
    if row['method']=='unfiltered':indices=list(range(len(d['y'])))
    elif mode=='matched':indices=retained_indices(d[row['method']].tolist(),d['ids'].tolist(),float(row['target_coverage']))
    else:indices=np.flatnonzero(d[row['method']]<=float(row['threshold'])).tolist()
    independent=selective_metrics(d['y'].tolist(),d['pred'].tolist(),indices)
    for key,value in independent.items():
     if value is not None:assert abs(float(row[key])-value)<1e-12,(seed,cohort,path.stem,row['method'],key)
    checked+=1
 noisy=[conditions[f'em_{snr}']for snr in [24,12,6,0]];people=noisy[0]['participant'];unique=np.unique(people);rng=np.random.default_rng(20261030);expanded=[]
 for _ in range(25):
  draw=rng.integers(0,len(unique),len(unique));indices=np.concatenate([np.flatnonzero(people==unique[i])for i in draw]);diffs=[]
  for d in noisy:
   values=[]
   for method in ['quality_no_class','quality_only']:
    order=np.lexsort((d['ids'][indices].astype(str),d[method][indices]));ret=indices[order[:math.ceil(.9*len(indices))]];y=d['y'][ret];pred=d['pred'][ret]
    values.append(.5*(np.mean(pred[y==1]==0)+np.mean(pred[y==0]==1)))
   diffs.append(values[0]-values[1])
  expanded.append(np.mean(diffs))
 with np.load(HERE/f'aggregate_bootstrap_{cohort}_{seed}.npz')as z:
  assert np.allclose(z['effects'][:25],expanded,atol=1e-12,equal_nan=True)
  if summary['CI95']is not None:assert np.allclose(np.nanpercentile(z['effects'],[2.5,97.5]),summary['CI95'],atol=1e-12)
 explicit_checks+=25
required=[s for s in result['comparisons']if s['required_for_new_replication_rule']]
assert len(required)==5
independent_verdict=all(s['CI95']is not None and s['CI95'][1]<0 and min(s['candidate_AF_retention'])>=.8 for s in required)
assert independent_verdict==result['global_replication_requirement_passed']
audit={'status':'passed','independently_recomputed_metric_rows':checked,'expanded_bootstrap_checks':explicit_checks,'max_recomputed_score_difference':max_score_difference,'global_rule_independently_recomputed':independent_verdict,'original_preserved_inputs_unchanged':True,'scope':'internal automated audit using independent scalar count formulas and explicit bootstrap expansion; not independent human review'}
(HERE/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
