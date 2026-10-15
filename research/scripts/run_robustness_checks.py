"""Fixed targeted Step11 ablations, budget, cohort and subgroup sensitivities."""
import csv,json,ast
from collections import Counter
import numpy as np
from cohort_loader import ROOT
from robustness_tools import check,load,subset,interval,ABLATIONS,AblationRanker,score_ablation,metadata,group_masks,class_gate,flagging_metrics
from reliability_methods import raw_confidence,score_saved,policy_weights,weighted_quantile
from test_evaluation import METHODS,metrics,selection_weights
from run_generalization import write_array
from audit_metadata import classify

def write_csv(path,rows):
 if not rows:raise ValueError('Empty result table')
 with path.open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 spec=check();o=ROOT/'outputs/robustness';o.mkdir(exist_ok=True);models=o/'ablations';models.mkdir(exist_ok=True)
 if (o/'robustness_summary.json').exists():raise FileExistsError('Robustness diagnostics complete; use separate copy')
 rank=load(ROOT/'outputs/reliability_development/ranker_features_predictions.npz');val=load(ROOT/'outputs/reliability_development/validation_features_predictions.npz')
 if any(k in val for k in ['y','reference_AFIB','classification_error']):raise ValueError('Validation label exclusion failed')
 threshold=json.loads((ROOT/'config/step8_specification.json').read_text())['classification_threshold'];val_scores={'raw_confidence':raw_confidence(val['p_AFIB'])};fitted={}
 for method in ['confidence_adjusted','quality_only','combined']:val_scores[method]=score_saved(ROOT/f'models/reliability/{method}.json',val['p_AFIB'],val['quality'])
 for name in ABLATIONS:
  model=AblationRanker(name,threshold).fit(rank['p_AFIB'],rank['quality'],rank['classification_error'],np.full(len(rank['ids']),1/9));path=models/(name+'.json');model.save(path);fitted[name]=model;val_scores[name]=model.score(val['p_AFIB'],val['quality'])
  if np.max(np.abs(score_ablation(path,val['p_AFIB'],val['quality'])-val_scores[name]))>1e-12:raise ValueError('Ablation JSON score mismatch')
 weights=policy_weights(val['condition']);policies={name:{str(c):weighted_quantile(v,weights,c)for c in [.8,.9,.95]}for name,v in val_scores.items()};original=json.loads((ROOT/'outputs/reliability_development/policy_thresholds.json').read_text())
 for name in METHODS:
  if abs(policies[name]['0.9']-original[name]['threshold'])>1e-12:raise ValueError('Original90percent policy differs')
 (o/'review_policies.json').write_text(json.dumps(policies,indent=2)+'\n');write_array(o/'validation_scores.npz',dict(ids=val['ids'],condition=val['condition'],**val_scores))
 conditions={};matched=[];fixed=[]
 for condition in ['clean','em_24','em_12','em_6','em_0']:
  d=load(ROOT/f'outputs/final_experiment/{condition}.npz')
  for name,model in fitted.items():d[name]=model.score(d['p'],d['quality'])
  conditions[condition]=d;write_array(o/(condition+'_ablation_scores.npz'),d)
  for name in val_scores:
   for coverage in [.8,.9,.95]:
    matched.append(dict(condition=condition,method=name,target_coverage=coverage,**metrics(d['y'],d['pred'],selection_weights(d[name],d['ids'],coverage))))
    fixed.append(dict(condition=condition,method=name,validation_target=coverage,threshold=policies[name][str(coverage)],**metrics(d['y'],d['pred'],(d[name]<=policies[name][str(coverage)]).astype(int))))
 write_csv(o/'ablation_matched_metrics.csv',matched);write_csv(o/'ablation_fixed_policy_metrics.csv',fixed)
 ablations={}
 for name,(reference,removed)in ABLATIONS.items():
  print('Ablation interval '+name,flush=True);ablations[name]=dict(base_method=reference,removed_features=removed,feature_count=len(fitted[name].feature_names),comparison=interval('ablation_'+name,[conditions[f'em_{s}']for s in [24,12,6,0]],candidate=name,reference=reference))
 rows,codes=metadata();masks=group_masks(conditions['clean']['ids'],rows,codes);groups={};group_rows=[]
 for name,mask in masks.items():
  if name=='all':continue
  gate=class_gate(conditions['clean'],mask);groups[name]={'gate':gate};(o/'subgroup_masks').mkdir(exist_ok=True);write_array(o/'subgroup_masks'/(name+'.npz'),dict(ids=conditions['clean']['ids'],included=mask))
  demographic=name.startswith(('age','recorded_sex'));native=name.startswith(('native','no_native'))
  # Same conservative class gate for natural-artifact strata as for demographics.
  allowed=gate['numeric_subgroup_metrics_permitted']if demographic or native else gate['positive_records']>0 and gate['negative_records']>0
  if not allowed:groups[name]['status']='insufficient_class_participants';continue
  groups[name]['status']='completed';data=[subset(conditions[f'em_{s}'],mask)for s in [24,12,6,0]];print('Subgroup/cohort interval '+name,flush=True);groups[name]['comparison']=interval('group_'+name,data)
  for condition,d in conditions.items():
   dd=subset(d,mask)
   for method in METHODS:group_rows.append(dict(group=name,condition=condition,method=method,**metrics(dd['y'],dd['pred'],selection_weights(dd[method],dd['ids'],.9))))
 (o/'group_results.json').write_text(json.dumps(groups,indent=2)+'\n');write_csv(o/'group_metrics.csv',group_rows)
 # AF-case flagging across original13 test and sampled external5 conditions.
 flag_rows=[]
 for cohort,path,condition in [('internal',ROOT/f'outputs/final_experiment/{name}.npz',name)for name in ['clean']+[f'{f}_{s}'for f in ['bw','ma','em']for s in [24,12,6,0]]]+[('external',ROOT/f'outputs/generalization/{name}.npz',name)for name in ['external_clean']+[f'external_em_{s}'for s in [24,12,6,0]]]:
  d=load(path)
  for method in METHODS:
   for coverage in [.8,.9,.95]:
    flag_rows.append(dict(cohort=cohort,condition=condition,method=method,policy='matched_budget',target_coverage=coverage,**flagging_metrics(d,selection_weights(d[method],d['ids'],coverage)>0)))
    flag_rows.append(dict(cohort=cohort,condition=condition,method=method,policy='validation_fixed',target_coverage=coverage,**flagging_metrics(d,(d[method]<=policies[method][str(coverage)]).astype(int))))
 write_csv(o/'positive_case_flagging.csv',flag_rows)
 # Metadata-only review-status audit; do not expand the actual fitted cohort.
 review={}
 for fold in range(1,11):
  selected=[r for r in rows if int(r['strat_fold'])==fold];all_status=[(r,classify(r,codes,False))for r in selected];all_eligible=[(r,v)for r,v in all_status if v[0]=='eligible'];human=[(r,v)for r,v in all_eligible if r['validated_by_human']=='True'];review[str(fold)]={'all_status_adult_rhythm_records':len(all_eligible),'all_status_AFIB':sum(v[1]for r,v in all_eligible),'human_reviewed_adult_rhythm_records':len(human),'human_reviewed_AFIB':sum(v[1]for r,v in human),'metadata_only':True}
 (o/'review_status_metadata_audit.json').write_text(json.dumps(review,indent=2)+'\n')
 summary=dict(scope='Exploratory targeted robustness; primary experiment unchanged; pointwise intervals only',ablations=ablations,group_results=groups,ablation_matched_rows=len(matched),ablation_fixed_rows=len(fixed),positive_flagging_rows=len(flag_rows),validation_reference_labels_used=False,original90percent_policies_reproduced=True,seed_results_file='outputs/robustness/seeds/seed_summary.json',native_annotation_limit='Not randomized, not confirmed leadII-specific; no annotation does not establish clean signal',review_status_audit='metadata_only_no_model_expansion')
 (o/'robustness_summary.json').write_text(json.dumps(summary,indent=2)+'\n');check();print('Completed targeted robustness tables',flush=True)
if __name__=='__main__':main()
