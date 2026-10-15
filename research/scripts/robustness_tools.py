"""Fixed Step11 helpers. Original classifiers/rankers are never overwritten."""
import ast,csv,hashlib,json
from pathlib import Path
import numpy as np
from cohort_loader import ROOT
from run_generalization import check_step10,write_array
from test_evaluation import bootstrap_primary
from reliability_methods import ErrorRanker

ABLATIONS={
 'confidence_no_class':('confidence_adjusted',['predicted_AFIB']),
 'quality_no_class':('quality_only',['predicted_AFIB']),
 'combined_no_class':('combined',['predicted_AFIB']),
 'combined_no_amplitude':('combined',['AC_std_mV','peak_to_peak_mV','missing_AC_std_mV','missing_peak_to_peak_mV']),
 'combined_no_spectral':('combined',['low_band_ratio','high_band_ratio','missing_low_band_ratio','missing_high_band_ratio']),
 'combined_no_temporal':('combined',['flat_difference_fraction','RMSSD_over_std','missing_flat_difference_fraction','missing_RMSSD_over_std']),
 'combined_no_kurtosis':('combined',['Pearson_kurtosis','missing_Pearson_kurtosis'])}

def check():
 check_step10();spec=json.loads((ROOT/'config/step11_specification.json').read_text())
 for p,h in {**spec['preserved_files'],**spec.get('implementation_files',{})}.items():
  if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Original result or model changed: '+p)
 return spec

def load(path):
 with np.load(path,allow_pickle=False)as z:return {k:z[k]for k in z.files}

def subset(d,mask):return {k:v[mask]for k,v in d.items()}

def interval(name,conditions,candidate='combined',reference='confidence_adjusted',out=None):
 ds=[dict(ids=d['ids'],participant=d['participant'],y=d['y'],pred=d['pred'],combined=d[candidate],confidence_adjusted=d[reference])for d in conditions]
 summary,effects,per=bootstrap_primary(ds,2000,20261018)
 from test_evaluation import metrics,selection_weights
 diffs=[];ret=[]
 for d in ds:
  a=metrics(d['y'],d['pred'],selection_weights(d['combined'],d['ids'],.9));b=metrics(d['y'],d['pred'],selection_weights(d['confidence_adjusted'],d['ids'],.9));av=a['balanced_selective_error'];bv=b['balanced_selective_error'];diffs.append(av-bv if av is not None and bv is not None else None);ret.append(a['positive_retention'])
 summary.update(candidate=candidate,reference=reference,effect=float(np.mean(diffs))if None not in diffs else None,condition_effects=diffs,AF_retention=ret,AF_guardrail_passed=all(x is not None and x>=.8 for x in ret),scope='Exploratory paired participant sensitivity; fixed fitted models and noise library; pointwise interval')
 if out is None:out=ROOT/'outputs/robustness'
 write_array(out/(name+'_bootstrap.npz'),dict(effects=effects,per_condition=per));return summary

class AblationRanker(ErrorRanker):
 def __init__(self,name,threshold):
  if name not in ABLATIONS:raise ValueError('Unspecified ablation')
  self.ablation=name;super().__init__(ABLATIONS[name][0],threshold)
 def design(self,p,quality,fit=False):
  x=super().design(p,quality,fit);names=super_names(self.kind);keep=[i for i,n in enumerate(names)if n not in ABLATIONS[self.ablation][1]]
  if fit:self.feature_names=[names[i]for i in keep]
  return x[:,keep]
 def save(self,path):
  super().save(path);d=json.loads(Path(path).read_text());d['ablation']=self.ablation;Path(path).write_text(json.dumps(d,indent=2)+'\n')

def super_names(kind):
 from reliability_methods import QUALITY_NAMES
 columns=['p_AFIB','distance_from_classification_threshold','predicted_AFIB']if kind!='quality_only'else['predicted_AFIB']
 return columns+QUALITY_NAMES+['missing_'+k for k in QUALITY_NAMES]if kind!='confidence_adjusted'else columns

def score_ablation(path,p,q):
 from scipy.special import expit
 d=json.loads(Path(path).read_text());r=AblationRanker(d['ablation'],d['classification_threshold'])
 if d['medians']is not None:r.medians=np.array(d['medians'])
 return expit(((r.design(p,q)-np.array(d['center']))/np.array(d['scale']))@np.array(d['coefficient'])+d['intercept'])

def metadata():
 with (ROOT/'data/source/ptbxl_database.csv').open()as f:rows=list(csv.DictReader(f))
 with (ROOT/'data/source/scp_statements.csv').open()as f:codes={r['']for r in csv.DictReader(f)if r['rhythm']=='1.0'}
 return rows,codes

def has_native_artifact(row):
 return any(str(row.get(k,'')).strip().lower()not in ['', 'nan','none']for k in ['baseline_drift','static_noise','burst_noise','electrodes_problems'])

def group_masks(ids,rows,codes):
 lookup={r['ecg_id']:r for r in rows};m=[lookup[str(i)]for i in ids];earliest={}
 for row in m:
  if not row['recording_date']:raise ValueError('Missing date; earliest-record sensitivity unresolved')
  key=(row['recording_date'],row['ecg_id']);person=row['patient_id']
  if person not in earliest or key<earliest[person]:earliest[person]=key
 return {
  'all':np.ones(len(m),bool),
  'exclude_mixed_rhythms':np.array([len(set(ast.literal_eval(r['scp_codes']))&codes)==1 for r in m]),
  'one_earliest_per_participant':np.array([(r['recording_date'],r['ecg_id'])==earliest[r['patient_id']]for r in m]),
  'age18_64':np.array([18<=float(r['age'])<65 for r in m]),
  'age65plus':np.array([float(r['age'])>=65 for r in m]),
  'recorded_sex0':np.array([r['sex']=='0'for r in m]),
  'recorded_sex1':np.array([r['sex']=='1'for r in m]),
  'native_artifact_annotated':np.array([has_native_artifact(r)for r in m]),
  'no_native_artifact_annotation':np.array([not has_native_artifact(r)for r in m])}

def class_gate(d,mask,minimum=30):
 people=d['participant'][mask];y=d['y'][mask];pos=len(set(people[y==1]));neg=len(set(people[y==0]))
 return dict(records=int(mask.sum()),participants=len(set(people)),positive_records=int(y.sum()),negative_records=int((1-y).sum()),positive_participants=pos,negative_participants=neg,numeric_subgroup_metrics_permitted=bool(pos>=minimum and neg>=minimum))

def flagging_metrics(d,retained):
 from test_evaluation import metrics
 r=np.asarray(retained,dtype=int);m=metrics(d['y'],d['pred'],r);y=d['y'];pred=d['pred'];error=pred!=y;flag=r==0;total_tp=int(((y==1)&(pred==1)).sum());total_fn=int(((y==1)&(pred==0)).sum());high=np.maximum(d['p'],1-d['p'])>=.9
 m.update(AF_records_deferred=int(((y==1)&flag).sum()),true_positives_deferred=int(((y==1)&(pred==1)&flag).sum()),all_original_true_positives=total_tp,false_negatives_retained=int(((y==1)&(pred==0)&(r>0)).sum()),all_original_false_negatives=total_fn,all_errors=int(error.sum()),errors_flagged=int((error&flag).sum()),error_capture_fraction=float((error&flag).sum()/error.sum())if error.any()else None,high_confidence_errors=int((error&high).sum()),high_confidence_errors_flagged=int((error&high&flag).sum()),deferred_AF_fraction=1-m['positive_retention']if m['positive_retention']is not None else None)
 return m
