"""Step8: fit fixed error rankers on fold8, select unlabeled fold9 policies.

No final-test source or em source is accessed. No validation reference labels
are read. Fold8 fit summaries are not independent performance estimates.
"""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch,wfdb
from cohort_loader import ROOT,manifest,load_records
from signal_tools import resample_noise,add_noise,quality_features
from fault_library import DevelopmentFaultLibrary,assignment,FAMILIES,SNR_LEVELS
from model_definition import ECGClassifier
from reliability_methods import ErrorRanker,raw_confidence,weighted_quantile,policy_weights,score_saved


class PolicyValidationNoise:
    """Explicit separate decoder for only the reserved policy-validation region."""
    def __init__(self):
        self.bank={}
        expected={r['path']:r['sha256']for r in json.loads((ROOT/'outputs/pilot_download_manifest.json').read_text())}
        for family in FAMILIES:
            base=ROOT/'data/pilot/noise'/family
            for suffix in ['.dat','.hea']:
                path=base.with_suffix(suffix)
                if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[path.relative_to(ROOT).as_posix()]:raise ValueError('Noise source changed')
            header=wfdb.rdheader(str(base))
            if header.fs!=360 or header.n_sig!=2:raise ValueError('Unexpected source layout')
            source=wfdb.rdrecord(str(base),sampfrom=600*360,sampto=1180*360,physical=True).p_signal
            if source.shape!=(580*360,2)or not np.isfinite(source).all():raise ValueError('Invalid policy-validation source')
            bank=np.empty((2,58,1000))
            for channel in range(2):
                for slot in range(58):
                    x=resample_noise(source[slot*3600:(slot+1)*3600,channel],360)
                    if not np.isfinite(x).all()or np.std(x)<=1e-12:raise ValueError('Invalid policy-validation noise segment; stop without omission')
                    bank[channel,slot]=x
            self.bank[family]=bank

    def make_pair(self,x,identifier,family,snr,role='validation'):
        if role!='validation':raise PermissionError('Policy decoder refuses other roles')
        if snr not in SNR_LEVELS:raise ValueError('Unknown severity')
        meta=assignment(identifier,family,role)
        corrupted,injected=add_noise(x,self.bank[family][meta['channel'],meta['slot']],snr)
        return x-x.mean(),corrupted,injected,{**meta,'requested_snr_db':snr}


def load_role(role):
    if role not in ['ranker','validation']:raise PermissionError('Only Step8 roles allowed')
    rows=manifest(role);ids=[r['ecg_id']for r in rows]
    chunks=[]
    for start in range(0,len(ids),256):chunks.append(load_records(role,ids[start:start+256]).astype('float32'))
    x=np.concatenate(chunks)
    if np.any(np.std(x,axis=1,dtype=np.float64)<=1e-12):raise ValueError('Zero-power ECG; stop without omission')
    arrays={'x':x,'ids':np.asarray(ids)}
    if role=='ranker':arrays['y']=np.asarray([int(r['target_afib_annotation'])for r in rows])
    # Validation targets are deliberately not requested or copied.
    path=ROOT/f'data/derived/{role}_leadII.npz';temporary=path.with_name(path.stem+'.pending.npz')
    np.savez_compressed(temporary,**arrays)
    with np.load(temporary,allow_pickle=False)as z:
        if not np.array_equal(z['x'],x):raise ValueError('Derived array readback failed')
    temporary.replace(path)
    return rows,x


def condition_arrays(rows,signals,library,role):
    waves=[];qualities=[];ids=[];conditions=[];people=[];labels=[];sources=[]
    for row,x in zip(rows,signals):
        variants=[('clean',x-x.mean(),None)]
        for family in FAMILIES:
            for snr in SNR_LEVELS:
                _,corrupted,_,meta=library.make_pair(x,row['ecg_id'],family,snr,role)
                variants.append((f'{family}_{snr}',corrupted,meta))
        for condition,wave,source in variants:
            waves.append(np.asarray(wave,dtype='float32'));qualities.append(quality_features(wave));ids.append(row['ecg_id']);conditions.append(condition);people.append(row['patient_id'])
            sources.append(source)
            if role=='ranker':labels.append(int(row['target_afib_annotation']))
    return {'waves':np.stack(waves),'quality':np.stack(qualities),'ids':np.asarray(ids),'condition':np.asarray(conditions),'participant':np.asarray(people),'labels':np.asarray(labels)if role=='ranker'else None,'sources':sources}


def predict(model,scale,waves):
    x=waves-waves.mean(axis=1,keepdims=True);tensor=torch.from_numpy(x/scale).unsqueeze(1)
    with torch.no_grad():p=torch.cat([model(tensor[i:i+128])for i in range(0,len(tensor),128)]).sigmoid().numpy()
    raw_confidence(p);return p


def main():
    begin=time.perf_counter();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    out=ROOT/'outputs/reliability_development';out.mkdir(exist_ok=True);models=ROOT/'models/reliability';models.mkdir(exist_ok=True)
    spec=json.loads((ROOT/'config/step8_specification.json').read_text())
    acquisition=json.loads((ROOT/'outputs/step8_acquisition_summary.json').read_text())
    if acquisition['recordings']!=3496 or not acquisition['all_provider_sha256_verified']:raise ValueError('Complete acquisition gate not met')
    checkpoint_path=ROOT/'models/cnn_seed20261015.pt'
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()!=spec['classifier_sha256']:raise ValueError('Baseline model changed')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True);model=ECGClassifier();model.load_state_dict(checkpoint['state_dict']);model.eval()
    threshold=spec['classification_threshold'];scale=checkpoint['training_scale_mV']
    rank_rows,rank_x=load_role('ranker');val_rows,val_x=load_role('validation')
    hashes={}
    for role in ['train','checkpoint']:
        with np.load(ROOT/f'data/derived/{role}_leadII.npz',allow_pickle=False)as z:hashes[role]={hashlib.sha256(x.tobytes()).hexdigest()for x in z['x']}
    hashes['ranker']={hashlib.sha256(x.tobytes()).hexdigest()for x in rank_x};hashes['validation']={hashlib.sha256(x.tobytes()).hexdigest()for x in val_x}
    overlaps={f'{a}__{b}':len(hashes[a]&hashes[b])for i,a in enumerate(hashes)for b in list(hashes)[i+1:]}
    if any(overlaps.values()):raise ValueError('Exact waveform shared across roles; resolve before fitting')
    rank=condition_arrays(rank_rows,rank_x,DevelopmentFaultLibrary(),'ranker');p=predict(model,scale,rank['waves']);pred=p>=threshold;error=(pred!=rank['labels']).astype(int)
    error_people=len(set(rank['participant'][error==1]));correct_people=len(set(rank['participant'][error==0]))
    gate={'distinct_participants_with_any_error':error_people,'distinct_participants_with_any_correct':correct_people,'minimum_each':30}
    (out/'fit_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    if min(error_people,correct_people)<30:raise ValueError('Too few distinct participant error/correct cases; no learned rankers fitted')
    fitted={};fit_weights=np.full(len(p),1/9)
    for name in ['confidence_adjusted','quality_only','combined']:
        ranker=ErrorRanker(name,threshold).fit(p,rank['quality'],error,fit_weights);path=models/(name+'.json');ranker.save(path)
        if np.max(np.abs(score_saved(path,p,rank['quality'])-ranker.score(p,rank['quality'])))>1e-12:raise ValueError('Saved ranker scores do not reproduce')
        fitted[name]=ranker
    print('Fitted all three fixed rankers using fold8 only',flush=True)
    val=condition_arrays(val_rows,val_x,PolicyValidationNoise(),'validation');vp=predict(model,scale,val['waves'])
    weights=policy_weights(val['condition']);scores={'raw_confidence':raw_confidence(vp),**{name:ranker.score(vp,val['quality'])for name,ranker in fitted.items()}}
    policies={}
    for name,values in scores.items():
        cut=weighted_quantile(values,weights,.9);retained=values<=cut
        policies[name]={'threshold':cut,'retain_when':'score <= threshold','target_weighted_coverage':.9,'actual_weighted_coverage':float(np.average(retained,weights=weights)),'actual_unweighted_coverage':float(retained.mean()),'coverage_by_condition':{c:float(retained[val['condition']==c].mean())for c in sorted(set(val['condition']))}}
    (out/'policy_thresholds.json').write_text(json.dumps(policies,indent=2)+'\n')
    for name,data,probs in [('ranker',rank,p),('validation',val,vp)]:
        values={'p_AFIB':probs,'quality':data['quality'],'ids':data['ids'],'condition':data['condition'],'participant':data['participant']}
        if name=='ranker':values.update(reference_AFIB=data['labels'],classification_error=error)
        path=out/(name+'_features_predictions.npz');temporary=path.with_name(path.stem+'.pending.npz')
        np.savez_compressed(temporary,**values)
        with np.load(temporary,allow_pickle=False)as saved:
            for key,value in values.items():
                if not np.array_equal(saved[key],value,equal_nan=value.dtype.kind in 'fc'):raise ValueError('Feature/prediction artifact readback failed')
        temporary.replace(path)
        (out/(name+'_noise_assignments.json')).write_text(json.dumps(data['sources'])+'\n')
    summary={'stage':'Step8 reliability method development','ranker_original_records':len(rank_rows),'ranker_participants':len(set(rank['participant'])),'ranker_variants':len(p),'ranker_error_variants':int(error.sum()),'ranker_correct_variants':int((1-error).sum()),'fit_gate':gate,'effective_training_weight':float(fit_weights.sum()),'validation_original_records':len(val_rows),'validation_variants':len(vp),'validation_labels_used':False,'validation_performance_evaluated':False,'cross_role_exact_waveform_overlap':overlaps,'quality_missing_values_ranker':np.isnan(rank['quality']).sum(axis=0).tolist(),'quality_missing_values_validation':np.isnan(val['quality']).sum(axis=0).tolist(),'methods':list(scores),'method_feature_counts':{name:len(v.feature_names)for name,v in fitted.items()},'policies':policies,'classifier_retrained':False,'withheld_em_used':False,'final_test_used':False,'seconds':time.perf_counter()-begin,'scope':'Fold8 fit diagnostics and unlabeled fold9 policy selection only; no independent comparison or primary result'}
    (out/'development_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    paths=[checkpoint_path,ROOT/'config/step8_specification.json',out/'policy_thresholds.json']+sorted(models.glob('*.json'))
    paths+=[ROOT/p for p in ['config/analysis_protocol.json','config/protocol_amendments/step8_clarifications.json','scripts/reliability_methods.py','scripts/run_reliability_development.py','scripts/fault_library.py','scripts/signal_tools.py','outputs/partitions/ranker.csv','outputs/partitions/validation.csv']]
    lock={'status':'methods_and_policies_locked_not_full_protocol','files':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},'final_test_used':False,'full_protocol_frozen':False}
    (ROOT/'config/reliability_methods_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
