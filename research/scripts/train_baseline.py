"""Step6 clean-signal baseline development. No final-test evaluation.

Uses train folds1-6 and checkpoint fold7 only. Reads the entire declared role;
any missing/invalid waveform stops the run. Does not fit error rankers.
"""
import csv,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from wfdb import processing
from cohort_loader import ROOT,manifest,load_records
from model_definition import ECGClassifier


def threshold_at_sensitivity(y,p,target=.90):
    positive=np.sort(np.asarray(p)[np.asarray(y)==1])[::-1]
    if not len(positive):raise ValueError('No positive checkpoint cases')
    return float(positive[int(np.ceil(target*len(positive)))-1])


def metrics(y,p,threshold):
    predicted=(np.asarray(p)>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,predicted,labels=[0,1]).ravel()
    return {'AUROC':float(roc_auc_score(y,p)),'AUPRC':float(average_precision_score(y,p)),'Brier':float(brier_score_loss(y,p)),'threshold':float(threshold),'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)),'balanced_accuracy':float(.5*(tp/(tp+fn)+tn/(tn+fp))),'TP':int(tp),'FN':int(fn),'TN':int(tn),'FP':int(fp)}


def arrays(role):
    rows=manifest(role);ids=[r['ecg_id']for r in rows]
    if role not in ['train','checkpoint']:raise ValueError('Baseline development roles only')
    out=ROOT/'data/derived';out.mkdir(exist_ok=True)
    cache=out/(role+'_leadII.npz')
    if cache.exists():
        with np.load(cache,allow_pickle=False)as data:
            if data['ids'].tolist()!=ids:raise ValueError('Cached role IDs do not match manifest')
            x=data['x'].copy();y=data['y'].copy()
        expected_y=np.asarray([int(r['target_afib_annotation'])for r in rows])
        if x.shape!=(len(rows),1000)or not np.isfinite(x).all()or not np.array_equal(y,expected_y):raise ValueError('Invalid or inconsistent cached role array')
        return x,y,ids
    chunks=[]
    for start in range(0,len(ids),256):
        chunk=load_records(role,ids[start:start+256]).astype('float32');chunks.append(chunk)
        print(f'Verified and loaded {role} {min(start+256,len(ids))}/{len(ids)}',flush=True)
    x=np.concatenate(chunks);y=np.asarray([int(r['target_afib_annotation'])for r in rows],dtype='int64')
    if not np.isfinite(x).all():raise ValueError('Nonfinite source array')
    np.savez_compressed(cache,x=x,y=y,ids=np.asarray(ids))
    return x,y,ids


def rr_features(x):
    result=[]
    for i,signal in enumerate(x):
        peaks=processing.xqrs_detect(signal,fs=100,verbose=False)
        rr=np.diff(peaks)/100
        if len(rr)>=2:
            result.append([len(peaks),np.median(rr),np.std(rr),np.std(rr)/np.mean(rr),np.sqrt(np.mean(np.diff(rr)**2)),np.mean(np.abs(np.diff(rr))>.05)])
        else:result.append([len(peaks),np.nan,np.nan,np.nan,np.nan,np.nan])
        if (i+1)%1000==0:print(f'RR feature records{i+1}/{len(x)}',flush=True)
    return np.asarray(result)


def main():
    seed=20261015;torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    output=ROOT/'outputs/baseline';output.mkdir(exist_ok=True)
    models=ROOT/'models';models.mkdir(exist_ok=True)
    x_train,y_train,train_ids=arrays('train')
    x_val,y_val,val_ids=arrays('checkpoint')
    train_hashes={hashlib.sha256(x.tobytes()).hexdigest()for x in x_train}
    val_hashes={hashlib.sha256(x.tobytes()).hexdigest()for x in x_val}
    flat_train=[identifier for identifier,x in zip(train_ids,x_train)if np.std(x,dtype=np.float64)<=1e-12]
    flat_val=[identifier for identifier,x in zip(val_ids,x_val)if np.std(x,dtype=np.float64)<=1e-12]
    qc={'train_records_verified':len(x_train),'checkpoint_records_verified':len(x_val),'train_unique_leadII_hashes':len(train_hashes),'checkpoint_unique_leadII_hashes':len(val_hashes),'cross_role_identical_leadII_hashes':len(train_hashes&val_hashes),'zero_power_train_ids':flat_train,'zero_power_checkpoint_ids':flat_val,'all_source_samples_finite':True,'no_record_dropped':True}
    (output/'source_quality_audit.json').write_text(json.dumps(qc,indent=2)+'\n')
    if train_hashes&val_hashes or flat_train or flat_val:raise ValueError('Stop before fitting: resolve duplicate or zero-power source records prospectively')
    x_train=x_train-x_train.mean(axis=1,keepdims=True)
    x_val=x_val-x_val.mean(axis=1,keepdims=True)
    scale=float(np.std(x_train,dtype=np.float64))
    if scale<=1e-12:raise ValueError('Invalid training scale')
    train_tensor=torch.from_numpy(x_train/scale).unsqueeze(1)
    val_tensor=torch.from_numpy(x_val/scale).unsqueeze(1)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(train_tensor,torch.from_numpy(y_train.astype('float32'))),batch_size=64,shuffle=True,generator=generator)
    model=ECGClassifier();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best_loss=float('inf');patience_reference=float('inf');stale=0;history=[];begin=time.perf_counter()
    for epoch in range(1,61):
        model.train();loss_sum=0
        for xb,yb in loader:
            optimizer.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(model(xb),yb)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(xb)
        model.eval();logits=[]
        with torch.no_grad():
            for start in range(0,len(val_tensor),128):logits.append(model(val_tensor[start:start+128]))
        logits=torch.cat(logits);val_loss=float(nn.functional.binary_cross_entropy_with_logits(logits,torch.from_numpy(y_val.astype('float32'))))
        p=logits.sigmoid().numpy()
        history.append({'epoch':epoch,'train_BCE':loss_sum/len(x_train),'checkpoint_BCE':val_loss,'checkpoint_AUROC':float(roc_auc_score(y_val,p))})
        if val_loss<best_loss:
            best_loss=val_loss;best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'training_scale_mV':scale,'seed':seed,'best_epoch':epoch,'protocol_sha256':hashlib.sha256((ROOT/'config/analysis_protocol.json').read_bytes()).hexdigest()},models/'cnn_seed20261015.pt')
        if val_loss<patience_reference-.0001:patience_reference=val_loss;stale=0
        else:stale+=1
        (output/'epoch_history.json').write_text(json.dumps(history,indent=2)+'\n')
        print(json.dumps(history[-1]),flush=True)
        if stale>=8:break
    training_seconds=time.perf_counter()-begin
    checkpoint=torch.load(models/'cnn_seed20261015.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    with torch.no_grad():p=torch.cat([model(val_tensor[i:i+128])for i in range(0,len(val_tensor),128)]).sigmoid().numpy()
    threshold=threshold_at_sensitivity(y_val,p)
    cnn=metrics(y_val,p,threshold)
    gate=cnn['AUROC']>=.80 and cnn['specificity']>=.50
    with (output/'checkpoint_predictions.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=['ecg_id','reference_AFIB','p_AFIB','predicted_AFIB'])
        writer.writeheader()
        for identifier,label,prob in zip(val_ids,y_val,p):writer.writerow({'ecg_id':identifier,'reference_AFIB':int(label),'p_AFIB':float(prob),'predicted_AFIB':int(prob>=threshold)})
    # Simple RR-feature reference model; no tuning or gate decisions depend on
    # its comparison. It is not the study's clinical classifier.
    f_train=rr_features(x_train);f_val=rr_features(x_val)
    rr_model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=seed))
    rr_model.fit(f_train,y_train);rr_p=rr_model.predict_proba(f_val)[:,1]
    rr_threshold=threshold_at_sensitivity(y_val,rr_p)
    # Joblib is used only for our own private estimator, not an untrusted input.
    import joblib
    joblib.dump(rr_model,models/'rr_logistic_baseline.joblib')
    np.savez_compressed(output/'rr_feature_arrays.npz',train=f_train,checkpoint=f_val)
    summary={'stage':'Step6 clean-signal development only','seed':seed,'train_records':len(x_train),'checkpoint_records':len(x_val),'train_positive_records':int(y_train.sum()),'checkpoint_positive_records':int(y_val.sum()),'best_epoch':best_epoch,'epochs_run':len(history),'CNN_training_seconds':training_seconds,'training_scale_mV':scale,'CNN_checkpoint_metrics':cnn,'RR_logistic_checkpoint_metrics':metrics(y_val,rr_p,rr_threshold),'constant_negative_reference':metrics(y_val,np.full(len(y_val),y_train.mean()),1.0),'development_gate_passed':gate,'gate_note':'Gate uses checkpoint data used for model selection and threshold setting; this is not independent validation','primary_risk_experiment_run':False,'ranker_fitted':False,'policy_validation_used':False,'final_test_used':False,'RR_baseline_features':['beat_count','median_RR','RR_std','RR_coefficient_of_variation','RMSSD','pNN50'],'RR_baseline_limit':'Short-record and automatic-QRS feature baseline; not a validated HRV measurement tool'}
    (output/'baseline_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    if not gate:print('STOP: development adequacy gate failed; do not begin reliability experiment',flush=True)


if __name__=='__main__':main()
