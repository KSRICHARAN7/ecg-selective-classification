"""Fixed error rankers and weighted policy thresholds; no final-test access."""
import json,warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

QUALITY_NAMES=['AC_std_mV','peak_to_peak_mV','flat_difference_fraction','RMSSD_over_std','low_band_ratio','high_band_ratio','Pearson_kurtosis']


def raw_confidence(p):
    p=np.asarray(p,dtype=float)
    if not np.isfinite(p).all()or np.any((p<0)|(p>1)):raise ValueError('Invalid classifier probabilities')
    return 1-np.maximum(p,1-p)


def policy_weights(condition):
    result=[]
    for value in condition:
        if value=='clean':result.append(1/3)
        elif value in [f'{family}_{snr}'for family in ['bw','ma']for snr in [24,12,6,0]]:result.append(1/12)
        else:raise ValueError('Unknown policy mixture condition')
    return np.asarray(result)


def weighted_quantile(values,weights,probability=.9):
    values=np.asarray(values,dtype=float);weights=np.asarray(weights,dtype=float)
    if values.ndim!=1 or values.shape!=weights.shape or len(values)==0:raise ValueError('Mismatching threshold arrays')
    if not np.isfinite(values).all()or not np.isfinite(weights).all()or np.any(weights<=0)or not 0<probability<1:raise ValueError('Invalid weighted quantile inputs')
    order=np.argsort(values,kind='stable');cdf=np.cumsum(weights[order]);index=np.searchsorted(cdf,probability*cdf[-1],side='left')
    return float(values[order[min(index,len(values)-1)]])


class ErrorRanker:
    def __init__(self,kind,classification_threshold):
        if kind not in ['confidence_adjusted','quality_only','combined']:raise ValueError('Unknown ranker')
        if not 0<=classification_threshold<=1:raise ValueError('Invalid classification threshold')
        self.kind=kind;self.classification_threshold=float(classification_threshold)

    def design(self,p,quality,fit=False):
        p=np.asarray(p,dtype=float);raw_confidence(p)
        quality=np.asarray(quality,dtype=float)
        if p.ndim!=1 or quality.shape!=(len(p),7)or np.isinf(quality).any():raise ValueError('Invalid feature layout or infinite feature')
        predicted=(p>=self.classification_threshold).astype(float)
        columns=[];names=[]
        if self.kind in ['confidence_adjusted','combined']:
            columns.extend([p,np.abs(p-self.classification_threshold),predicted]);names.extend(['p_AFIB','distance_from_classification_threshold','predicted_AFIB'])
        else:columns.append(predicted);names.append('predicted_AFIB')
        if self.kind in ['quality_only','combined']:
            if fit:
                self.medians=np.asarray([np.median(c[~np.isnan(c)])if np.any(~np.isnan(c))else 0. for c in quality.T])
            elif not hasattr(self,'medians'):raise ValueError('Ranker has not been fitted')
            filled=np.where(np.isnan(quality),self.medians,quality)
            columns.extend(list(filled.T));columns.extend(list(np.isnan(quality).astype(float).T))
            names.extend(QUALITY_NAMES);names.extend(['missing_'+name for name in QUALITY_NAMES])
        if fit:self.feature_names=names
        return np.column_stack(columns)

    def fit(self,p,quality,error,weights):
        error=np.asarray(error);weights=np.asarray(weights,dtype=float)
        if error.shape!=np.asarray(p).shape or not np.isin(error,[0,1]).all()or len(np.unique(error))!=2:raise ValueError('Both error classes required')
        if weights.shape!=error.shape or not np.isfinite(weights).all()or np.any(weights<=0):raise ValueError('Invalid fit weights')
        features=self.design(p,quality,fit=True);self.scaler=StandardScaler().fit(features)
        self.model=LogisticRegression(C=1,solver='lbfgs',max_iter=2000,tol=1e-4,random_state=20261015)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            self.model.fit(self.scaler.transform(features),error,sample_weight=weights)
        return self

    def score(self,p,quality):
        if not hasattr(self,'model'):raise ValueError('Ranker has not been fitted')
        return self.model.predict_proba(self.scaler.transform(self.design(p,quality)))[:,1]

    def save(self,path):
        data={'kind':self.kind,'classification_threshold':self.classification_threshold,'feature_names':self.feature_names,'medians':self.medians.tolist()if hasattr(self,'medians')else None,'center':self.scaler.mean_.tolist(),'scale':self.scaler.scale_.tolist(),'coefficient':self.model.coef_[0].tolist(),'intercept':float(self.model.intercept_[0]),'fit_iterations':int(self.model.n_iter_[0]),'meaning':'estimated probability that the fixed classifier prediction is incorrect'}
        Path(path).write_text(json.dumps(data,indent=2)+'\n')


def score_saved(path,p,quality):
    from scipy.special import expit
    data=json.loads(Path(path).read_text());ranker=ErrorRanker(data['kind'],data['classification_threshold'])
    if data['medians']is not None:ranker.medians=np.asarray(data['medians'])
    features=ranker.design(p,quality)
    return expit(((features-np.asarray(data['center']))/np.asarray(data['scale']))@np.asarray(data['coefficient'])+data['intercept'])
