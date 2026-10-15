"""Locked selective metrics and paired participant bootstrap for Step 9."""
import math
import numpy as np

METHODS=('raw_confidence','confidence_adjusted','quality_only','combined')
COVERAGES=(.5,.6,.7,.8,.9,.95,1.)

def selection_weights(scores,ids,coverage=.9,multiplicity=None):
    scores=np.asarray(scores);ids=np.asarray(ids).astype(str)
    if len(scores)!=len(ids) or len(set(ids))!=len(ids) or not np.isfinite(scores).all():raise ValueError('Finite scores and unique original IDs required')
    if not 0<coverage<=1:raise ValueError('Invalid coverage')
    w=np.ones(len(scores),dtype=int) if multiplicity is None else np.asarray(multiplicity)
    if w.shape!=scores.shape or (w<0).any() or not np.equal(w,np.floor(w)).all() or not w.sum():raise ValueError('Nonnegative integer multiplicities required')
    order=np.lexsort((ids,scores));ordered=w[order];k=math.ceil(coverage*int(w.sum()))
    take=np.minimum(ordered,np.maximum(0,k-(np.cumsum(ordered)-ordered)))
    result=np.zeros(len(scores),dtype=int);result[order]=take
    return result

def metrics(y,pred,retained,total=None):
    y=np.asarray(y,dtype=int);pred=np.asarray(pred,dtype=int);r=np.asarray(retained)
    total=np.ones(len(y)) if total is None else np.asarray(total)
    if y.shape!=pred.shape or y.shape!=r.shape or total.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isin(pred,[0,1]).all() or (r<0).any() or (r>total).any():raise ValueError('Invalid metric inputs')
    pos=float(np.sum(total*y));neg=float(np.sum(total*(1-y)));rp=float(np.sum(r*y));rn=float(np.sum(r*(1-y)))
    fn=float(np.sum(r*((y==1)&(pred==0))));fp=float(np.sum(r*((y==0)&(pred==1))));n=float(r.sum())
    return dict(N=float(total.sum()),retained=n,TP=rp-fn,FN=fn,TN=rn-fp,FP=fp,positive_retained=rp,negative_retained=rn,coverage=n/total.sum() if total.sum() else None,balanced_selective_error=.5*(fn/rp+fp/rn) if rp and rn else None,selective_error=(fn+fp)/n if n else None,positive_retention=rp/pos if pos else None,negative_retention=rn/neg if neg else None,retained_sensitivity=1-fn/rp if rp else None,retained_specificity=1-fp/rn if rn else None,retained_true_positive_fraction_of_all_positives=(rp-fn)/pos if pos else None)

def bootstrap_primary(conditions,replicates=2000,seed=20261018):
    """Same participant draw across severities/methods; duplicate copies retained.

    Tied duplicate copies are equivalent. Count multiplicities to avoid expanding
    arrays; boundary duplicate copies can be partially retained, exactly as the
    explicit resampled-record ranking. Re-rank at each resample's ceil(.9*N).
    """
    first=conditions[0];people=np.asarray(first['participant']);unique,inverse=np.unique(people,return_inverse=True);rng=np.random.default_rng(seed);effects=[];per_snr=[]
    for c in conditions:
        if not np.array_equal(c['participant'],people) or not np.array_equal(c['ids'],first['ids']):raise ValueError('Paired cohorts not aligned')
    for _ in range(replicates):
        multiplicity=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[inverse];diffs=[]
        for c in conditions:
            values=[]
            for name in ['combined','confidence_adjusted']:
                take=selection_weights(c[name],c['ids'],.9,multiplicity)
                values.append(metrics(c['y'],c['pred'],take,multiplicity)['balanced_selective_error'])
            diffs.append(values[0]-values[1] if None not in values else np.nan)
        per_snr.append(diffs);effects.append(np.mean(diffs))
    effects=np.asarray(effects);undefined=int(np.isnan(effects).sum())
    interval=np.nanpercentile(effects,[2.5,97.5]).tolist() if undefined<=.01*replicates else None
    return {'replicates':replicates,'seed':seed,'participants':len(unique),'undefined_replicates':undefined,'percentile95_CI':interval,'scope':'conditional on fixed fitted models and assigned noise library; participant sampling only'},effects,np.asarray(per_snr)
