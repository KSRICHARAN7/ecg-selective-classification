"""Pure-Python reference for selective metrics, not an experiment runner."""
import math


def retained_indices(scores, ids, coverage=0.9):
    if len(scores) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError('Scores and unique recording IDs must match')
    if not 0 < coverage <= 1 or not all(math.isfinite(s) for s in scores):
        raise ValueError('Invalid coverage or score')
    return sorted(range(len(scores)), key=lambda i: (scores[i], str(ids[i])))[:math.ceil(coverage*len(scores))]


def selective_metrics(labels, predicted, retained):
    if len(labels) != len(predicted) or any(y not in [0,1] for y in labels+predicted):
        raise ValueError('Binary labels and predictions required')
    if len(set(retained)) != len(retained) or any(i<0 or i>=len(labels) for i in retained):
        raise ValueError('Invalid retained indices')
    pos=sum(labels); neg=len(labels)-pos
    rp=sum(labels[i] for i in retained); rn=len(retained)-rp
    fn=sum(labels[i]==1 and predicted[i]==0 for i in retained)
    fp=sum(labels[i]==0 and predicted[i]==1 for i in retained)
    return {
      'coverage':len(retained)/len(labels) if labels else None,
      'balanced_selective_error':0.5*(fn/rp+fp/rn) if rp and rn else None,
      'selective_error':(fn+fp)/len(retained) if retained else None,
      'positive_retention':rp/pos if pos else None,
      'negative_retention':rn/neg if neg else None,
      'retained_sensitivity':1-fn/rp if rp else None,
      'retained_true_positive_fraction_of_all_positives':(rp-fn)/pos if pos else None,
    }
