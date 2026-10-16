"""Training-OOF calibration, run-balanced weight selection and frozen fusion."""
import numpy as np
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score

LAMBDAS=(0.,.05,.1,.2,.3)

def probabilities(p,n=None):
    p=np.asarray(p)
    if p.ndim!=2 or p.shape[1]!=4 or (n is not None and len(p)!=n) or not np.isfinite(p).all():
        raise ValueError('finite n-by-four probabilities required')
    if p.min()<0 or p.max()>1 or not np.allclose(p.sum(1),1,atol=1e-6,rtol=0):
        raise ValueError('invalid class probabilities')

def calibrated(p,temperature):
    return softmax(np.log(np.clip(p,1e-12,1))/float(temperature),axis=1)

def choose_weight(deep_oof,branch_oof,y,runs,candidates=LAMBDAS):
    probabilities(deep_oof,len(y));probabilities(branch_oof,len(y))
    scores=[]
    for weight in candidates:
        p=(1-weight)*deep_oof+weight*branch_oof
        scores.append(float(np.mean([balanced_accuracy_score(y[runs==r],p[runs==r].argmax(1)) for r in np.unique(runs)])))
    best=max(range(len(scores)),key=lambda i:(scores[i],-float(candidates[i])))
    return float(candidates[best]),scores

def fuse(deep,branch,weight):
    probabilities(deep);probabilities(branch,len(deep))
    if not 0<=weight<=1:raise ValueError('invalid fusion weight')
    return (1-weight)*deep+weight*branch
