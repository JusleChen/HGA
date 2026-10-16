"""Training-only F1 transforms, logistic regression and calibration."""
from __future__ import annotations
import warnings
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning

def fit_blocks(arrays,indices):
    blocks=[];transforms=[]
    for a in arrays:
        scale=StandardScaler().fit(a[indices]);z=scale.transform(a)
        pca=PCA(n_components=48,svd_solver='full',whiten=False).fit(z[indices]);z=pca.transform(z)
        score_scale=StandardScaler().fit(z[indices]);z=score_scale.transform(z)
        assert np.isfinite(z).all();blocks.append(z);transforms.append(dict(scale=scale,pca=pca,score_scale=score_scale))
    return blocks,transforms


def transform(arrays,transforms):return [t['score_scale'].transform(t['pca'].transform(t['scale'].transform(a))) for a,t in zip(arrays,transforms)]


def fit(C,z,y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        m=LogisticRegression(C=C,solver='lbfgs',max_iter=2000,tol=1e-7).fit(z,y)
    assert not any(issubclass(w.category,ConvergenceWarning) for w in caught),'No silent classifier optimizer changes.'
    assert np.array_equal(m.classes_,np.arange(4));return m

def calibrate(probs,y):
    logits=np.log(np.clip(probs,1e-12,1))
    def loss(logt):
        p=softmax(logits/np.exp(logt),axis=1)
        return -float(np.mean(np.log(np.clip(p[np.arange(len(y)),y],1e-12,1))))
    result=minimize_scalar(loss,bounds=(np.log(.1),np.log(10)),method='bounded',options={'xatol':1e-7})
    return float(np.exp(result.x)),dict(training_oof_nll_before=loss(0),training_oof_nll_after=loss(result.x),success=bool(result.success))
