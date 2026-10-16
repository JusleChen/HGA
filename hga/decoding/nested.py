"""Final nested training-run LR selection, with no held-out labels."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from .classical import fit_blocks,transform,fit,calibrate
from .fusion import calibrated

CGRID=(.01,.1,1.)

def view(blocks,condition):
    if condition=='sensor':return blocks[0]
    j={'extra':1,'source':2,'random':3}[condition]
    return np.concatenate([blocks[0],blocks[j]],axis=1)

def fit_nested(arrays,y,runs,condition='source'):
    """Outer LORO predictions plus an all-training refit for one participant.

    Each inner/outer fold refits BOTH scalers and full-SVD PCA48. Final C and
    temperature use all-training OOF only. Runs may be BCIC six or HGD native.
    """
    y=np.asarray(y,dtype=int);runs=np.asarray(runs)
    groups=np.unique(runs).tolist()
    if len(groups)<3:raise ValueError('nested run selection requires at least three runs')
    if any(len(a)!=len(y) for a in arrays):raise ValueError('feature/trial mismatch')
    cache={}
    def blocks(rr):
        key=tuple(sorted(rr))
        if key not in cache:cache[key]=fit_blocks(arrays,np.flatnonzero(np.isin(runs,key)))
        return cache[key]
    def select(rr):
        choices=[]
        for C in CGRID:
            p=np.zeros((len(y),4));scores=[]
            for held in rr:
                tr=np.isin(runs,[r for r in rr if r!=held]);va=runs==held
                bb,_=blocks([r for r in rr if r!=held]);zz=view(bb,condition)
                model=fit(C,zz[tr],y[tr]);p[va]=model.predict_proba(zz[va])
                scores.append(balanced_accuracy_score(y[va],p[va].argmax(1)))
            choices.append((float(np.mean(scores)),C,p))
        best=max(range(len(choices)),key=lambda j:(choices[j][0],-j))
        score,C,p=choices[best];selected=np.isin(runs,rr)
        temp,diagnostics=calibrate(p[selected],y[selected])
        if not diagnostics['success'] or not np.isfinite(temp) or not .1<=temp<=10:temp=1.
        return C,temp,score
    oof=np.zeros((len(y),4))
    for held in groups:
        rr=[r for r in groups if r!=held];C,temp,_=select(rr)
        bb,_=blocks(rr);zz=view(bb,condition);va=runs==held
        model=fit(C,zz[~va],y[~va]);oof[va]=calibrated(model.predict_proba(zz[va]),temp)
    C,temp,score=select(groups);bb,ts=blocks(groups)
    model=fit(C,view(bb,condition),y)
    return oof,dict(model=model,transforms=ts,condition=condition,C=C,temperature=temp,training_OOF_selection_BA=score)

def predict(branch,arrays):
    z=view(transform(arrays,branch['transforms']),branch['condition'])
    return calibrated(branch['model'].predict_proba(z),branch['temperature'])
