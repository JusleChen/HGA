"""Final F1 ROI and sensor power features (four bands, three windows)."""
from __future__ import annotations
import numpy as np
from scipy.signal import welch
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
BANDS = [(8,12),(12,16),(16,22),(22,30)]
WINDOWS = [(250,525),(525,800),(800,1075)]

def powers(signal,vector=False,groups=None):
    out=[]
    for lo,hi in WINDOWS:
        f,psd=welch(signal[...,lo:hi],fs=250,window='hann',nperseg=250,noverlap=125,nfft=250,detrend='constant',scaling='density',axis=-1)
        if vector: psd=psd.sum(axis=1)
        p=np.stack([np.trapezoid(psd[..., (f>=a)&(f<=b)],f[(f>=a)&(f<=b)],axis=-1) for a,b in BANDS],axis=-1)
        if groups is not None: p=np.stack([p[ix].mean(axis=0) for ix in groups])
        out.append(p)
    return np.stack(out,axis=-1)


def extra_features(power):
    log=np.log(power+1e-30)
    n=len(power); assert n%2==0
    normalized=np.log((power+1e-30)/(power.sum(axis=0,keepdims=True)+n*1e-30))
    return np.concatenate([log.ravel(),(log[:n//2]-log[n//2:]).ravel(),(log[...,2]-log[...,0]).ravel(),normalized.ravel()])
