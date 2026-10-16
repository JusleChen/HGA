"""Anatomical F1 features with matched, fixed sensor controls."""
import numpy as np
from .features import powers,extra_features

CHANNEL_GROUPS=([1,2],[6,7,8],[13,14],[18],[4,5],[10,11,12],[16,17],[20])

def control_maps():
    pool=np.zeros((8,22))
    for i,g in enumerate(CHANNEL_GROUPS):pool[i,g]=1/len(g)
    random=np.random.default_rng(20261009).normal(size=(8,22))
    random/=np.linalg.norm(random,axis=1,keepdims=True)
    return pool,random

def extract_trials(epochs,kernel,roi_indices):
    """Return [sensor264, extra272, source272, random272] arrays.

    ROI order is four left parcels followed by their four right homologues.
    Kernel rows are vertex-major XYZ; epochs are 22-by-1125 CAR volts.
    """
    x=np.asarray(epochs)
    k=np.asarray(kernel).reshape(-1,3,22)
    if x.ndim!=3 or x.shape[1:]!=(22,1125) or len(roi_indices)!=8:
        raise ValueError('22-channel/1125-sample epochs and eight ordered ROIs are required')
    union=np.unique(np.concatenate(roi_indices))
    groups=[np.searchsorted(union,g) for g in roi_indices]
    if any(len(g)==0 for g in groups):raise ValueError('empty anatomical ROI')
    roi_kernel=k[union].reshape(-1,22)
    pool,random=control_maps()
    arrays=[[],[],[],[]]
    for trial in x:
        q=(roi_kernel@trial).reshape(len(union),3,1125)
        arrays[0].append(np.log(powers(trial)+1e-30).ravel())
        arrays[1].append(extra_features(powers(pool@trial)))
        arrays[2].append(extra_features(powers(q,True,groups)))
        arrays[3].append(extra_features(powers(random@trial)))
    return [np.stack(a) for a in arrays]
