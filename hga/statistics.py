"""Participant-level paired inference; average matched seeds before calling."""
import numpy as np
from scipy.stats import wilcoxon

def holm(pvalues):
    p=np.asarray(pvalues,dtype=float);order=np.argsort(p)
    result=np.empty(len(p));result[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return result

def paired_contrast(first,second,resamples=10000,seed=20261011):
    d=np.asarray(first,dtype=float)-np.asarray(second,dtype=float)
    if d.ndim!=1 or len(d)<2:raise ValueError('one mean metric per participant is required')
    rng=np.random.default_rng(seed)
    bootstrap=d[rng.integers(0,len(d),(resamples,len(d)))].mean(1)
    nonzero=d[np.abs(d)>1e-12]
    p=float(wilcoxon(nonzero,alternative='two-sided',method='auto').pvalue) if len(nonzero) else 1.
    return dict(mean=float(d.mean()),median=float(np.median(d)),CI95=np.quantile(bootstrap,[.025,.975]).tolist(),p=p,positive=int((d>1e-12).sum()),zero=int((np.abs(d)<=1e-12).sum()),negative=int((d< -1e-12).sum()))
