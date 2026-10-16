"""Minimum-norm decoding kernel and classical clinical inverse interfaces."""
import numpy as np
from scipy.linalg import solve


def minimum_norm_kernel(gain, lambda2=1/9):
    """K=G.T (G G.T + lambda2 trace(G G.T)/E I)^-1; no data-fitted covariance."""
    g = np.asarray(gain, dtype=float)
    gram = g@g.T
    alpha = float(lambda2)*np.trace(gram)/len(g)
    if alpha <= 0 or not np.isfinite(g).all():
        raise ValueError('a finite nonzero gain and positive regularization are required')
    return solve(gram+alpha*np.eye(len(g)),g,assume_a='pos').T


def clinical_inverse(evoked, forward, noise_covariance, method='dSPM', lambda2=1/9):
    """Apply one classical inverse to the same fixed cortical-normal grid."""
    import mne
    if method not in ('MNE','dSPM','sLORETA','eLORETA'):
        raise ValueError('unsupported clinical inverse')
    fwd = mne.convert_forward_solution(forward, surf_ori=True, force_fixed=True, copy=True, verbose=False)
    inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, noise_covariance,
                                                 loose=0, depth=None, fixed=True, verbose=False)
    return mne.minimum_norm.apply_inverse(evoked, inv, lambda2=lambda2, method=method, verbose=False)
