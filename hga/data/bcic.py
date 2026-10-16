"""BCIC IV-2a GDF reconstruction for the final decoding input interfaces."""
from pathlib import Path
import numpy as np


def read_session(gdf_path,training=False):
    """Read all 288 trials, keeping E event classes unopened.

    X_v is CAR volts over trial-start [1.5,6) s. mi_X is acquisition-reference
    run-filtered 8–30 Hz over cue [0,4] s, before batch EA/interpolation.
    """
    import mne
    raw=mne.io.read_raw_gdf(Path(gdf_path),preload=True,verbose='ERROR')
    if raw.info['sfreq']!=250:raise ValueError('expected official 250-Hz GDF')
    signal=raw.get_data(picks=list(range(22)))
    ev,codes=mne.events_from_annotations(raw,verbose='ERROR')
    starts=ev[ev[:,2]==codes['768'],0]
    if len(starts)!=288:raise ValueError('expected all 288 official trials')
    runs=np.repeat(np.arange(1,7),48);cues=starts+500
    y=None
    if training:
        labels=sorted((int(e[0]),c-769) for c in range(769,773) for e in ev[ev[:,2]==codes[str(c)]])
        if not np.array_equal([a for a,b in labels],cues):raise ValueError('cue/class alignment failed')
        y=np.array([b for a,b in labels],dtype=np.int64)
    elif any(str(c) in codes for c in range(769,773)):
        raise ValueError('evaluation GDF unexpectedly contains class events')
    # Preserve the final frozen cache precision convention for ATC/TC inputs.
    uv=(signal*1e6).astype(np.float32)
    x=np.stack([uv[:,a+375:a+1500].astype(np.float64)*1e-6 for a in starts])
    x-=x.mean(axis=1,keepdims=True)
    bounds=np.unique(np.r_[0,ev[ev[:,2]==codes['32766'],0],raw.n_times])
    mi=np.empty((288,22,1001),dtype=np.float32)
    for run in range(1,7):
        ix=np.flatnonzero(runs==run);b=np.searchsorted(bounds,starts[ix[0]],side='right')-1
        lo,hi=int(bounds[b]),int(bounds[b+1])
        if cues[ix[-1]]+1001>hi:raise ValueError('incomplete run; no silent trial deletion')
        filtered=mne.filter.filter_data(signal[:,lo:hi],250,8,30,method='iir',
            iir_params=dict(order=4,ftype='butter',output='sos'),phase='zero',verbose='ERROR')
        for i in ix:mi[i]=filtered[:,cues[i]-lo:cues[i]-lo+1001]
    result=dict(X_v=x,mi_X=mi,run_id=runs)
    if y is not None:result['y']=y
    return result
