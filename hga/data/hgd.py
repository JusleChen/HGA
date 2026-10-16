"""Final HGD EDF reader: test classes are decoded only on explicit request."""
from pathlib import Path
import numpy as np
CHANNELS = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']

def edf_header(p):
    with Path(p).open('rb') as f:
        head=f.read(256);n=int(head[252:256]);ext=f.read(256*n)
    assert head[:8].strip()==b'0' and b'EDF+C' in head[192:236]
    pos=0;fields={}
    for name,width in [('labels',16),('transducers',80),('units',8),('physical_min',8),('physical_max',8),('digital_min',8),('digital_max',8),('prefilter',80),('samples_per_record',8),('reserved',32)]:
        fields[name]=[ext[pos+i*width:pos+(i+1)*width].decode('ascii').strip() for i in range(n)];pos+=n*width
    samples=np.array(fields['samples_per_record'],dtype=int);duration=float(head[244:252]);records=int(head[236:244]);header_bytes=int(head[184:192])
    assert Path(p).stat().st_size==header_bytes+records*int(samples.sum())*2,'Incomplete EDF excluded.'
    names=[v[4:] if v.startswith('EEG ') else v for v in fields['labels']]
    picks=[names.index(c) for c in CHANNELS]
    assert all(samples[i]/duration==500 and fields['units'][i]=='uV' for i in picks)
    return dict(fields=fields,names=names,picks=picks,samples=samples,duration=duration,records=records,header_bytes=header_bytes)


def edf_events(p,allow_classes=False):
    # Blind path parses only the TAL numeric prefix, never decodes class text.
    h=edf_header(p);ns=h['samples'];ann=h['fields']['labels'].index('EDF Annotations')
    data=np.memmap(p,dtype='<i2',mode='r',offset=h['header_bytes'],shape=(h['records'],int(ns.sum())))
    offset=int(ns[:ann].sum());onsets=[];classes=[]
    for rec in data[:,offset:offset+ns[ann]]:
        for tal in rec.tobytes().split(b'\x00'):
            if b'\x14' not in tal:continue
            prefix,_,rest=tal.partition(b'\x14')
            if b'\x15' not in prefix:continue
            onset,duration=prefix.split(b'\x15',1)
            if not duration or abs(float(duration)-4.)>1e-6:continue
            onsets.append(float(onset))
            if allow_classes:
                label=rest.split(b'\x14',1)[0].decode('utf-8')
                assert label in ['left_hand','right_hand','feet','rest'],label
                classes.append(['left_hand','right_hand','feet','rest'].index(label))
    del data
    onsets=np.array(onsets);assert len(onsets)>0 and np.all(np.diff(onsets)>0)
    return onsets,np.array(classes,dtype=int) if allow_classes else None

def read_signal_and_onsets(s,split,data_root):
    from scipy.signal import resample_poly
    p=Path(data_root)/split/(str(int(s[1:]))+'.edf');h=edf_header(p);ns=h['samples'];offsets=np.r_[0,np.cumsum(ns)]
    data=np.memmap(p,dtype='<i2',mode='r',offset=h['header_bytes'],shape=(h['records'],int(ns.sum())))
    signals=[]
    for i in h['picks']:
        vals=np.asarray(data[:,offsets[i]:offsets[i+1]]).reshape(-1).astype(np.float64)
        pmin,pmax=float(h['fields']['physical_min'][i]),float(h['fields']['physical_max'][i])
        dmin,dmax=float(h['fields']['digital_min'][i]),float(h['fields']['digital_max'][i])
        signals.append((vals-dmin)*(pmax-pmin)/(dmax-dmin)+pmin)
    del data;signals=np.stack(signals);onset,_=edf_events(p,False)
    assert np.isfinite(signals).all()
    signal=resample_poly(signals,1,2,axis=-1)*1e-6
    signal-=signal.mean(axis=0,keepdims=True)
    centers=np.round(onset*250).astype(int);start=centers-125
    assert np.all(start>=0) and np.all(start+1125<=signal.shape[1]),'No silent boundary-trial exclusion.'
    x=np.stack([signal[:,a:a+1125] for a in start])
    ids=np.array([f'{s}_{split}_trial{i+1:04d}' for i in range(len(x))])
    runs=np.cumsum(np.r_[True,np.diff(onset)>10.]).astype(int)
    return x,runs,ids,onset
