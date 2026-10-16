"""Prepare user-supplied recordings without shipping data or evaluation labels."""
import argparse
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['bcic','hgd'],required=True)
    p.add_argument('--input',required=True,help='BCIC GDF path or HGD EDF dataset root')
    p.add_argument('--subject',help='HGD participant, e.g. H01')
    p.add_argument('--split',choices=['train','test'],required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.dataset=='bcic':
        from hga.data.bcic import read_session
        arrays=read_session(a.input,training=a.split=='train')
    else:
        from hga.data.hgd import read_signal_and_onsets,edf_events
        if not a.subject:p.error('--subject is required for HGD')
        x,runs,ids,onsets=read_signal_and_onsets(a.subject,a.split,a.input)
        arrays=dict(X_v=x,run_id=runs,trial_id=ids,cue_onset_s=onsets)
        if a.split=='train':
            path=Path(a.input)/'train'/(str(int(a.subject[1:]))+'.edf')
            _,arrays['y']=edf_events(path,allow_classes=True)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**arrays)

if __name__=='__main__':main()
