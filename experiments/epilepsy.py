"""Apply the four classical inverses with matched external forward/covariance."""
import argparse
from pathlib import Path
from hga.inverse import clinical_inverse

def main():
    import mne
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evoked',required=True)
    p.add_argument('--forward',required=True)
    p.add_argument('--covariance',required=True)
    p.add_argument('--method',choices=['MNE','dSPM','sLORETA','eLORETA'],default='dSPM')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    evoked=mne.read_evokeds(a.evoked,condition=0,verbose=False)
    stc=clinical_inverse(evoked,mne.read_forward_solution(a.forward,verbose=False),
        mne.read_cov(a.covariance,verbose=False),a.method)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);stc.save(a.output,ftype='h5',overwrite=False)

if __name__=='__main__':main()
