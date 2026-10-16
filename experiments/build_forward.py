"""Build an HGA leadfield from an externally supplied head asset NPZ."""
import argparse
import json
from pathlib import Path
import numpy as np
from hga.forward import build_leadfield
from hga.inverse import minimum_norm_kernel

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',required=True,help='External NPZ: three coefficients, center_m, radius_scale_m, sources_m, electrodes_m')
    p.add_argument('--config',default='configs/forward.json')
    p.add_argument('--output',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    with np.load(a.assets,allow_pickle=False) as z:
        coef={k:z[k] for k in ('inner_skull','outer_skull','outer_skin')}
        gain,_=build_leadfield(coef,z['center_m'],float(z['radius_scale_m']),z['sources_m'],z['electrodes_m'],
            geometry_order=cfg['geometry_order'],field_order=cfg['field_order'],
            scanner_to_surface=z['scanner_to_surface'] if 'scanner_to_surface' in z else None)
        kernel=minimum_norm_kernel(gain,cfg['inverse']['lambda2'])
        roi={f'roi_{i}':z[f'roi_{i}'] for i in range(8) if f'roi_{i}' in z}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,gain=gain,kernel=kernel,**roi)

if __name__=='__main__':main()
