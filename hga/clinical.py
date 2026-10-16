"""Spatial clinical endpoints separated from outcome-blind forward construction."""
import numpy as np


def peak_spatial_metrics(amplitude, sources_mm, clinical_hemisphere_indices,
                         mask_vertices_mm, mask_faces, sources_inside_mask):
    """Peak-to-mask exact triangle distance and Hit@0/10/20.

    All positions must already share native scanner-RAS coordinates. The
    boolean inside-mask array comes from the postoperative image sampling;
    source activity cannot define the mask or the clinical hemisphere.
    """
    import trimesh
    a=np.asarray(amplitude,dtype=float)
    if a.ndim!=1 or len(a)!=len(sources_mm):
        raise ValueError('supply a source vector at the frozen imaging sample')
    indices=np.asarray(clinical_hemisphere_indices,dtype=int)
    peak=int(indices[np.argmax(np.abs(a[indices]))])
    inside=np.asarray(sources_inside_mask,dtype=bool)
    if inside.shape!=a.shape:
        raise ValueError('mask sampling must match the source grid')
    if inside[peak]:
        distance=0.0
    else:
        mesh=trimesh.Trimesh(vertices=mask_vertices_mm,faces=mask_faces,process=False)
        _,dist,_=trimesh.proximity.closest_point_naive(mesh,np.asarray(sources_mm)[peak:peak+1])
        distance=float(dist[0])
    squared=np.square(np.abs(a))
    energy=float(squared[inside].sum()/squared.sum()) if squared.sum()>0 else 0.0
    return dict(peak_index=peak,distance_mm=distance,Hit0=int(distance==0),
                Hit10=int(distance<=10),Hit20=int(distance<=20),mask_energy=energy)
