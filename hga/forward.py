"""Small public API over the final harmonic operator.

All coordinates accepted here are metres in one common coordinate frame.
The retained internal operator takes a millimetre center/scale, so that
conversion is explicit at its boundary. Leadfields have XYZ source columns.
"""
from __future__ import annotations
import copy
import numpy as np
from .harmonics import SURFACE_NAMES, joint_sphere_center, fit_log_radius, sphere_quadrature
from .covariant_operator import _operator_class


def fit_geometry(surfaces_m, order=12, radius_scale_m=0.1):
    """Fit three anatomical meshes without reading EEG or clinical outcomes."""
    surfaces_mm = {k: (np.asarray(surfaces_m[k][0])*1000, surfaces_m[k][1]) for k in SURFACE_NAMES}
    center_mm, radii_mm = joint_sphere_center(surfaces_mm)
    coefficients = {k: fit_log_radius(surfaces_mm[k][0], center_mm, radius_scale_m*1000, order) for k in SURFACE_NAMES}
    return coefficients, center_mm/1000, radii_mm/1000


def common_average(array, channel_axis=0):
    a = np.asarray(array, dtype=float)
    return a-a.mean(axis=channel_axis, keepdims=True)


def build_leadfield(coefficients, center_m, radius_scale_m, sources_m, electrodes_m,
                    geometry_order=12, field_order=12, scanner_to_surface=None,
                    batch_size=64, compare_permuted_system=False):
    """Assemble once and solve multiple exact Cartesian source right-hand sides.

    The coordinate transform must describe the geometry coefficient frame
    and supplied arrays consistently. It is not an implicit registration.
    """
    transform = np.eye(4) if scanner_to_surface is None else np.asarray(scanner_to_surface)
    sources_m = np.asarray(sources_m, dtype=float)
    electrodes_m = np.asarray(electrodes_m, dtype=float)
    if sources_m.ndim != 2 or sources_m.shape[1] != 3 or electrodes_m.ndim != 2 or electrodes_m.shape[1] != 3:
        raise ValueError('source/electrode coordinates must have shape (n,3)')
    Operator, _, _, _ = _operator_class()
    op = Operator(coefficients, geometry_order, field_order, np.asarray(center_m)*1000,
                  radius_scale_m*1000, sphere_quadrature(4*field_order+16, 8*field_order+32),
                  scanner_to_surface=transform)
    gain, details = op.leadfield(sources_m, electrodes_m, realization='dgm', batch_size=batch_size)
    # The retained numerical kernel returns source x electrode x XYZ.
    gain = gain.transpose(1,0,2).reshape(len(electrodes_m),-1)
    gain = common_average(gain)
    report = dict(solve=details, operator=op.diagnostics.__dict__, source_injection=op.source_injection_diagnostics)
    if compare_permuted_system:
        other, _ = op.leadfield(sources_m, electrodes_m, realization='tmm', batch_size=batch_size)
        other = other.transpose(1,0,2).reshape(len(electrodes_m),-1)
        other = common_average(other)
        report['permuted_relative_error'] = float(np.linalg.norm(gain-other)/max(np.linalg.norm(gain),1e-300))
    return gain, report


def replace_mne_leadfield(template_forward, gain):
    """Use an existing matching MNE forward as metadata container.

    The template must use identical channels, Cartesian source positions and
    source order. This preserves MNE coordinate/orientation metadata.
    """
    fwd = copy.deepcopy(template_forward)
    gain = np.asarray(gain, dtype=float)
    if gain.shape != fwd['_orig_sol'].shape:
        raise ValueError('gain must match the original Cartesian template matrix')
    if fwd.get('surf_ori', False):
        raise ValueError('pass a Cartesian template (surf_ori=False)')
    fwd['sol']['data'] = gain.copy()
    fwd['_orig_sol'] = gain.copy()
    fwd['sol_grad'] = None
    fwd['_orig_sol_grad'] = None
    return fwd
