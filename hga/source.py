"""Analytic Cartesian source injection into the shared harmonic system."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from .harmonics import real_sh, sh_index, sphere_quadrature, xyz_to_angles, CONDUCTIVITY
from .operator import FDSHAOperator
SIGMAS = CONDUCTIVITY

def fast_real_sh(theta: np.ndarray, phi: np.ndarray, L: int, derivatives: bool = False) -> Any:
    """Vectorized equivalent of the frozen real SH convention.

    The frozen helper evaluates one mode at a time, which is convenient for
    the historical L=12 operator but becomes the dominant cost in the R4.3G
    source-depth sweep.  This routine evaluates all (l,m) modes in one SciPy
    call while preserving the frozen m>0 real-part and m<0 imaginary-part
    convention.
    """
    from scipy.special import sph_harm_y

    theta = np.asarray(theta, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    labels = sh_index(int(L))
    ell = np.asarray([item[0] for item in labels], dtype=int)[:, None]
    m_abs = np.asarray([abs(item[1]) for item in labels], dtype=int)[:, None]
    values, deriv = sph_harm_y(ell, m_abs, theta[None, :], phi[None, :], diff_n=1)
    values = np.asarray(values, dtype=complex)
    deriv = np.asarray(deriv, dtype=complex)
    Y = np.empty((len(labels), len(theta)), dtype=float)
    dtheta = np.empty_like(Y)
    dphi = np.empty_like(Y)
    for index, (_, emm) in enumerate(labels):
        if emm == 0:
            factor = 1.0
            Y[index] = values[index].real
            dtheta[index] = deriv[index, :, 0].real
            dphi[index] = deriv[index, :, 1].real
        elif emm > 0:
            factor = math.sqrt(2.0)
            Y[index] = factor * values[index].real
            dtheta[index] = factor * deriv[index, :, 0].real
            dphi[index] = factor * deriv[index, :, 1].real
        else:
            factor = math.sqrt(2.0)
            Y[index] = factor * values[index].imag
            dtheta[index] = factor * deriv[index, :, 0].imag
            dphi[index] = factor * deriv[index, :, 1].imag
    if derivatives:
        return Y.T, dtheta.T, dphi.T
    return Y.T


def anti_alias_rule(L: int) -> tuple[int, int]:
    """Pre-registered quadrature rule used for FD-SHA projections."""
    return max(4, 2 * int(L) + 8), max(8, 4 * int(L) + 16)


def _source_frames(source_relative_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radius, theta, phi = xyz_to_angles(np.asarray(source_relative_m, dtype=float))
    st = np.sin(theta)
    er = np.asarray(source_relative_m, dtype=float) / np.maximum(radius[:, None], 1e-300)
    et = np.column_stack((np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -st))
    ep = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)))
    return radius, theta, phi, er, np.stack((et, ep), axis=1)


def source_coefficients_real(
    source_relative_m: np.ndarray,
    moments: np.ndarray | None,
    L: int,
    radius_scale_m: float,
    sigma: float,
) -> np.ndarray:
    """Exact real-SH coefficients of the eccentric dipole primary field.

    The expansion is

        phi_primary(u) = sum_lm c_lm u^(-(l+1)) Y^R_lm(u_hat),

    where u=(x-center)/R and

        c_lm = p dot grad_u[u0^l Y^R_lm(u0_hat)]
               / (sigma R^2 (2l+1)).

    This is the real-basis form of the Cartesian dipole addition theorem.  It
    is source injection, not a fit to boundary samples.
    """
    source_relative_m = np.asarray(source_relative_m, dtype=float)
    if source_relative_m.ndim == 1:
        source_relative_m = source_relative_m[None, :]
    if source_relative_m.shape[1] != 3:
        raise ValueError("source_relative_m must have shape (n,3)")
    if moments is None:
        moments = np.broadcast_to(np.eye(3)[None, :, :], (len(source_relative_m), 3, 3)).copy()
    else:
        moments = np.asarray(moments, dtype=float)
        if moments.ndim == 1:
            moments = np.broadcast_to(moments[None, :], (len(source_relative_m), 3)).copy()
        if moments.shape == (len(source_relative_m), 3):
            moments = moments[:, None, :]
        if moments.shape[0] != len(source_relative_m) or moments.shape[2] != 3:
            raise ValueError("moments must be (n,3) or (n,n_orientation,3)")
    if float(radius_scale_m) <= 0.0:
        raise ValueError("radius_scale_m must be positive")
    # The gradient in the coefficient formula is with respect to the
    # dimensionless source coordinate u0=(r0-center)/R.  Keeping this
    # conversion explicit is important: using the physical radius here would
    # accidentally apply R^(-2) for every degree instead of the required
    # R^(-(ell+1)) conversion from physical to dimensionless coefficients.
    source_u = source_relative_m / float(radius_scale_m)
    radius, theta, phi, er, tangents = _source_frames(source_u)
    Y, dtheta, dphi = fast_real_sh(theta, phi, int(L), derivatives=True)
    n_orientation = moments.shape[1]
    coefficients = np.zeros((len(source_relative_m), Y.shape[1], n_orientation), dtype=float)
    st = np.maximum(np.sin(theta), 1e-14)
    et = tangents[:, 0, :]
    ep = tangents[:, 1, :]
    for mode, (ell, _) in enumerate(sh_index(int(L))):
        if ell == 0:
            continue
        gradient_u = radius[:, None] ** (ell - 1) * (
            ell * Y[:, mode, None] * er
            + dtheta[:, mode, None] * et
            + (dphi[:, mode] / st)[:, None] * ep
        )
        scale = float(sigma) * float(radius_scale_m) ** 2 * float(2 * ell + 1)
        coefficients[:, mode, :] = np.einsum("ni,nki->nk", gradient_u, moments) / scale
    return coefficients


def source_coefficients_physical(source_relative_m: np.ndarray, moments: np.ndarray, L: int, sigma: float) -> np.ndarray:
    """Physical-basis coefficients S_lm for homogeneous dipole tests."""
    return source_coefficients_real(source_relative_m, moments, L, 1.0, sigma)


def primary_dipole_field(observation_m: np.ndarray, source_m: np.ndarray, moment: np.ndarray, sigma: float) -> np.ndarray:
    d = np.asarray(observation_m, dtype=float) - np.asarray(source_m, dtype=float)[None, :]
    dist = np.linalg.norm(d, axis=1)
    return (d @ np.asarray(moment, dtype=float)) / (4.0 * np.pi * float(sigma) * np.maximum(dist, 1e-300) ** 3)


def reconstruct_primary_from_coefficients(
    observation_relative_m: np.ndarray,
    source_relative_m: np.ndarray,
    moment: np.ndarray,
    L: int,
    sigma: float,
) -> np.ndarray:
    obs = np.asarray(observation_relative_m, dtype=float)
    src = np.asarray(source_relative_m, dtype=float).reshape(1, 3)
    coeff = source_coefficients_physical(src, np.asarray(moment, dtype=float).reshape(1, 1, 3), L, sigma)[0, :, 0]
    radius, theta, phi = xyz_to_angles(obs)
    Y = fast_real_sh(theta, phi, int(L))
    out = np.zeros(len(obs), dtype=float)
    for mode, (ell, _) in enumerate(sh_index(int(L))):
        if ell:
            out += coeff[mode] * radius ** (-(ell + 1)) * Y[:, mode]
    return out


class ExactSourceInjectionOperator(FDSHAOperator):
    """Frozen FD-SHA equations with exact analytic primary-source injection."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.source_injection_diagnostics: list[dict[str, Any]] = []

    def _source_rhs(self, sources_m: np.ndarray) -> np.ndarray:  # noqa: D401
        sources_m = np.asarray(sources_m, dtype=float)
        relative = sources_m - self.center_m[None, :]
        source_u = relative / float(self.radius_scale_m)
        source_r = np.linalg.norm(source_u, axis=1)
        inner = self.boundaries[0]
        min_inner = float(np.min(np.linalg.norm(inner["points_u"], axis=1)))
        valid = source_r < (min_inner - 1e-10)
        coeff = source_coefficients_real(relative, None, self.L_field, self.radius_scale_m, float(SIGMAS[0]))
        primary_value = np.einsum("qn,bno->qbo", inner["decaying"], coeff)
        primary_normal = np.einsum("qn,bno->qbo", inner["decaying_normal"], coeff)
        weights = inner["integration_weights"] * inner["graph_jacobian_u"]
        rhs_v = -np.einsum("qn,qbo->nbo", inner["test_basis"], weights[:, None, None] * self.radius_scale_m**2 * primary_value)
        rhs_j = -np.einsum("qn,qbo->nbo", inner["test_basis"], weights[:, None, None] * float(SIGMAS[0]) * self.radius_scale_m * primary_normal)
        rhs = np.zeros((self.block_size, len(sources_m), 3), dtype=float)
        rhs[0:self.N] = rhs_v
        rhs[self.N:2 * self.N] = rhs_j
        direct_count = 0
        # A deformed SHA inner surface is not guaranteed to enclose every
        # frozen source.  In that case the centered multipole series is not a
        # valid representation on the interface.  Use the mathematically
        # equivalent primary/secondary split with the exact Cartesian dipole
        # potential and normal derivative for those sources, on the
        # pre-registered anti-aliased surface quadrature.  No source is moved
        # or discarded and no BEM quantity is changed.
        if np.any(~valid):
            invalid = np.where(~valid)[0]
            x_m = self.center_m[None, :] + self.radius_scale_m * inner["points_u"]
            normal = inner["normal"]
            d = x_m[:, None, :] - sources_m[invalid][None, :, :]
            distance = np.linalg.norm(d, axis=2)
            if np.any(distance <= 1e-12):
                raise RuntimeError("source/interface coincidence in direct primary fallback")
            inv_r3 = 1.0 / np.maximum(distance**3, 1e-300)
            inv_r5 = 1.0 / np.maximum(distance**5, 1e-300)
            primary = d * inv_r3[:, :, None] / (4.0 * np.pi * float(SIGMAS[0]))
            nd = np.einsum("qi,qbi->qb", normal, d)
            primary_grad = (
                normal[:, None, :] * inv_r3[:, :, None]
                - 3.0 * nd[:, :, None] * d * inv_r5[:, :, None]
            ) / (4.0 * np.pi * float(SIGMAS[0]))
            area_weights = weights[:, None, None] * self.radius_scale_m**2
            direct_v = -np.einsum("qn,qbo->nbo", inner["test_basis"], area_weights * primary)
            direct_j = -np.einsum("qn,qbo->nbo", inner["test_basis"], area_weights * float(SIGMAS[0]) * primary_grad)
            rhs[0:self.N, invalid, :] = direct_v
            rhs[self.N:2 * self.N, invalid, :] = direct_j
            direct_count = int(len(invalid))
        self.source_injection_diagnostics.append({
            "source_count": int(len(sources_m)),
            "source_r_min_u": float(np.min(source_r)) if len(source_r) else None,
            "source_r_max_u": float(np.max(source_r)) if len(source_r) else None,
            "inner_r_min_u": min_inner,
            "series_valid_all_sources": bool(np.all(valid)),
            "invalid_source_count": int(np.sum(~valid)),
            "direct_primary_fallback_count": direct_count,
            "method": "exact_analytic_source_coefficients_with_direct_primary_secondary_fallback",
        })
        return rhs
