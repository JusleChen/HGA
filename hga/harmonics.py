"""Real harmonics and fitted log-radial head boundaries."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
SURFACE_NAMES = ('inner_skull', 'outer_skull', 'outer_skin')
CONDUCTIVITY = np.asarray([0.33, 0.013, 0.33], dtype=float)

def sh_index(L: int) -> list[tuple[int, int]]:
    return [(ell, emm) for ell in range(L + 1) for emm in range(-ell, ell + 1)]


def _real_component(value: np.ndarray, deriv: np.ndarray, emm: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if emm == 0:
        return value.real, deriv[:, 0].real, deriv[:, 1].real
    if emm > 0:
        scale = math.sqrt(2.0)
        return scale * value.real, scale * deriv[:, 0].real, scale * deriv[:, 1].real
    scale = math.sqrt(2.0)
    return scale * value.imag, scale * deriv[:, 0].imag, scale * deriv[:, 1].imag


def real_sh(theta: np.ndarray, phi: np.ndarray, L: int, derivatives: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
    """Orthonormal real SH basis using scipy's colatitude/longitude convention.

    For m>0 the basis is sqrt(2)*Re(Y_l^m); for m<0 it is
    sqrt(2)*Im(Y_l^{|m|}); m=0 is Re(Y_l^0).  The same convention is used
    for geometry fits, surface blocks and source/electrode evaluation.
    """
    from scipy.special import sph_harm_y

    theta = np.asarray(theta, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    values: list[np.ndarray] = []
    dtheta: list[np.ndarray] = []
    dphi: list[np.ndarray] = []
    for ell, emm in sh_index(L):
        m_abs = abs(emm)
        if derivatives:
            val, deriv = sph_harm_y(ell, m_abs, theta, phi, diff_n=1)
            v, dt, dp = _real_component(np.asarray(val), np.asarray(deriv), emm)
            dtheta.append(dt)
            dphi.append(dp)
        else:
            val = np.asarray(sph_harm_y(ell, m_abs, theta, phi), dtype=complex)
            v, _, _ = _real_component(val, np.zeros((len(theta), 2), dtype=complex), emm)
        values.append(v)
    Y = np.column_stack(values)
    if derivatives:
        return Y, np.column_stack(dtheta), np.column_stack(dphi)
    return Y


def xyz_to_angles(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    radius = np.linalg.norm(points, axis=1)
    direction = points / np.maximum(radius[:, None], 1e-30)
    theta = np.arccos(np.clip(direction[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(direction[:, 1], direction[:, 0]), 2.0 * np.pi)
    return radius, theta, phi


def sphere_quadrature(n_theta: int, n_phi: int) -> dict[str, np.ndarray]:
    """Tensor Gauss-Legendre x uniform-phi spherical quadrature.

    Gauss-Legendre integrates polynomials in cos(theta), while the uniform
    longitude rule resolves Fourier modes.  The selected order is frozen only
    after comparing successive rules in the outcome-blind convergence stage.
    """
    from numpy.polynomial.legendre import leggauss

    u, wu = leggauss(int(n_theta))
    phi_1d = np.arange(int(n_phi), dtype=float) * (2.0 * np.pi / int(n_phi))
    uu, pp = np.meshgrid(u, phi_1d, indexing="ij")
    theta = np.arccos(np.clip(uu.ravel(), -1.0, 1.0))
    phi = pp.ravel()
    weights = np.repeat(wu, int(n_phi)) * (2.0 * np.pi / int(n_phi))
    sin_theta = np.sin(theta)
    er = np.column_stack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)))
    et = np.column_stack((np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -sin_theta))
    ep = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)))
    return {
        "theta": theta,
        "phi": phi,
        "weights": weights,
        "er": er,
        "et": et,
        "ep": ep,
        "sin_theta": sin_theta,
        "n_theta": np.asarray([n_theta], dtype=np.int64),
        "n_phi": np.asarray([n_phi], dtype=np.int64),
    }


def joint_sphere_center(surfaces: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    from scipy.optimize import least_squares

    all_vertices = [np.asarray(surfaces[name][0], dtype=float) for name in SURFACE_NAMES]
    initial_center = np.mean(np.vstack(all_vertices), axis=0)
    initial_radii = np.asarray([np.median(np.linalg.norm(v - initial_center, axis=1)) for v in all_vertices])

    def residual(parameters: np.ndarray) -> np.ndarray:
        center = parameters[:3]
        radii = parameters[3:]
        return np.concatenate([np.linalg.norm(v - center[None, :], axis=1) - radii[i] for i, v in enumerate(all_vertices)])

    result = least_squares(
        residual,
        np.r_[initial_center, initial_radii],
        method="trf",
        loss="linear",
        x_scale="jac",
        max_nfev=100,
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"geometry-only joint sphere center failed: {result.message}")
    return np.asarray(result.x[:3], dtype=float), np.asarray(result.x[3:], dtype=float)


def fit_log_radius(vertices: np.ndarray, center_mm: np.ndarray, radius_scale_mm: float, L: int, max_fit_points: int = 4096) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) > int(max_fit_points):
        # Deterministic uniform vertex-index thinning keeps the fit geometry
        # only while avoiding an expensive dense least-squares solve.  All
        # vertices are still used for the reconstruction/error audit.
        indices = np.linspace(0, len(vertices) - 1, int(max_fit_points), dtype=int)
        vertices = vertices[np.unique(indices)]
    shifted = vertices - np.asarray(center_mm, dtype=float)[None, :]
    radius, theta, phi = xyz_to_angles(shifted)
    Y = real_sh(theta, phi, L)
    coefficients, *_ = np.linalg.lstsq(Y, np.log(np.maximum(radius, 1e-9) / float(radius_scale_mm)), rcond=None)
    return np.asarray(coefficients, dtype=float)


def eval_log_radius(coefficients: np.ndarray, theta: np.ndarray, phi: np.ndarray, L: int, derivatives: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
    if derivatives:
        Y, dtheta, dphi = real_sh(theta, phi, L, derivatives=True)
        log_r = Y @ coefficients
        return np.exp(log_r), np.exp(log_r) * (dtheta @ coefficients), np.exp(log_r) * (dphi @ coefficients)
    return np.exp(real_sh(theta, phi, L) @ coefficients)


def radial_graph_geometry(coefficients: np.ndarray, L: int, quadrature: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    r, rt, rp = eval_log_radius(coefficients, quadrature["theta"], quadrature["phi"], L, derivatives=True)
    er, et, ep = quadrature["er"], quadrature["et"], quadrature["ep"]
    sin_theta = quadrature["sin_theta"]
    xt = rt[:, None] * er + r[:, None] * et
    xp = rp[:, None] * er + r[:, None] * sin_theta[:, None] * ep
    cross = np.cross(xt, xp)
    jacobian = np.linalg.norm(cross, axis=1)
    normal = cross / np.maximum(jacobian[:, None], 1e-30)
    outward = np.sum(normal * er, axis=1) < 0
    normal[outward] *= -1.0
    metric11 = np.einsum("ij,ij->i", xt, xt)
    metric12 = np.einsum("ij,ij->i", xt, xp)
    metric22 = np.einsum("ij,ij->i", xp, xp)
    return {
        "radius": r,
        "radius_theta": rt,
        "radius_phi": rp,
        "tangent_theta": xt,
        "tangent_phi": xp,
        "metric11": metric11,
        "metric12": metric12,
        "metric22": metric22,
        "jacobian": jacobian,
        "normal": normal,
        "er": er,
    }


def fitted_normals(points_from_center: np.ndarray, coefficients: np.ndarray, L: int) -> np.ndarray:
    _, theta, phi = xyz_to_angles(points_from_center)
    q = sphere_quadrature(max(4, len(theta)), 1) if False else None
    # Reuse the exact spherical tangent frame at the input directions.
    radius, theta, phi = xyz_to_angles(points_from_center)
    st = np.sin(theta)
    er = np.column_stack((st * np.cos(phi), st * np.sin(phi), np.cos(theta)))
    et = np.column_stack((np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -st))
    ep = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)))
    r, rt, rp = eval_log_radius(coefficients, theta, phi, L, derivatives=True)
    xt = rt[:, None] * er + r[:, None] * et
    xp = rp[:, None] * er + r[:, None] * st[:, None] * ep
    n = np.cross(xt, xp)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    if float(np.median(np.sum(n * er, axis=1))) < 0:
        n *= -1.0
    return n
