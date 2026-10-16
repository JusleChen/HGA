"""Frame-covariant finite-quadrature primary-field fallback."""
from __future__ import annotations
from typing import Any
import numpy as np
def _load_frozen_modules():
    from . import source, harmonics
    return source, harmonics

def validate_transform(scanner_to_surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(scanner_to_surface, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("scanner_to_surface must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-14):
        raise ValueError("scanner_to_surface has an invalid homogeneous last row")
    q = matrix[:3, :3]
    if not np.allclose(q.T @ q, np.eye(3), rtol=0.0, atol=2e-12):
        raise ValueError("scanner_to_surface linear component is not orthogonal")
    determinant = float(np.linalg.det(q))
    if abs(abs(determinant) - 1.0) > 2e-12:
        raise ValueError("scanner_to_surface determinant is not +/-1")
    return q, matrix[:3, 3].copy()


def transformed_quadrature(
    geometry_coefficients: dict[str, np.ndarray],
    L_geometry: int,
    L_field: int,
    center_current_m: np.ndarray,
    radius_scale_m: float,
    canonical_quadrature: dict[str, np.ndarray],
    scanner_to_surface: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the same physical nodes as the canonical grid in current axes.

    The canonical directions are transformed as vectors. Points are rebuilt
    about the inverse-transformed center and then transformed with Qx+t;
    normals are evaluated from the same transformed radial graph in current
    axes, which is equivalent to Q*n and remains valid for det(Q)=-1.
    """
    frozen, common = _load_frozen_modules()
    q, translation_m = validate_transform(scanner_to_surface)
    er_common = np.asarray(canonical_quadrature["er"], dtype=np.float64)
    er_current = er_common @ q.T
    _, theta_current, phi_current = common.xyz_to_angles(er_current)
    sin_theta = np.sin(theta_current)
    et_current = np.column_stack((
        np.cos(theta_current) * np.cos(phi_current),
        np.cos(theta_current) * np.sin(phi_current),
        -sin_theta,
    ))
    ep_current = np.column_stack((
        -np.sin(phi_current),
        np.cos(phi_current),
        np.zeros_like(phi_current),
    ))
    quad_current = {
        "theta": theta_current,
        "phi": phi_current,
        "weights": np.asarray(canonical_quadrature["weights"], dtype=np.float64),
        "er": er_current,
        "et": et_current,
        "ep": ep_current,
        "sin_theta": sin_theta,
        "n_theta": np.asarray(canonical_quadrature.get("n_theta", [0]), dtype=np.int64),
        "n_phi": np.asarray(canonical_quadrature.get("n_phi", [0]), dtype=np.int64),
    }
    graph = common.radial_graph_geometry(
        np.asarray(geometry_coefficients["inner_skull"], dtype=np.float64),
        int(L_geometry),
        quad_current,
    )
    center_current = np.asarray(center_current_m, dtype=np.float64).reshape(3)
    center_common = (center_current - translation_m) @ q
    points_common = center_common[None, :] + float(radius_scale_m) * graph["radius"][:, None] * er_common
    points_current = points_common @ q.T + translation_m[None, :]
    points_u_current = (points_current - center_current[None, :]) / float(radius_scale_m)
    graph_jacobian_u = graph["jacobian"] / np.maximum(sin_theta, 1e-12)
    test_basis = common.real_sh(theta_current, phi_current, int(L_field))
    return {
        "points_current_m": points_current,
        "points_u": points_u_current,
        "normal_current": graph["normal"],
        "test_basis": test_basis,
        "weights": quad_current["weights"],
        "graph_jacobian_u": graph_jacobian_u,
        "theta_current": theta_current,
        "phi_current": phi_current,
        "er_current": er_current,
    }


def coordinate_covariant_direct_fallback_rhs(
    *,
    sources_current_m: np.ndarray,
    invalid_indices: np.ndarray,
    geometry_coefficients: dict[str, np.ndarray],
    center_current_m: np.ndarray,
    radius_scale_m: float,
    quadrature: dict[str, np.ndarray],
    scanner_to_surface: np.ndarray,
    L_geometry: int = 12,
    L_field: int = 12,
    conductivity_s_per_m: float = 0.33,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return only the direct-primary fallback RHS for selected sources."""
    frozen, _ = _load_frozen_modules()
    indices = np.asarray(invalid_indices, dtype=np.int64)
    if not len(indices):
        n = (int(L_field) + 1) ** 2
        empty = np.zeros((n, 0, 3), dtype=np.float64)
        return empty, empty.copy(), {"fallback_source_count": 0}
    current_q = transformed_quadrature(
        geometry_coefficients,
        int(L_geometry),
        int(L_field),
        center_current_m,
        float(radius_scale_m),
        quadrature,
        scanner_to_surface,
    )
    sources = np.asarray(sources_current_m, dtype=np.float64)[indices]
    d = current_q["points_current_m"][:, None, :] - sources[None, :, :]
    distance = np.linalg.norm(d, axis=2)
    if np.any(distance <= 1e-12):
        raise RuntimeError("source/interface coincidence in coordinate-covariant direct fallback")
    inv_r3 = 1.0 / np.maximum(distance**3, 1e-300)
    inv_r5 = 1.0 / np.maximum(distance**5, 1e-300)
    primary = d * inv_r3[:, :, None] / (4.0 * np.pi * float(conductivity_s_per_m))
    normal = current_q["normal_current"]
    nd = np.einsum("qi,qbi->qb", normal, d)
    primary_grad = (
        normal[:, None, :] * inv_r3[:, :, None]
        - 3.0 * nd[:, :, None] * d * inv_r5[:, :, None]
    ) / (4.0 * np.pi * float(conductivity_s_per_m))
    weights = current_q["weights"] * current_q["graph_jacobian_u"]
    area_weights = weights[:, None, None] * float(radius_scale_m) ** 2
    rhs_v = -np.einsum("qn,qbo->nbo", current_q["test_basis"], area_weights * primary)
    rhs_j = -np.einsum(
        "qn,qbo->nbo",
        current_q["test_basis"],
        area_weights * float(conductivity_s_per_m) * primary_grad,
    )
    return rhs_v, rhs_j, {
        "fallback_source_count": int(len(indices)),
        "quadrature_node_count": int(len(current_q["weights"])),
        "point_transform": "Qx+t",
        "vector_normal_transform": "Qv; translation not applied",
        "quadrature_weights_unchanged": True,
        "determinant": float(np.linalg.det(np.asarray(scanner_to_surface)[:3, :3])),
    }
