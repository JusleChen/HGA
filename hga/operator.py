"""Finite-deformation spherical-harmonic three-shell operator.

The implementation expands the Laplace potential in each of the three
conductive regions (brain, skull and scalp) in regular/decaying solid
harmonics.  On each fitted finite-deformation boundary it projects the
potential and normal-current interface equations onto the complete real SH
basis.  The resulting 5N-by-5N system (N=(L+1)^2) is global and contains the
full (l,m) coupling induced by the boundary geometry.

FD-SHA-TMM is the same finite-deformation system under a fixed block
permutation.  This makes the control an algebraically equivalent stabilized
realization rather than a deliberately weaker approximation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .harmonics import CONDUCTIVITY, real_sh, radial_graph_geometry, xyz_to_angles


def _solid_basis(points_u: np.ndarray, L: int, decaying: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return dimensionless solid-harmonic values and Cartesian gradients."""
    points_u = np.asarray(points_u, dtype=float)
    radius, theta, phi = xyz_to_angles(points_u)
    if np.any(radius <= 1e-12):
        raise ValueError("solid harmonics cannot be evaluated at the origin")
    Y, dtheta, dphi = real_sh(theta, phi, L, derivatives=True)
    from .harmonics import sh_index

    er = points_u / radius[:, None]
    st = np.sin(theta)
    et = np.column_stack((np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -st))
    ep = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)))
    values = np.empty_like(Y)
    gradients = np.empty((len(points_u), Y.shape[1], 3), dtype=float)
    for j, (ell, _) in enumerate(sh_index(L)):
        exponent = -(ell + 1) if decaying else ell
        radial_power = np.power(radius, exponent)
        gradient_factor = np.power(radius, exponent - 1)
        values[:, j] = radial_power * Y[:, j]
        gradients[:, j, :] = gradient_factor[:, None] * (
            exponent * Y[:, j, None] * er
            + dtheta[:, j, None] * et
            + (dphi[:, j] / np.maximum(st, 1e-12))[:, None] * ep
        )
    return values, gradients


def _projection(test_basis: np.ndarray, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return test_basis.T @ (weights[:, None] * values)


@dataclass
class OperatorDiagnostics:
    L_field: int
    L_geometry: int
    n_quadrature: int
    raw_condition_number: float
    equilibrated_condition_number: float
    matrix_rank: int
    matrix_size: int
    max_matrix_entry: float
    coupling_offdiagonal_ratio: dict[str, float]
    build_runtime_s: float
    memory_rss_gb: float | None


class FDSHAOperator:
    """Global finite-deformation modal operator for one head."""

    def __init__(
        self,
        geometry_coefficients: dict[str, np.ndarray],
        L_geometry: int,
        L_field: int,
        center_mm: np.ndarray,
        radius_scale_mm: float,
        quadrature: dict[str, np.ndarray],
        ridge: float = 0.0,
        surface_quadratures: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> None:
        started = time.perf_counter()
        self.geometry_coefficients = {k: np.asarray(v, dtype=float) for k, v in geometry_coefficients.items()}
        self.L_geometry = int(L_geometry)
        self.L_field = int(L_field)
        self.center_mm = np.asarray(center_mm, dtype=float)
        self.radius_scale_mm = float(radius_scale_mm)
        self.center_m = self.center_mm / 1000.0
        self.radius_scale_m = self.radius_scale_mm / 1000.0
        self.quadrature = quadrature
        self.ridge = float(ridge)
        self.surface_quadratures = surface_quadratures
        self.N = (self.L_field + 1) ** 2
        self.block_size = 5 * self.N
        self.test_basis = real_sh(quadrature["theta"], quadrature["phi"], self.L_field)
        self.boundaries: list[dict[str, np.ndarray]] = []
        for name in ("inner_skull", "outer_skull", "outer_skin"):
            if surface_quadratures is None:
                graph = radial_graph_geometry(self.geometry_coefficients[name], self.L_geometry, quadrature)
                rho = graph["radius"]
                points_u = rho[:, None] * quadrature["er"]
                normal = graph["normal"]
                graph_jacobian_u = graph["jacobian"] / np.maximum(quadrature["sin_theta"], 1e-12)
                integration_weights = quadrature["weights"]
                test_basis = self.test_basis
                mode = "single_center_radial_graph"
            else:
                supplied = surface_quadratures[name]
                points_u = np.asarray(supplied["points_u"], dtype=float)
                normal = np.asarray(supplied["normal"], dtype=float)
                integration_weights = np.asarray(supplied["weights"], dtype=float)
                if points_u.ndim != 2 or points_u.shape[1] != 3:
                    raise ValueError(f"MC surface points for {name} must be (n, 3)")
                if normal.shape != points_u.shape or integration_weights.shape != (len(points_u),):
                    raise ValueError(f"MC surface quadrature shape mismatch for {name}")
                rho, node_theta, node_phi = xyz_to_angles(points_u)
                if np.any(rho <= 1e-12) or not np.isfinite(rho).all():
                    raise ValueError(f"MC surface points for {name} contain the expansion center")
                test_basis = real_sh(node_theta, node_phi, self.L_field)
                graph_jacobian_u = np.ones_like(integration_weights)
                mode = "mc_pou_mesh_surface"
            regular, regular_grad = _solid_basis(points_u, self.L_field, decaying=False)
            decaying, decaying_grad = _solid_basis(points_u, self.L_field, decaying=True)
            regular_normal = np.einsum("qni,qi->qn", regular_grad, normal)
            decaying_normal = np.einsum("qni,qi->qn", decaying_grad, normal)
            self.boundaries.append({
                "name": np.asarray([name]),
                "graph_jacobian": graph_jacobian_u,
                "graph_jacobian_u": graph_jacobian_u,
                "integration_weights": integration_weights,
                "normal": normal,
                "points_u": points_u,
                "test_basis": test_basis,
                "regular": regular,
                "decaying": decaying,
                "regular_normal": regular_normal,
                "decaying_normal": decaying_normal,
                "radius": rho,
                "surface_quadrature_mode": mode,
            })
        self.coupling_matrices = self._build_coupling_matrices()
        self.A = self._assemble_matrix()
        if self.ridge:
            self.A += self.ridge * np.eye(self.block_size)
        self.row_scale, self.col_scale, self.Aeq = self._equilibrate(self.A)
        self.raw_condition_number = float(np.linalg.cond(self.A))
        self.equilibrated_condition_number = float(np.linalg.cond(self.Aeq))
        self.matrix_rank = int(np.linalg.matrix_rank(self.Aeq))
        self._factor_dgm()
        self._make_tmm_permutation()
        self._factor_tmm()
        self.diagnostics = OperatorDiagnostics(
            L_field=self.L_field,
            L_geometry=self.L_geometry,
            n_quadrature=int(max(len(boundary["integration_weights"]) for boundary in self.boundaries)),
            raw_condition_number=self.raw_condition_number,
            equilibrated_condition_number=self.equilibrated_condition_number,
            matrix_rank=self.matrix_rank,
            matrix_size=self.block_size,
            max_matrix_entry=float(np.max(np.abs(self.A))),
            coupling_offdiagonal_ratio={
                name: float(np.linalg.norm(k - np.diag(np.diag(k))) / max(np.linalg.norm(k), 1e-30))
                for name, k in self.coupling_matrices.items()
            },
            build_runtime_s=float(time.perf_counter() - started),
            memory_rss_gb=self._memory_rss_gb(),
        )

    @staticmethod
    def _memory_rss_gb() -> float | None:
        try:
            import psutil

            return float(psutil.Process().memory_info().rss / 1024**3)
        except Exception:
            return None

    def _build_coupling_matrices(self) -> dict[str, np.ndarray]:
        return {
            boundary["name"][0]: boundary["test_basis"].T @ (
                (boundary["integration_weights"] * boundary["graph_jacobian_u"])[:, None] * boundary["test_basis"]
            )
            for boundary in self.boundaries
        }

    def _interface_blocks(self, boundary: dict[str, np.ndarray], sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        weights = boundary["integration_weights"] * boundary["graph_jacobian_u"]
        potential_scale = self.radius_scale_m**2
        current_scale = sigma * self.radius_scale_m
        test_basis = boundary["test_basis"]
        p_reg = _projection(test_basis, boundary["regular"], weights * potential_scale)
        p_dec = _projection(test_basis, boundary["decaying"], weights * potential_scale)
        j_reg = _projection(test_basis, boundary["regular_normal"], weights * current_scale)
        j_dec = _projection(test_basis, boundary["decaying_normal"], weights * current_scale)
        return p_reg, p_dec, j_reg, j_dec

    def _assemble_matrix(self) -> np.ndarray:
        N = self.N
        A = np.zeros((self.block_size, self.block_size), dtype=float)
        # The same modal field is evaluated on different interfaces for each
        # region.  In particular, the skull basis at the inner boundary is
        # not the same block as the skull basis at the outer boundary.
        inner_brain = self._interface_blocks(self.boundaries[0], CONDUCTIVITY[0])
        inner_skull = self._interface_blocks(self.boundaries[0], CONDUCTIVITY[1])
        outer_skull = self._interface_blocks(self.boundaries[1], CONDUCTIVITY[1])
        outer_scalp = self._interface_blocks(self.boundaries[1], CONDUCTIVITY[2])
        skin_scalp = self._interface_blocks(self.boundaries[2], CONDUCTIVITY[2])
        inner_p_r, _, inner_j_r, _ = inner_brain
        inner_skull_p_r, inner_skull_p_d, inner_skull_j_r, inner_skull_j_d = inner_skull
        outer_skull_p_r, outer_skull_p_d, outer_skull_j_r, outer_skull_j_d = outer_skull
        outer_scalp_p_r, outer_scalp_p_d, outer_scalp_j_r, outer_scalp_j_d = outer_scalp
        skin_p_r, skin_p_d, skin_j_r, skin_j_d = skin_scalp

        # Unknown blocks: A_brain_regular, B_skull_regular, B_skull_decaying,
        # C_scalp_regular, C_scalp_decaying.
        # Inner skull: V_brain = V_skull and sigma*dV_brain = sigma*dV_skull.
        A[0:N, 0:N] = inner_p_r
        A[0:N, N:2 * N] = -inner_skull_p_r
        A[0:N, 2 * N:3 * N] = -inner_skull_p_d
        A[N:2 * N, 0:N] = inner_j_r
        A[N:2 * N, N:2 * N] = -inner_skull_j_r
        A[N:2 * N, 2 * N:3 * N] = -inner_skull_j_d

        # Outer skull: V_skull = V_scalp and current is continuous.
        A[2 * N:3 * N, N:2 * N] = outer_skull_p_r
        A[2 * N:3 * N, 2 * N:3 * N] = outer_skull_p_d
        A[2 * N:3 * N, 3 * N:4 * N] = -outer_scalp_p_r
        A[2 * N:3 * N, 4 * N:5 * N] = -outer_scalp_p_d
        A[3 * N:4 * N, N:2 * N] = outer_skull_j_r
        A[3 * N:4 * N, 2 * N:3 * N] = outer_skull_j_d
        A[3 * N:4 * N, 3 * N:4 * N] = -outer_scalp_j_r
        A[3 * N:4 * N, 4 * N:5 * N] = -outer_scalp_j_d

        # Scalp-air boundary: zero normal current.  Replace its constant-mode
        # Neumann row with a zero-mean scalp gauge to remove the Laplace null.
        A[4 * N:5 * N, 3 * N:4 * N] = skin_j_r
        A[4 * N:5 * N, 4 * N:5 * N] = skin_j_d
        gauge_test = self.boundaries[2]["test_basis"][:, 0:1]
        gauge_weights = self.boundaries[2]["integration_weights"] * self.boundaries[2]["graph_jacobian_u"] * self.radius_scale_m**2
        gauge_p_r = _projection(gauge_test, self.boundaries[2]["regular"], gauge_weights)
        gauge_p_d = _projection(gauge_test, self.boundaries[2]["decaying"], gauge_weights)
        A[4 * N, :] = 0.0
        A[4 * N, 3 * N:4 * N] = gauge_p_r[0]
        A[4 * N, 4 * N:5 * N] = gauge_p_d[0]
        return A

    @staticmethod
    def _equilibrate(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row_norm = np.maximum(np.linalg.norm(A, axis=1), 1e-30)
        col_norm = np.maximum(np.linalg.norm(A, axis=0), 1e-30)
        row_scale = 1.0 / row_norm
        col_scale = 1.0 / col_norm
        return row_scale, col_scale, row_scale[:, None] * A * col_scale[None, :]

    def _factor_dgm(self) -> None:
        from scipy.linalg import lu_factor

        self._lu_dgm = lu_factor(self.Aeq, check_finite=False)

    def _make_tmm_permutation(self) -> None:
        # A fixed interface/region block reordering realizes the same global
        # finite-deformation equations in a transfer-style ordering.
        block_order = (3, 4, 2, 1, 0)
        self._permutation = np.concatenate([np.arange(i * self.N, (i + 1) * self.N) for i in block_order])

    def _factor_tmm(self) -> None:
        from scipy.linalg import lu_factor

        p = self._permutation
        self._lu_tmm = lu_factor(self.Aeq[np.ix_(p, p)], check_finite=False)

    def _solve(self, rhs: np.ndarray, realization: str) -> tuple[np.ndarray, float]:
        from scipy.linalg import lu_solve

        # rhs shape is (5N, batch, 3).
        rhs_flat = np.asarray(rhs, dtype=float).reshape(self.block_size, -1)
        rhs_eq = self.row_scale[:, None] * rhs_flat
        if realization == "dgm":
            z = lu_solve(self._lu_dgm, rhs_eq, check_finite=False)
        elif realization == "tmm":
            p = self._permutation
            z_perm = lu_solve(self._lu_tmm, rhs_eq[p], check_finite=False)
            z = np.empty_like(z_perm)
            z[p] = z_perm
        else:
            raise ValueError(realization)
        solution = self.col_scale[:, None] * z
        residual = self.Aeq @ z - rhs_eq
        backward_error = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs_eq), 1e-30))
        return solution.reshape(self.block_size, rhs.shape[1], rhs.shape[2]), backward_error

    def _source_rhs(self, sources_m: np.ndarray) -> np.ndarray:
        """Project the dipole particular solution at the inner boundary."""
        sources_m = np.asarray(sources_m, dtype=float)
        inner = self.boundaries[0]
        x_m = self.center_m[None, :] + self.radius_scale_m * inner["points_u"]
        n = inner["normal"]
        d = x_m[:, None, :] - sources_m[None, :, :]
        distance = np.linalg.norm(d, axis=2)
        if np.any(distance <= 1e-8):
            raise RuntimeError("source too close to inner boundary for stable modal projection")
        inv_r3 = 1.0 / np.maximum(distance**3, 1e-30)
        inv_r5 = 1.0 / np.maximum(distance**5, 1e-30)
        q = d * inv_r3[:, :, None] / (4.0 * np.pi * CONDUCTIVITY[0])
        nd = np.einsum("qi,qbi->qb", n, d)
        grad_dot_n = (n[:, None, :] * inv_r3[:, :, None] - 3.0 * nd[:, :, None] * d * inv_r5[:, :, None]) / (4.0 * np.pi * CONDUCTIVITY[0])
        weights_area = inner["integration_weights"] * inner["graph_jacobian_u"] * self.radius_scale_m**2
        rhs_v = -np.einsum("qn,qbo->nbo", inner["test_basis"], weights_area[:, None, None] * q)
        rhs_j = -np.einsum("qn,qbo->nbo", inner["test_basis"], weights_area[:, None, None] * CONDUCTIVITY[0] * grad_dot_n)
        rhs = np.zeros((self.block_size, len(sources_m), 3), dtype=float)
        rhs[0:self.N] = rhs_v
        rhs[self.N:2 * self.N] = rhs_j
        return rhs

    def leadfield(
        self,
        sources_m: np.ndarray,
        electrodes_m: np.ndarray,
        realization: str = "dgm",
        batch_size: int = 128,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        sources_m = np.asarray(sources_m, dtype=float)
        electrodes_m = np.asarray(electrodes_m, dtype=float)
        if electrodes_m.ndim != 2 or electrodes_m.shape[1] != 3:
            raise ValueError("electrodes_m must be (channels, 3)")
        electrode_u = (electrodes_m - self.center_m[None, :]) / self.radius_scale_m
        electrode_regular, _ = _solid_basis(electrode_u, self.L_field, decaying=False)
        electrode_decaying, _ = _solid_basis(electrode_u, self.L_field, decaying=True)
        output = np.empty((len(sources_m), len(electrodes_m), 3), dtype=float)
        max_backward_error = 0.0
        started = time.perf_counter()
        for start in range(0, len(sources_m), int(batch_size)):
            stop = min(start + int(batch_size), len(sources_m))
            rhs = self._source_rhs(sources_m[start:stop])
            solution, backward_error = self._solve(rhs, realization)
            max_backward_error = max(max_backward_error, backward_error)
            c_reg = solution[3 * self.N:4 * self.N]
            c_dec = solution[4 * self.N:5 * self.N]
            values = np.einsum("mn,nbo->mbo", electrode_regular, c_reg) + np.einsum("mn,nbo->mbo", electrode_decaying, c_dec)
            output[start:stop] = np.transpose(values, (1, 0, 2))
        output -= output.mean(axis=1, keepdims=True)
        metadata = {
            "realization": realization,
            "source_count": int(len(sources_m)),
            "electrode_count": int(len(electrodes_m)),
            "batch_size": int(batch_size),
            "runtime_s": float(time.perf_counter() - started),
            "max_backward_error": float(max_backward_error),
            "finite": bool(np.isfinite(output).all()),
            "common_average_reference": True,
        }
        return output, metadata


def modal_surface_potential(operator: FDSHAOperator, boundary_index: int = 0) -> np.ndarray:
    boundary = operator.boundaries[int(boundary_index)]
    return operator.coupling_matrices[boundary["name"][0]]
