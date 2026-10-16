"""The final A09 coordinate-covariant HGA operator."""
from __future__ import annotations
from typing import Any
import numpy as np
def _load():
    from . import coordinates, source, harmonics
    return coordinates, source, harmonics

def _frame_quadrature(canonical: dict[str, np.ndarray], transform: np.ndarray) -> dict[str, np.ndarray]:
    """Represent the exact same canonical nodes in the current coordinate axes."""
    fix, frozen, common = _load()
    q, _ = fix.validate_transform(transform)
    er_canonical = np.asarray(canonical["er"], dtype=np.float64)
    er_current = er_canonical @ q.T
    _, theta, phi = common.xyz_to_angles(er_current)
    sin_theta = np.sin(theta)
    et = np.column_stack((np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -sin_theta))
    ep = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)))
    return {
        "theta": theta,
        "phi": phi,
        "weights": np.asarray(canonical["weights"], dtype=np.float64),
        "er": er_current,
        "et": et,
        "ep": ep,
        "sin_theta": sin_theta,
        "n_theta": np.asarray(canonical.get("n_theta", [0]), dtype=np.int64),
        "n_phi": np.asarray(canonical.get("n_phi", [0]), dtype=np.int64),
    }


def _operator_class() -> tuple[Any, Any, Any, Any]:
    fix, frozen, common = _load()

    class A09CovariantOperator(frozen.ExactSourceInjectionOperator):
        def __init__(self, *args: Any, scanner_to_surface: np.ndarray, **kwargs: Any) -> None:
            self.scanner_to_surface = np.asarray(scanner_to_surface, dtype=np.float64)
            self.transform_q, self.transform_t_m = fix.validate_transform(self.scanner_to_surface)
            # Keep canonical nodes for the fallback helper. The superclass and
            # all three interface matrices receive those same nodes expressed
            # in the current frame; no quadrature density/weight is changed.
            self._canonical_quadrature = args[5] if len(args) > 5 else kwargs["quadrature"]
            if not np.array_equal(self.scanner_to_surface, np.eye(4, dtype=np.float64)):
                args = list(args)
                if len(args) > 5:
                    args[5] = _frame_quadrature(self._canonical_quadrature, self.scanner_to_surface)
                else:
                    kwargs["quadrature"] = _frame_quadrature(self._canonical_quadrature, self.scanner_to_surface)
            super().__init__(*args, **kwargs)

        def _source_rhs(self, sources_m: np.ndarray) -> np.ndarray:
            matrix = self.scanner_to_surface
            if np.array_equal(matrix, np.eye(4, dtype=np.float64)):
                # Preserve the canonical identity-frame operation order.
                return super()._source_rhs(sources_m)

            sources_m = np.asarray(sources_m, dtype=np.float64)
            relative = sources_m - self.center_m[None, :]
            source_u = relative / float(self.radius_scale_m)
            source_r = np.linalg.norm(source_u, axis=1)
            inner = self.boundaries[0]
            # The min is sampled on the transformed canonical physical nodes
            # now used to build self.boundaries. Thus the frozen predicate
            # source_r < min_inner - 1e-10 is unchanged, but its sampled
            # boundary radius no longer depends on scanner-vs-surface axes.
            min_inner = float(np.min(np.linalg.norm(inner["points_u"], axis=1)))
            valid = source_r < (min_inner - 1e-10)
            coeff = frozen.source_coefficients_real(
                relative, None, self.L_field, self.radius_scale_m, float(frozen.SIGMAS[0])
            )
            primary_value = np.einsum("qn,bno->qbo", inner["decaying"], coeff)
            primary_normal = np.einsum("qn,bno->qbo", inner["decaying_normal"], coeff)
            weights = inner["integration_weights"] * inner["graph_jacobian_u"]
            rhs_v = -np.einsum(
                "qn,qbo->nbo", inner["test_basis"],
                weights[:, None, None] * self.radius_scale_m**2 * primary_value,
            )
            rhs_j = -np.einsum(
                "qn,qbo->nbo", inner["test_basis"],
                weights[:, None, None] * float(frozen.SIGMAS[0]) * self.radius_scale_m * primary_normal,
            )
            rhs = np.zeros((self.block_size, len(sources_m), 3), dtype=np.float64)
            rhs[0:self.N] = rhs_v
            rhs[self.N:2*self.N] = rhs_j
            direct_count = 0
            fallback_meta = None
            if np.any(~valid):
                invalid = np.where(~valid)[0]
                direct_v, direct_j, fallback_meta = fix.coordinate_covariant_direct_fallback_rhs(
                    sources_current_m=sources_m,
                    invalid_indices=invalid,
                    geometry_coefficients=self.geometry_coefficients,
                    center_current_m=self.center_m,
                    radius_scale_m=self.radius_scale_m,
                    quadrature=self._canonical_quadrature,
                    scanner_to_surface=self.scanner_to_surface,
                    L_geometry=self.L_geometry,
                    L_field=self.L_field,
                    conductivity_s_per_m=float(frozen.SIGMAS[0]),
                )
                rhs[0:self.N, invalid, :] = direct_v
                rhs[self.N:2*self.N, invalid, :] = direct_j
                direct_count = int(len(invalid))
            self.source_injection_diagnostics.append({
                "source_count": int(len(sources_m)),
                "source_r_min_u": float(np.min(source_r)) if len(source_r) else None,
                "source_r_max_u": float(np.max(source_r)) if len(source_r) else None,
                "inner_r_min_u": min_inner,
                "series_valid_all_sources": bool(np.all(valid)),
                "invalid_source_count": int(np.sum(~valid)),
                "direct_primary_fallback_count": direct_count,
                "method": "frozen exact source injection on frame-covariant canonical quadrature",
                "quadrature_frame_covariance": {
                    "node_count": int(len(self._canonical_quadrature["weights"])),
                    "weights_unchanged": True,
                    "determinant": float(np.linalg.det(self.transform_q)),
                },
                "fallback_coordinate_covariance": fallback_meta,
            })
            return rhs

    return A09CovariantOperator, frozen, common, fix
