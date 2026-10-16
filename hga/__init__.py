"""Harmonic global assembly (HGA), formerly SHA-DGM."""
from .forward import build_leadfield, fit_geometry, common_average
from .inverse import minimum_norm_kernel
__version__ = '0.1.0'
