from __future__ import annotations

from typing import Tuple
import numpy as np


def stable_inverse_constant(F: np.ndarray, floor: float = 1e-12) -> Tuple[float, float, np.ndarray]:
    """Return a finite-dimensional stability proxy C=1/sigma_min.

    This is an operational diagnostic for an admissible discretized subspace.
    It is not a universal PDE stability theorem.
    """
    if F.ndim != 2:
        raise ValueError("F must be a 2D array.")
    if min(F.shape) == 0:
        raise ValueError("F must be nonempty.")
    svals = np.linalg.svd(F, compute_uv=False)
    sigma_min = float(np.min(svals))
    Cstab = float(1.0 / max(sigma_min, floor))
    return Cstab, sigma_min, svals
