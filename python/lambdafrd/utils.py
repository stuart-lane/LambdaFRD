"""Kernel weights and polynomial design matrices.

Internal module; not part of the public API.
"""

"""
PACKAGE SETUP
"""

import numpy as np

"""
KERNEL WEIGHTS AND POLYNOMIAL DESIGN MATRICES
"""

def kern_weights(
        xc: np.ndarray,
        h: float,
        kernel: str
    ) -> np.ndarray:

    """Kernel weights K((x - c) / h)"""
    if kernel == "uniform":
        return (np.abs(xc) <= h).astype(float) * 0.5
    elif kernel == "triangular":
        return np.maximum(0, 1 - np.abs(xc) / h)
    else:
        return np.maximum(0, 0.75 * (1 - (xc / h) ** 2))


def build_poly(
        u: np.ndarray,
        order: int
    ) -> np.ndarray:

    """Polynomial design matrix in u: [1, u, u^2, ..., u^order]"""
    mat = np.ones((len(u), order + 1))
    for j in range(1, order + 1):
        mat[:, j] = u ** j

    return mat
