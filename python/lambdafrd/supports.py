"""Main, bias and union sample windows used by the estimator.

Internal module; not part of the public API.
"""

"""
PACKAGE SETUP
"""

import numpy as np

from .utils import kern_weights

"""
SAMPLE SUPPORTS

Three windows are used:
    main  : observations with K((x - c)/h) > 0  (the lambda-class fit)
    bias  : observations with K((x - c)/b) > 0  (the order-q bias fit)
    union : main OR bias.

When b > h the order-q fit uses observations outside the main window. They
contribute sampling variability to the bias estimate, so every influence
quantity is built on the UNION support. When b <= h the union is the main
window.
"""

def lfrd_supports(
        y: np.ndarray,
        d: np.ndarray,
        x: np.ndarray,
        c: float,
        h: float,
        b: float,
        kernel: str,
        covs: np.ndarray = None,
        cluster: np.ndarray = None
    ) -> dict:

    """Full-sample, main-window and union-window data"""
    z = (x >= c).astype(float)
    xc = x - c
    w_h = kern_weights(xc, h, kernel)
    w_b = kern_weights(xc, b, kernel)

    in_h = w_h > 0
    in_b = w_b > 0
    in_u = in_h | in_b
    ids_h = np.flatnonzero(in_h)
    ids_u = np.flatnonzero(in_u)

    return {
        # Full sample
        'y': y, 'd': d, 'x': x, 'z': z, 'xc': xc, 'covs': covs, 'cluster': cluster,
        'w_h': w_h, 'w_b': w_b, 'in_h': in_h, 'in_b': in_b, 'in_u': in_u,

        # Main window
        'Y': y[in_h], 'D': d[in_h], 'Z': z[in_h], 'Xc': xc[in_h], 'w': w_h[in_h],
        'W': None if covs is None else covs[in_h, :],
        'M': int(in_h.sum()),

        # Bias window
        'M_b': int(in_b.sum()),

        # Union
        'Y_u': y[in_u], 'D_u': d[in_u], 'Z_u': z[in_u], 'Xc_u': xc[in_u],
        'w_h_u': w_h[in_u], 'w_b_u': w_b[in_u],
        'W_u': None if covs is None else covs[in_u, :],
        'M_u': len(ids_u),
        'idx_main_in_union': np.searchsorted(ids_u, ids_h),
        'ids_u': ids_u
    }
