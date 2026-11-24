"""Population and empirical leading-bias constants (reported as diagnostics).

Internal module; not part of the public API.
"""

"""
PACKAGE SETUP
"""

import numpy as np

from .utils import build_poly

"""
BIAS CONSTANTS (reported as diagnostics)
"""

def pop_bias_const(
        kernel: str,
        p: int
    ) -> float:

    """Population constant

        c_pop = e_0' Gamma^{-1} vartheta,
        Gamma_{jk}   = int_0^1 K(u) u^{j+k} du,      j,k = 0..p
        vartheta_j   = int_0^1 K(u) u^{j+p+1} du

    Kernel normalising constants cancel. Known values (right side, p = 1):
    uniform -1/6, triangular -1/10, Epanechnikov -11/95.

    Left side: c_pop_minus = (-1)^{p+1} * c_pop_plus. For odd p the two
    coincide, so the bias is a constant times the JUMP in the (p+1)-th
    derivative; for even p it is a constant times the SUM.
    """
    def nu(j):
        if kernel == "uniform":
            return 1 / (j + 1)
        elif kernel == "triangular":
            return 1 / (j + 1) - 1 / (j + 2)
        else:
            return 1 / (j + 1) - 1 / (j + 3)

    idx = np.arange(p + 1)
    G = np.array([[nu(a + b) for b in idx] for a in idx])
    th = np.array([nu(a + p + 1) for a in idx])

    return float(np.linalg.solve(G, th)[0])


def emp_bias_const_side(
        u: np.ndarray,
        wt: np.ndarray,
        p: int
    ) -> float:

    """Empirical constant (one side), on the MAIN bandwidth window

        c_hat = e_0' [ sum_i w_i r_p(u_i) r_p(u_i)' ]^{-1}
                     [ sum_i w_i r_p(u_i) u_i^{p+1} ],     u_i = (X_i - c)/h

    Converges to pop_bias_const() but differs in finite samples through the
    local design density. Covariates are not partialled out.
    """
    if len(u) < p + 2:
        return np.nan

    R = build_poly(u, p)
    RW = R * wt[:, None]
    G = R.T @ RW
    th = RW.T @ u ** (p + 1)
    try:
        return float(np.linalg.solve(G, th)[0])
    except np.linalg.LinAlgError:
        return np.nan


def bias_constants(
        sup: dict,
        h: float,
        p: int,
        kernel: str
    ) -> dict:

    """Both sides, with a population fallback when the empirical constant
    cannot be formed on one side"""
    U_h = sup['Xc'] / h
    c_pop = pop_bias_const(kernel, p)
    side_sign = (-1) ** (p + 1)
    above = sup['Z'] == 1
    below = sup['Z'] == 0
    c_emp_plus = emp_bias_const_side(U_h[above], sup['w'][above], p)
    c_emp_minus = emp_bias_const_side(U_h[below], sup['w'][below], p)

    if np.isfinite(c_emp_plus) and np.isfinite(c_emp_minus):
        c_plus, c_minus, used = c_emp_plus, c_emp_minus, "empirical"
    else:
        c_plus, c_minus, used = c_pop, side_sign * c_pop, "population (fallback)"

    return {
        'used': used, 'plus': c_plus, 'minus': c_minus, 'pop': c_pop,
        'emp_plus': c_emp_plus, 'emp_minus': c_emp_minus, 'side_sign': side_sign
    }
