"""Partialled lambda-class IV design, trace quantities and the lambda-class estimate.

Internal module; not part of the public API.
"""

"""
PACKAGE SETUP
"""

import warnings

import numpy as np

"""
LAMBDA-CLASS DESIGN AND ESTIMATOR

The local fuzzy-RD model on the main window is written as a just-identified
IV regression

    Y = tau D + V gamma + u,      D = pi Z + V delta + v,

with V = [1, (1-Z) Xc^j, Z Xc^j (j = 1..p), covariates] and instrument
Z = 1{x >= c}, all weighted by the kernel. V is partialled out
(Frisch-Waugh), after which the lambda-class estimator is

    tau_lambda = D'(I - lambda M_z) Y / D'(I - lambda M_z) D

on the partialled data. lambda = 1 is the conventional fuzzy-RD estimator
and lambda = 0 is the local OLS sharp RD estimator.
"""

def lambda_class_design(
        sup: dict,
        p: int
    ) -> dict:

    """Partialled design, projection diagonals and trace quantities"""
    Y = sup['Y']
    D = sup['D']
    Z = sup['Z']
    Xc = sup['Xc']
    w = sup['w']
    M = sup['M']

    # V matrix
    V = np.ones((M, 1))
    for j in range(1, p + 1):
        V = np.hstack([V, ((1 - Z) * Xc ** j).reshape(-1, 1), (Z * Xc ** j).reshape(-1, 1)])
    if sup['W'] is not None:
        V = np.hstack([V, sup['W']])
    df = M - V.shape[1] - 1

    if df <= 0:
        raise ValueError("Too few observations in the bandwidth.")

    # Kernel weights (sqrt-scaling convention)
    sw = np.sqrt(w)
    Vs = sw[:, None] * V
    Zs = sw * Z
    Ys = sw * Y
    Ds = sw * D

    # Partial out V
    VtV_inv = np.linalg.inv(Vs.T @ Vs)

    def partial_out(v):
        return v - Vs @ (VtV_inv @ (Vs.T @ v))

    Y_tilde = partial_out(Ys)
    D_tilde = partial_out(Ds)
    Z_tilde = partial_out(Zs)

    ZtZ = float(Z_tilde @ Z_tilde)

    if not np.isfinite(ZtZ) or ZtZ <= 0:
        raise ValueError("Partialled instrument has no variation.")
    ZtZ_inv = 1 / ZtZ

    # Projection diagonals: H (polynomial controls), P (instrument), Q (residual)
    H_diag = np.sum((Vs @ VtV_inv) * Vs, axis=1)
    P_ii = Z_tilde ** 2 * ZtZ_inv
    Q_ii_raw = 1 - H_diag - P_ii

    if np.min(Q_ii_raw) < -1e-8:
        warnings.warn("Some Q diagonal elements are materially negative.")
    Q_ii = np.maximum(Q_ii_raw, 0)

    # Linear first-stage fitted probabilities, for variance traces
    try:
        A = np.hstack([Vs, Zs.reshape(-1, 1)])
        coef_fs = np.linalg.lstsq(A, Ds, rcond=None)[0]
        fs_fit = A @ coef_fs
    except np.linalg.LinAlgError:
        fs_fit = np.full(M, np.nan)

    if np.any(np.isnan(fs_fit)):
        pi_i = np.full(M, np.mean(D))
    else:
        pi_i = fs_fit / np.maximum(sw, 1e-12)
    pi_i = np.minimum(np.maximum(pi_i, 1e-4), 1 - 1e-4)

    # Traces: C_P = tr(P Omega), C_Q = tr(Q Omega), L = tr(PK), q = tr(QK)
    c_i = w * pi_i * (1 - pi_i)
    C_P = np.sum(P_ii * c_i)
    C_Q = np.sum(Q_ii * c_i)
    L_eff = np.sum(P_ii * w)
    q_eff = np.sum(Q_ii * w)

    if not np.isfinite(C_P) or C_P < 0:
        raise ValueError("C_P is invalid.")
    if not np.isfinite(C_Q) or C_Q <= 0:
        raise ValueError("C_Q <= 0: trace-Fuller lambda cannot be formed.")
    if not np.isfinite(L_eff) or L_eff <= 0 or not np.isfinite(q_eff) or q_eff <= 0:
        raise ValueError("Kernel traces tr(PK), tr(QK) are not positive.")

    return {
        'sw': sw, 'df': df, 'Y_tilde': Y_tilde, 'D_tilde': D_tilde, 'Z_tilde': Z_tilde,
        'ZtZ': ZtZ, 'ZtZ_inv': ZtZ_inv, 'H_diag': H_diag, 'P_ii': P_ii, 'Q_ii': Q_ii,
        'pi_i': pi_i, 'C_P': C_P, 'C_Q': C_Q, 'L_eff': L_eff, 'q_eff': q_eff
    }


def lambda_class_estimate(
        dsg: dict,
        _lambda: float
    ) -> dict:

    """Lambda-class estimate, plus the IV moments"""
    Z_tilde = dsg['Z_tilde']
    D_tilde = dsg['D_tilde']
    Y_tilde = dsg['Y_tilde']

    Mz_D = D_tilde - Z_tilde * (dsg['ZtZ_inv'] * np.sum(Z_tilde * D_tilde))
    ImLMD = D_tilde - _lambda * Mz_D

    tau_Y_lambda = float(ImLMD @ Y_tilde)
    tau_D_lambda = float(ImLMD @ D_tilde)
    if not np.isfinite(tau_D_lambda) or abs(tau_D_lambda) < np.finfo(float).eps:
        raise ValueError("lambda-class denominator is numerically zero.")

    tau_IV_D = float(Z_tilde @ D_tilde)
    tau_IV_Y = float(Z_tilde @ Y_tilde)

    return {
        'Mz_D': Mz_D, 'ImLMD': ImLMD,
        'tau_Y_lambda': tau_Y_lambda, 'tau_D_lambda': tau_D_lambda,
        'tau_lambda': tau_Y_lambda / tau_D_lambda,
        'tau_IV_Y': tau_IV_Y, 'tau_IV_D': tau_IV_D,
        'pi_hat': tau_IV_D / dsg['ZtZ']
    }
