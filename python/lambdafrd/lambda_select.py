"""Choice of lambda: alpha rules, trace weights and trace targets.

Internal module; not part of the public API.
"""

"""
PACKAGE SETUP
"""

import warnings

import numpy as np

"""
CHOICE OF LAMBDA

ALPHA RULES
    "trace"  : alpha = psi  (psi = 1: second-order unbiased for every rho)
    "psi_df" : implements a simple Fuller  1 - lambda = psi / (M - 2(p+1))

TRACE WEIGHTS (which traces convert alpha into 1 - lambda)
    "kernel"  : tr(P K) / tr(Q K). Exactly the leading-bias-cancelling value
                under homoskedasticity (Theorem 1)
    "variance": tr(P K Sigma_v) / tr(Q K Sigma_v), Sigma_v = diag(pi_i(1-pi_i))
                from the linear first stage (heteroskedastic version).
alpha_eff is always reported in the "variance" canonical units (C_P/C_Q).

TRACE TARGETS (multiplier on the trace ratio)
    "uncorrected": 1                     -- removes the leading bias of tau_lambda
    "bc"         : max(2 rho_x - 1, 1)   -- removes it for the BC estimator
    "bc_raw"     : 2 rho_x - 1 without the floor (comparison only)
"""

def cross_weight_ratio(
        lp: dict,
        sup: dict,
        pi_i: np.ndarray,
        trace_weight: str
    ) -> float:

    """Cross-weight ratio rho_x = <w, w_BC> / <w, w>

    w and w_BC are the conventional and bias-corrected weights from lp_fits().
    For the BC estimator the leading ratio bias vanishes at
    1 - lambda = (2 rho_x - 1) tr(PK)/tr(QK). (Triangular, p = 1, b = h,
    locally uniform design: rho_x = 5/4.) With trace_weight = "variance" the
    inner products are weighted by pi_i(1 - pi_i). Covariates are ignored
    here (approximation).
    """
    cw_u = np.ones(sup['M_u'])

    if trace_weight == "variance":
        cw_u = np.zeros(sup['M_u'])
        cw_u[sup['idx_main_in_union']] = pi_i * (1 - pi_i)

    num = 0.0
    den = 0.0
    for s in (lp['left'], lp['right']):
        cw = cw_u[s['ii']]
        num += np.sum(cw * s['ell_p'] * s['ell_bc'])
        den += np.sum(cw * s['ell_p'] ** 2)

    return num / den


def select_lambda(
        dsg: dict,
        lp: dict,
        sup: dict,
        p: int,
        _lambda: float,
        alpha_rule: str,
        psi: float,
        trace_weight: str,
        trace_target: str
    ) -> dict:

    """Main selector

    With a user-supplied _lambda the rule-based quantities are still computed
    and reported, but _lambda is used as given.
    """
    C_P = dsg['C_P']
    C_Q = dsg['C_Q']
    M = sup['M']
    CP_CQ = C_P / C_Q
    ratio_K = dsg['L_eff'] / dsg['q_eff']
    ratio_use = ratio_K if trace_weight == "kernel" else CP_CQ

    # BC target: cross-weight ratio
    try:
        rho_cross = cross_weight_ratio(lp, sup, dsg['pi_i'], trace_weight)
    except (ZeroDivisionError, FloatingPointError, ValueError):
        rho_cross = np.nan
    bc_factor = 2 * rho_cross - 1

    # Floor at 1 for "bc": never less shrinkage than Lambda*, which keeps
    # 1 - lambda bounded below as in the moment corollary.
    if trace_target == "uncorrected":
        target_factor = 1.0
    elif trace_target == "bc":
        target_factor = max(bc_factor, 1) if np.isfinite(bc_factor) else 1.0
    else:
        target_factor = bc_factor if np.isfinite(bc_factor) else 1.0
    ratio_use = ratio_use * target_factor

    # Moment diagnostics
    # kappa_N = c_Q,N / c_P,N with c_Q,N = tr(QK)/r, c_P,N = tr(PK), r = M - 2(p+1).
    # psi_star = 1/kappa_N is the psi_df value that cancels the leading
    # strong-IV bias
    r_df = M - 2 * (p + 1)
    kappa_N = (dsg['q_eff'] / r_df) / dsg['L_eff']
    psi_star = 1 / kappa_N

    # Alpha rule -> 1 - lambda
    if alpha_rule == "psi_df":
        a_trace_raw = psi / r_df
    else:
        a_trace_raw = min(psi * ratio_use, 1)
    alpha_eff = a_trace_raw / CP_CQ

    if _lambda is None:
        a_trace = a_trace_raw
        lambda_val = 1 - a_trace
    else:
        lambda_val = float(_lambda)
        a_trace = 1 - lambda_val

    if not np.isfinite(lambda_val):
        raise ValueError("lambda is not finite.")
    if lambda_val < 0 or lambda_val >= 1:
        warnings.warn(f"lambda = {round(lambda_val, 4)} is outside [0,1), so proceed with caution.")

    return {
        'lambda': lambda_val, 'a_trace': a_trace, 'a_trace_raw': a_trace_raw,
        'alpha_eff': alpha_eff, 'CP_CQ': CP_CQ, 'ratio_K': ratio_K,
        'ratio_use': ratio_use, 'rho_cross': rho_cross,
        'bc_factor': bc_factor, 'target_factor': target_factor,
        'kappa_N': kappa_N, 'psi_star': psi_star
    }
