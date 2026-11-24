"""Influence functions (scores) for the conventional and bias-corrected estimators.

Internal module; not part of the public API.
"""

"""
PACKAGE SETUP
"""

import numpy as np

"""
INFLUENCE FUNCTIONS

Bias-corrected estimator
    tau_BC = tau_Y_lambda / tau_D_lambda
             - lambda (tau_IV_D a_Y - tau_IV_Y a_D) / tau_D_lambda,
linearised by the delta method in
    theta = (tau_Y_lambda, tau_D_lambda, a_Y, a_D, tau_IV_Y, tau_IV_D).
The derivatives through the IV moments in the bias term are dropped
(g5 = g6 = 0), matching the robust bias-corrected variance convention.

All scores live on the UNION support (see supports.py). The conventional
(uncorrected) score lives on the main window.
"""

def lfrd_influence(
        sup: dict,
        dsg: dict,
        est: dict,
        lp: dict,
        lambda_val: float
    ) -> dict:

    """Score coefficients and influence functions for both estimators"""
    M_u = sup['M_u']
    idx_m = sup['idx_main_in_union']
    dZ = lp['dZ']

    # Gradient vector for the delta method
    tYl = est['tau_Y_lambda']
    tDl = est['tau_D_lambda']
    C_const = lambda_val

    g1 = 1 / tDl
    g2 = -tYl / tDl ** 2
    g3 = -C_const * est['tau_IV_D'] / tDl
    g4 = C_const * est['tau_IV_Y'] / tDl
    g5 = 0.0
    g6 = 0.0

    # Side-specific residuals, leverages and jump weights on the union
    ncolD = lp['Dmat_u'].shape[1]
    res_p_u = np.zeros((M_u, ncolD))
    res_q_u = np.zeros((M_u, ncolD))
    ell_p_u = np.zeros(M_u)
    ell_bc_u = np.zeros(M_u)
    hlev_p_u = np.zeros(M_u)
    hlev_q_u = np.zeros(M_u)
    for s in (lp['left'], lp['right']):
        ii = s['ii']
        res_p_u[ii, :] = s['res_p']
        res_q_u[ii, :] = s['res_q']
        ell_p_u[ii] = s['ell_p']
        ell_bc_u[ii] = s['ell_bc']
        hlev_p_u[ii] = s['hii_p']
        hlev_q_u[ii] = s['hii_q']

    bias_weight_u = ell_p_u - ell_bc_u
    below = sup['Z_u'] == 0
    bias_weight_u[below] = -bias_weight_u[below]

    sw_u = np.zeros(M_u)
    sw_u[idx_m] = dsg['sw']
    ImLMD_u = np.zeros(M_u)
    ImLMD_u[idx_m] = est['ImLMD']
    Ztilde_u = np.zeros(M_u)
    Ztilde_u[idx_m] = dsg['Z_tilde']

    # Bias-corrected score coefficients
    cY = (g1 * ImLMD_u + g5 * Ztilde_u) * sw_u + g3 * bias_weight_u
    cD = (g2 * ImLMD_u + g6 * Ztilde_u) * sw_u + g4 * bias_weight_u
    if dZ > 0:
        cW = -np.outer(cY, lp['gamma_Y']) - np.outer(cD, lp['gamma_D'])
    else:
        cW = np.zeros((M_u, 0))

    xi = cY * res_q_u[:, 0] + cD * res_q_u[:, 1]
    if dZ > 0:
        xi = xi + np.sum(cW * res_q_u[:, 2:], axis=1)

    eY_p = res_p_u[idx_m, 0]
    eD_p = res_p_u[idx_m, 1]

    # Conventional lambda-class score, with the pooled covariate adjustment
    Pz_Y = dsg['Z_tilde'] * (dsg['ZtZ_inv'] * float(dsg['Z_tilde'] @ dsg['Y_tilde']))
    K_Y = (1 - lambda_val) * dsg['Y_tilde'] + lambda_val * Pz_Y
    cY_conv = dsg['sw'] * est['ImLMD'] / tDl
    cD_conv = dsg['sw'] * (K_Y - 2 * est['tau_lambda'] * est['ImLMD']) / tDl
    xi_conv = cY_conv * eY_p + cD_conv * eD_p
    cW_conv = None
    if dZ > 0:
        eW_p = res_p_u[idx_m, 2:]
        cW_conv = -np.outer(cY_conv, lp['gamma_Y']) - np.outer(cD_conv, lp['gamma_D'])
        xi_conv = xi_conv + np.sum(cW_conv * eW_p, axis=1)

    # Residuals entering the BC variance: order-q (default) or order-p
    xi_var_q = xi
    xi_var_p = cY * res_p_u[:, 0] + cD * res_p_u[:, 1]
    if dZ > 0:
        xi_var_p = xi_var_p + np.sum(cW * res_p_u[:, 2:], axis=1)

    # lambda = 1: exact covariate-adjusted fuzzy-RD score (rdrobust)
    # Inference-only nesting branch: tau_lambda and the IV moment structure are
    # not replaced or re-estimated.
    rdrobust_nesting = abs(lambda_val - 1) < 1e-12
    s_rd = np.concatenate([
        [1 / lp['tau_D_cl'], -lp['tau_Y_cl'] / lp['tau_D_cl'] ** 2],
        -lp['gamma_Y'] / lp['tau_D_cl'] + (lp['tau_Y_cl'] / lp['tau_D_cl'] ** 2) * lp['gamma_D']
    ])
    score_p_u = res_p_u @ s_rd
    score_q_u = res_q_u @ s_rd
    if rdrobust_nesting:
        xi_conv = (ell_p_u * score_p_u)[idx_m]
        xi_var_q = ell_bc_u * score_q_u
        xi_var_p = xi_var_q

    return {
        'g': np.array([g1, g2, g3, g4, g5, g6]),
        'cY': cY, 'cD': cD, 'cW': cW,
        'cY_conv': cY_conv, 'cD_conv': cD_conv, 'cW_conv': cW_conv,
        'xi_var_q': xi_var_q, 'xi_var_p': xi_var_p, 'xi_conv': xi_conv,
        'res_p_u': res_p_u, 'res_q_u': res_q_u, 'eY_p': eY_p, 'eD_p': eD_p,
        'ell_p_u': ell_p_u, 'ell_bc_u': ell_bc_u,
        'hlev_q': np.minimum(hlev_q_u, 0.99),
        'hlev_p': np.minimum(hlev_p_u[idx_m], 0.99),
        'rdrobust_nesting': rdrobust_nesting, 's_rd': s_rd
    }
