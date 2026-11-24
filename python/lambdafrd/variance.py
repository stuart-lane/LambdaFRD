"""Variance estimators (HC0-HC3, nearest-neighbour, cluster, homoskedastic).

Internal module; not part of the public API.
"""

"""
PACKAGE SETUP
"""

import numpy as np

from .nn import nn_residuals_side

"""
VARIANCE ESTIMATORS

    hc0     sum xi_i^2                          (no leverage correction)
    hc1     finite-sample rescaling of xi_i
    hc2     sum (xi_i / sqrt(1 - h_ii))^2
    hc3     sum (xi_i / (1 - h_ii))^2
    nn      nearest-neighbour residuals in place of fitted residuals (as in rdrobust)
    cluster sum_g (sum_{i in g} xi_i)^2, with the CR1 finite-sample factor
    homo    constant (co)variances from pooled residuals

Each is computed for the bias-corrected (robust) and the conventional score.
Only the selected vce is computed unless report_all_vce = True (hc0-hc3 are
always cheap and always returned).
"""

def _cluster_sums(
        xi: np.ndarray,
        cluster: np.ndarray
    ) -> np.ndarray:

    """Sum of xi within each cluster"""
    _, groups = np.unique(cluster, return_inverse=True)
    return np.bincount(groups.ravel(), weights=xi)


def lfrd_variance(
        sup: dict,
        dsg: dict,
        lp: dict,
        inf: dict,
        p: int,
        q: int,
        vce: str,
        nnmatch: int,
        nn_full_sample: bool,
        hc_lev: str,
        vce_residuals: str,
        report_all_vce: bool
    ) -> dict:

    """Bias-corrected and conventional variance estimates"""
    M = sup['M']
    df = dsg['df']
    w = sup['w']
    Z_u = sup['Z_u']
    idx_m = sup['idx_main_in_union']

    dZ = lp['dZ']
    cY = inf['cY']
    cD = inf['cD']
    cW = inf['cW']

    xi_var = inf['xi_var_q'] if vce_residuals == "q" else inf['xi_var_p']
    xi_conv = inf['xi_conv']

    # Leverages
    hlev_main = dsg['H_diag'] + dsg['P_ii'] if hc_lev == "full" else dsg['H_diag']
    hlev_main = np.minimum(hlev_main, 0.99)
    hlev_ii = np.zeros(sup['M_u'])
    hlev_ii[idx_m] = hlev_main
    if vce_residuals == "q":
        hlev_ii = inf['hlev_q']

    # HC family (bias-corrected)
    var_hc0 = np.sum(xi_var ** 2)
    if vce_residuals == "q":
        n_above = np.sum((Z_u == 1) & (sup['w_b_u'] > 0))
        n_below = np.sum((Z_u == 0) & (sup['w_b_u'] > 0))
        f_hc1 = np.where(
            Z_u == 1,
            np.sqrt(n_above / max(n_above - (q + 1), 1)),
            np.sqrt(n_below / max(n_below - (q + 1), 1))
        )
    else:
        f_hc1 = np.full(sup['M_u'], np.sqrt(M / df))
    var_hc1 = np.sum((f_hc1 * xi_var) ** 2)
    var_hc2 = np.sum((xi_var / np.sqrt(1 - hlev_ii)) ** 2)
    var_hc3 = np.sum((xi_var / (1 - hlev_ii)) ** 2)

    # Nearest neighbour
    var_nn = np.nan
    var_conv_nn = np.nan
    if vce == "nn" or report_all_vce:
        if nn_full_sample and not inf['rdrobust_nesting']:
            if dZ > 0:
                nn_data_full = np.column_stack([sup['y'], sup['d'], sup['covs']])
            else:
                nn_data_full = np.column_stack([sup['y'], sup['d']])
            nn_res_full = np.zeros(nn_data_full.shape)

            for side in (0, 1):
                ii = np.flatnonzero(sup['z'] == side)
                if len(ii) >= 2:
                    nn_res_full[ii, :] = nn_residuals_side(sup['x'][ii], nn_data_full[ii, :], nnmatch)
            nn_res_u = nn_res_full[sup['ids_u'], :]
        else:
            nn_data_u = lp['Dmat_u']
            nn_res_u = np.zeros(nn_data_u.shape)

            for side in (0, 1):
                ii = np.flatnonzero(Z_u == side)
                if len(ii) >= 2:
                    nn_res_u[ii, :] = nn_residuals_side(sup['Xc_u'][ii], nn_data_u[ii, :], nnmatch)

        if inf['rdrobust_nesting']:
            nn_score_u = nn_res_u @ inf['s_rd']
            var_conv_nn = np.sum((inf['ell_p_u'] * nn_score_u) ** 2)
            var_nn = np.sum((inf['ell_bc_u'] * nn_score_u) ** 2)
        else:
            nn_Y = nn_res_u[:, 0]
            nn_D = nn_res_u[:, 1]
            nn_score_bc = cY * nn_Y + cD * nn_D
            if dZ > 0:
                nn_score_bc = nn_score_bc + np.sum(cW * nn_res_u[:, 2:], axis=1)
            var_nn = np.sum(nn_score_bc ** 2)

            nn_score_conv = inf['cY_conv'] * nn_Y[idx_m] + inf['cD_conv'] * nn_D[idx_m]
            if dZ > 0:
                nn_score_conv = nn_score_conv + np.sum(inf['cW_conv'] * nn_res_u[idx_m, 2:], axis=1)
            var_conv_nn = np.sum(nn_score_conv ** 2)

    # Homoskedastic
    var_homo = np.nan
    var_conv_homo = np.nan
    if vce == "homo" or report_all_vce:
        eY = inf['eY_p']
        eD = inf['eD_p']
        s_YY = np.sum(eY ** 2 * w) / df
        s_DD = np.sum(eD ** 2 * w) / df
        s_YD = np.sum(eY * eD * w) / df
        var_homo = s_YY * np.sum(cY ** 2) + 2 * s_YD * np.sum(cY * cD) + s_DD * np.sum(cD ** 2)
        var_conv_homo = (s_YY * np.sum(inf['cY_conv'] ** 2)
                         + 2 * s_YD * np.sum(inf['cY_conv'] * inf['cD_conv'])
                         + s_DD * np.sum(inf['cD_conv'] ** 2))

    # Cluster (CR1)
    var_cl = np.nan
    var_conv_cl = np.nan
    n_clusters = None
    if (vce == "cluster" or report_all_vce) and sup['cluster'] is not None:
        g_sums = _cluster_sums(xi_var, sup['cluster'][sup['in_u']])
        G = len(g_sums)
        n_clusters = G
        var_cl = (G / max(G - 1, 1)) * ((M - 1) / df) * np.sum(g_sums ** 2)

        g_sums_m = _cluster_sums(xi_conv, sup['cluster'][sup['in_h']])
        Gm = len(g_sums_m)
        var_conv_cl = (Gm / max(Gm - 1, 1)) * ((M - 1) / df) * np.sum(g_sums_m ** 2)

    # Conventional HC family, on the order-p residual score
    hlev_p = inf['hlev_p']
    n_above_p = np.sum(sup['Z'] == 1)
    n_below_p = np.sum(sup['Z'] == 0)
    f_conv_hc1 = np.where(
        sup['Z'] == 1,
        np.sqrt(n_above_p / max(n_above_p - (p + 1), 1)),
        np.sqrt(n_below_p / max(n_below_p - (p + 1), 1))
    )
    var_conv_hc0 = np.sum(xi_conv ** 2)
    var_conv_hc1 = np.sum((f_conv_hc1 * xi_conv) ** 2)
    var_conv_hc2 = np.sum((xi_conv / np.sqrt(1 - hlev_p)) ** 2)
    var_conv_hc3 = np.sum((xi_conv / (1 - hlev_p)) ** 2)

    var_all = {
        'hc0': var_hc0, 'hc1': var_hc1, 'hc2': var_hc2, 'hc3': var_hc3,
        'nn': var_nn, 'cluster': var_cl, 'homo': var_homo
    }
    var_conv_all = {
        'hc0': var_conv_hc0, 'hc1': var_conv_hc1, 'hc2': var_conv_hc2,
        'hc3': var_conv_hc3, 'nn': var_conv_nn, 'cluster': var_conv_cl,
        'homo': var_conv_homo
    }

    var_BC = var_all[vce]
    if not np.isfinite(var_BC) or var_BC <= 0:
        raise ValueError(f"Selected variance estimator (vce = '{vce}') is not positive.")
    var_conv = var_conv_all[vce]
    if not np.isfinite(var_conv) or var_conv <= 0:
        raise ValueError(f"Selected conventional variance estimator (vce = '{vce}') is not positive.")

    return {
        'var_BC': var_BC, 'var_conv': var_conv,
        'var_all': var_all, 'var_conv_all': var_conv_all,
        'n_clusters': n_clusters, 'xi_var': xi_var
    }
