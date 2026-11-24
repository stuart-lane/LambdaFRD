"""Side-specific local-polynomial fits used for bias correction and inference.

Internal module; not part of the public API.
"""

"""
PACKAGE SETUP
"""

import numpy as np

from .utils import build_poly

"""
SIDE-SPECIFIC LOCAL POLYNOMIAL FITS FOR BIAS CORRECTION

The lambda-class estimator is left unchanged. This builds side-specific
order-p (bandwidth h) and order-q (bandwidth b) local-polynomial objects only
for bias correction and inference, following the CCT / rdrobust
equivalent-kernel algebra, including any pooled covariate adjustment.

The bias-corrected estimate uses the (p+1)-th coefficient of the order-q fit:

    Q_eq = (R_p W_h)' - h^{p+1} L_p e_{p+1}' G_q^{-1} (R_q W_b)',
    L_p  = (R_p W_h)' (x / h)^{p+1},

so q = p + 1 is the usual choice and q > p + 1 is also supported.
Everything is built on the UNION support (see supports.py).
"""

def lp_side(
        sup: dict,
        side: int,
        Dmat_u: np.ndarray,
        p: int,
        q: int,
        h: float
    ) -> dict:

    """Order-p and order-q fits on one side of the cutoff"""
    ii = sup['Z_u'] == side
    xc_s = sup['Xc_u'][ii]
    wh_s = sup['w_h_u'][ii]
    wb_s = sup['w_b_u'][ii]
    data_s = Dmat_u[ii, :]

    Rp = build_poly(xc_s, p)
    Rq = build_poly(xc_s, q)
    RpW = Rp * wh_s[:, None]
    RqW = Rq * wb_s[:, None]
    invGp = np.linalg.inv(Rp.T @ RpW)
    invGq = np.linalg.inv(Rq.T @ RqW)

    beta_p = invGp @ (RpW.T @ data_s)
    beta_q = invGq @ (RqW.T @ data_s)

    Lp = RpW.T @ (xc_s / h) ** (p + 1)
    e_p1 = np.zeros(q + 1)
    e_p1[p + 1] = 1
    q_score = invGq @ RqW.T
    Qeq = (RpW.T - h ** (p + 1) * np.outer(Lp, e_p1) @ q_score).T
    beta_bc = invGp @ (Qeq.T @ data_s)

    res_p = data_s - Rp @ beta_p
    res_q = data_s - Rq @ beta_q

    hii_p = np.sum((Rp @ invGp) * RpW, axis=1)
    hii_q = np.sum((Rq @ invGq) * RqW, axis=1)
    ell_p = RpW @ invGp[:, 0]   # conventional jump weights
    ell_bc = Qeq @ invGp[:, 0]  # bias-corrected jump weights

    return {
        'ii': ii, 'Rp': Rp, 'invGp': invGp,
        'beta_p': beta_p, 'beta_q': beta_q, 'beta_bc': beta_bc,
        'res_p': res_p, 'res_q': res_q, 'hii_p': hii_p, 'hii_q': hii_q,
        'ell_p': ell_p, 'ell_bc': ell_bc
    }


def lp_fits(
        sup: dict,
        p: int,
        q: int,
        h: float,
        b: float
    ) -> dict:

    """Both sides, covariate adjustment and the estimated bias of the jumps"""
    W_u = sup['W_u']
    if W_u is None:
        Dmat_u = np.column_stack([sup['Y_u'], sup['D_u']])
        dZ = 0
    else:
        Dmat_u = np.column_stack([sup['Y_u'], sup['D_u'], W_u])
        dZ = W_u.shape[1]

    left = lp_side(sup, 0, Dmat_u, p, q, h)
    right = lp_side(sup, 1, Dmat_u, p, q, h)
    beta_p_jump = right['beta_p'] - left['beta_p']
    beta_bc_jump = right['beta_bc'] - left['beta_bc']

    # Pooled covariate adjustment (rdrobust)
    if dZ > 0:
        def covariate_cross(lp):
            ii = lp['ii']
            Wcov = W_u[ii, :]
            data_s = Dmat_u[ii, :]
            wh_s = sup['w_h_u'][ii]
            Up = (lp['Rp'] * wh_s[:, None]).T @ data_s
            ZWD = (Wcov * wh_s[:, None]).T @ data_s
            UpZ = Up[:, 2:]
            UiGU = UpZ.T @ lp['invGp'] @ Up
            return {
                'ZWZ': ZWD[:, 2:] - UiGU[:, 2:],
                'ZWY': ZWD[:, :2] - UiGU[:, :2]
            }

        cc_l = covariate_cross(left)
        cc_r = covariate_cross(right)
        gamma_p = np.linalg.solve(cc_l['ZWZ'] + cc_r['ZWZ'], cc_l['ZWY'] + cc_r['ZWY'])
        gamma_Y = gamma_p[:, 0]
        gamma_D = gamma_p[:, 1]
    else:
        gamma_p = np.zeros((0, 2))
        gamma_Y = np.zeros(0)
        gamma_D = np.zeros(0)

    def adj_jump(jump, col, gam):
        adj = gam @ jump[0, 2:] if dZ > 0 else 0.0
        return float(jump[0, col] - adj)

    tau_Y_cl = adj_jump(beta_p_jump, 0, gamma_Y)
    tau_D_cl = adj_jump(beta_p_jump, 1, gamma_D)
    tau_Y_bc = adj_jump(beta_bc_jump, 0, gamma_Y)
    tau_D_bc = adj_jump(beta_bc_jump, 1, gamma_D)

    # Leading (p+1)-th coefficients of the order-q fit, in b^(p+1) units
    k1 = p + 1

    def lead(lp, col):
        return float(lp['beta_q'][k1, col] * b ** (p + 1))

    return {
        'left': left, 'right': right, 'Dmat_u': Dmat_u, 'dZ': dZ,
        'gamma_p': gamma_p, 'gamma_Y': gamma_Y, 'gamma_D': gamma_D,
        'tau_Y_cl': tau_Y_cl, 'tau_D_cl': tau_D_cl,
        # Estimated bias of the reduced-form and first-stage jumps
        'a_Y': tau_Y_cl - tau_Y_bc,
        'a_D': tau_D_cl - tau_D_bc,
        'a_Y_raw': float((right['beta_q'][k1, 0] - left['beta_q'][k1, 0]) * b ** (p + 1)),
        'a_D_raw': float((right['beta_q'][k1, 1] - left['beta_q'][k1, 1]) * b ** (p + 1)),
        'aY_plus': lead(right, 0), 'aY_minus': lead(left, 0),
        'aD_plus': lead(right, 1), 'aD_minus': lead(left, 1)
    }
