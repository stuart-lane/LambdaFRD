"""Main user-facing function, :func:`LambdaFRD`, and its input checks."""

"""
PACKAGE SETUP
"""

import warnings

import numpy as np

from .supports import lfrd_supports
from .lambda_class import lambda_class_design, lambda_class_estimate
from .local_poly import lp_fits
from .lambda_select import select_lambda
from .influence import lfrd_influence
from .variance import lfrd_variance
from .inference import lfrd_wald, lfrd_diagnostics
from .bias_constants import bias_constants
from .results import LambdaFRDResults

"""
LAMBDAFRD UTIL FUNCTIONS
"""

def _validate_inputs(
        y: np.ndarray,
        x: np.ndarray,
        fuzzy: np.ndarray,
        p: int,
        q: int,
        h: float,
        b: float,
        covs: np.ndarray,
        cluster: np.ndarray,
        kernel: str,
        alpha_rule: str,
        trace_weight: str,
        trace_target: str,
        vce: str,
        nnmatch: int,
        hc_lev: str,
        vce_residuals: str,
        report_all_vce: bool,
        alpha: float
    ) -> None:

    """Check for data/parameter errors"""

    # Vector lengths
    if len(y) != len(fuzzy) or len(y) != len(x):
        raise ValueError("y, x and fuzzy must be the same length.")
    if covs is not None and covs.shape[0] != len(y):
        raise ValueError("covs must have one row per observation.")
    if cluster is not None and len(cluster) != len(y):
        raise ValueError("cluster must be the same length as y.")

    # Polynomial orders and bandwidths
    if not isinstance(p, (int, np.integer)) or p < 1:
        raise ValueError("p must be a positive integer (p >= 1 required for bias correction).")
    if not isinstance(q, (int, np.integer)) or q <= p:
        raise ValueError("q must be an integer greater than p.")
    if h is None or not np.isscalar(h) or not np.isfinite(h) or h <= 0:
        raise ValueError("h must be a positive, finite number.")
    if not np.isscalar(b) or not np.isfinite(b) or b <= 0:
        raise ValueError("b must be a positive, finite number.")
    if b < h:
        warnings.warn("b < h, so bias estimates will use a narrower support than the main fit.")

    if not np.isscalar(alpha) or alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be a single number strictly between 0 and 1.")

    # Option values
    options = {
        'kernel': (kernel, ("triangular", "uniform", "epanechnikov")),
        'alpha_rule': (alpha_rule, ("trace", "psi_df")),
        'trace_weight': (trace_weight, ("kernel", "variance")),
        'trace_target': (trace_target, ("uncorrected", "bc", "bc_raw")),
        'vce': (vce, ("nn", "hc0", "hc1", "hc2", "hc3", "cluster", "homo")),
        'hc_lev': (hc_lev, ("controls", "full")),
        'vce_residuals': (vce_residuals, ("q", "p"))
    }
    for name, (value, allowed) in options.items():
        if value not in allowed:
            raise ValueError(f"{name} must be one of {', '.join(repr(a) for a in allowed)}.")

    if vce == "cluster" and cluster is None:
        raise ValueError("vce = 'cluster' requires a cluster ID vector.")
    if cluster is not None and vce != "cluster" and not report_all_vce:
        warnings.warn("cluster supplied but vce != 'cluster'; it will be ignored.")
    if nnmatch < 1:
        raise ValueError("nnmatch must be at least 1.")


def _is_missing(
        v: np.ndarray
    ) -> np.ndarray:

    """Elementwise missing-value check for numeric or object arrays"""
    if np.issubdtype(v.dtype, np.number):
        return np.isnan(v)
    return np.array([val is None or val != val for val in v])


"""
LAMBDAFRD MAIN FUNCTION
"""

def LambdaFRD(
        y: np.ndarray,
        x: np.ndarray,
        fuzzy: np.ndarray,
        c: float = 0,
        p: int = 1,
        q: int = None,
        h: float = None,
        b: float = None,
        covs: np.ndarray = None,
        kernel: str = "triangular",
        _lambda: float = None,
        alpha_rule: str = "trace",
        psi: float = 1,
        trace_weight: str = "kernel",
        trace_target: str = "uncorrected",
        vce: str = "nn",
        nnmatch: int = 3,
        cluster: np.ndarray = None,
        nn_full_sample: bool = True,
        hc_lev: str = "controls",
        vce_residuals: str = "q",
        report_all_vce: bool = False,
        alpha: float = 0.05,
        tau_0: float = 0
    ) -> LambdaFRDResults:

    """Lambda-class estimation and inference for fuzzy regression discontinuity.

    Estimates the FRD effect at the cutoff with a lambda-class estimator and
    provides robust bias-corrected confidence intervals. This estimator class
    has finite moments up to the order of the error distribution, and is
    shown to have good estimator and inference properties in Lane (2025).
    ``lambda = 1`` is the conventional FRD estimator. As in ``rdrobust``, a
    single call returns the conventional estimate with its standard error and
    the bias-corrected estimate with its robust standard error and confidence
    interval. Consult "The moment is here: a generalized class of estimators
    for fuzzy regression discontinuity designs" Lane (2025) for more details.

    Parameters
    ----------
    y : array_like
        Outcome vector.
    x : array_like
        Running variable.
    fuzzy : array_like
        Treatment receipt vector (the fuzzy-RD treatment indicator).
    c : float, default 0
        Cutoff. Observations with ``x >= c`` are on the right (instrument = 1)
        side.
    p : int, default 1
        Order of the local polynomial used for the point estimate. Must be at
        least 1.
    q : int, optional
        Order of the local polynomial used for bias correction. Must be
        greater than ``p``. Default ``p + 1``.
    h : float
        Main bandwidth (required).
    b : float, optional
        Bias bandwidth. Default ``h``.
    covs : array_like, optional
        Matrix (or 1-d vector) of covariates, one row per observation,
        included linearly in the lambda-class fit and partialled out of the
        bias fits as in ``rdrobust``.
    kernel : {'triangular', 'uniform', 'epanechnikov'}, default 'triangular'
        Kernel function.
    _lambda : float, optional
        Fixed value of lambda (named ``lambda`` in the R package; ``lambda``
        is a reserved word in Python). When ``None`` (default) lambda is
        chosen by ``alpha_rule``. When supplied, the rule-based quantities
        are still reported but ``_lambda`` is used as given (e.g.
        ``_lambda=1`` for the conventional FRD estimator and ``_lambda=0``
        for the conventional sharp RD estimator).
    alpha_rule : {'trace', 'psi_df'}, default 'trace'
        Rule for selecting lambda, with ``1 - lambda = alpha * trace ratio``:
        ``'trace'`` sets ``alpha = psi``; ``'psi_df'`` sets
        ``1 - lambda = psi / (M - 2(p+1))``, with ``M`` the number of
        observations in the main window.
    psi : float, default 1
        Multiplier for the ``'trace'`` and ``'psi_df'`` rules.
    trace_weight : {'kernel', 'variance'}, default 'kernel'
        Traces converting alpha into ``1 - lambda``: ``'kernel'``
        (``tr(PK) / tr(QK)``) or ``'variance'`` (first-stage-variance
        weighted).
    trace_target : {'uncorrected', 'bc', 'bc_raw'}, default 'uncorrected'
        Bias target: ``'uncorrected'`` removes the leading ratio bias of the
        conventional lambda-class estimator; ``'bc'`` removes it for the
        bias-corrected estimator (factor ``max(2 rho_x - 1, 1)``);
        ``'bc_raw'`` uses the same factor without the floor at 1.
    vce : {'nn', 'hc0', 'hc1', 'hc2', 'hc3', 'cluster', 'homo'}, default 'nn'
        Variance estimator.
    nnmatch : int, default 3
        Minimum number of nearest-neighbour matches for ``vce='nn'``.
    cluster : array_like, optional
        Cluster identifiers, required for ``vce='cluster'``.
    nn_full_sample : bool, default True
        If True, nearest-neighbour residuals are built from the full sample
        on each side, as in ``rdrobust``.
    hc_lev : {'controls', 'full'}, default 'controls'
        Leverages for the HC2/HC3 corrections when ``vce_residuals='p'``.
    vce_residuals : {'q', 'p'}, default 'q'
        Residuals entering the bias-corrected variance: ``'q'`` (order-q fit,
        as in ``rdrobust``) or ``'p'``.
    report_all_vce : bool, default False
        If True, compute every variance estimator (returned in ``var_all`` /
        ``var_conv_all``).
    alpha : float, default 0.05
        Significance level for the confidence intervals.
    tau_0 : float, default 0
        Null value for the reported t-statistics.

    Returns
    -------
    LambdaFRDResults
        A ``dict`` subclass with ``summary()`` and ``coefficients()``
        methods. Its keys include, among others:

        ``coef``, ``se``, ``ci``
            Arrays whose rows are ``Conventional`` and ``Robust``
            (bias-corrected estimate, robust standard error); ``ci`` has
            columns ``[lower, upper]``.
        ``tau_lambda``, ``se_conv``, ``ci_lower_conv``, ``ci_upper_conv``
            Conventional lambda-class estimate, standard error and interval.
        ``tau_BC``, ``se_BC``, ``ci_lower_BC``, ``ci_upper_BC``
            Bias-corrected estimate, robust standard error and interval.
        ``t_stat``, ``t_stat_conv``
            t-statistics for ``tau = tau_0`` (robust and conventional).
        ``lambda``
            The value of lambda used.
        ``F_stat``, ``F_can``, ``G_P``, ``C_P``, ``kappa_N``, ``psi_star``, ``rho_cross``
            First-stage and trace diagnostics.
        ``var_all``, ``var_conv_all``, ``se_all``, ``se_conv_all``
            Dicts of every variance estimator, keyed by ``vce`` name
            (``nan`` where not computed; see ``report_all_vce``).
        ``N``, ``N_h``, ``M``, ``M_bias``, ``M_union``
            Sample size after dropping missing values, main-window sizes
            left and right of the cutoff, and main / bias / union window
            sizes.

    Raises
    ------
    ValueError
        If the inputs are inconsistent (lengths differ, invalid options,
        missing bandwidth, too few observations in the window, ...).

    Warns
    -----
    UserWarning
        If ``b < h``, if ``cluster`` is supplied but unused, or if the
        selected lambda falls outside ``[0, 1)``.

    Notes
    -----
    On the main window the local fuzzy-RD model is written as a
    just-identified IV regression with instrument ``Z = 1{x >= c}`` and
    side-specific polynomial controls ``V``. After partialling out ``V``
    (with kernel weights), the lambda-class estimator is::

        tau_lambda = D'(I - lambda M_z) Y / D'(I - lambda M_z) D

    The bias correction subtracts::

        bias = lambda (tau_IV_D a_Y - tau_IV_Y a_D) / tau_D_lambda

    where ``a_Y`` and ``a_D`` are the estimated smoothing biases of the
    reduced-form and first-stage jumps from order-``q`` fits with bandwidth
    ``b``. Observations with missing values in ``y``, ``x``, ``fuzzy``,
    ``covs`` or ``cluster`` are dropped. With ``_lambda=1`` the estimates
    and standard errors coincide with ``rdrobust``'s fuzzy RD output.

    References
    ----------
    Lane, S. (2025). The moment is here: a generalized class of estimators
    for fuzzy regression discontinuity designs.

    Examples
    --------
    >>> import numpy as np
    >>> from lambdafrd import LambdaFRD
    >>> rng = np.random.default_rng(1)
    >>> n = 1000
    >>> x = rng.uniform(-1, 1, n)
    >>> d = (rng.normal(size=n) < -0.5 + 1.0 * (x >= 0)).astype(float)
    >>> y = 0.5 * x + 1 * d + rng.normal(scale=0.3, size=n)
    >>> fit = LambdaFRD(y, x, d, h=0.4)
    >>> fit                      # doctest: +SKIP
    >>> fit.summary()            # doctest: +SKIP
    >>> fit["tau_BC"], fit["ci_lower_BC"], fit["ci_upper_BC"]   # doctest: +SKIP
    """

    # Housekeeping
    if h is None:
        raise ValueError("Please supply a bandwidth h.")
    if q is None:
        q = p + 1
    if b is None:
        b = h

    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    d = np.asarray(fuzzy, dtype=float).ravel()
    if covs is not None:
        covs = np.asarray(covs, dtype=float)
        if covs.ndim == 1:
            covs = covs.reshape(-1, 1)
    if cluster is not None:
        cluster = np.asarray(cluster).ravel()

    # Check for errors/warnings
    _validate_inputs(y, x, d, p, q, h, b, covs, cluster, kernel, alpha_rule,
                     trace_weight, trace_target, vce, nnmatch, hc_lev,
                     vce_residuals, report_all_vce, alpha)

    # Drop incomplete observations
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(d)
    if covs is not None:
        ok = ok & np.all(np.isfinite(covs), axis=1)
    if cluster is not None:
        ok = ok & ~_is_missing(cluster)
    if not np.all(ok):
        y, x, d = y[ok], x[ok], d[ok]
        if covs is not None:
            covs = covs[ok, :]
        if cluster is not None:
            cluster = cluster[ok]

    ###################################################
    #### SUPPORTS, DESIGN AND SIDE-SPECIFIC FITS ####
    ###################################################

    sup = lfrd_supports(y, d, x, c, h, b, kernel, covs, cluster)
    dsg = lambda_class_design(sup, p)
    lp = lp_fits(sup, p, q, h, b)

    ###############################
    #### LAMBDA-CLASS ESTIMATE ####
    ###############################

    sel = select_lambda(dsg, lp, sup, p, _lambda, alpha_rule, psi,
                        trace_weight, trace_target)
    est = lambda_class_estimate(dsg, sel['lambda'])

    #########################
    #### BIAS CORRECTION ####
    #########################

    # The lambda-class has the same first-order smoothing bias as the
    # conventional estimator, so implement the bias estimator
    #   bias = lambda (tau_IV_D a_Y - tau_IV_Y a_D) / tau_D_lambda.
    bias_hat = sel['lambda'] * (est['tau_IV_D'] * lp['a_Y'] - est['tau_IV_Y'] * lp['a_D']) / est['tau_D_lambda']
    tau_BC = est['tau_lambda'] - bias_hat

    ##########################################
    #### INFLUENCE FUNCTIONS AND VARIANCE ####
    ##########################################

    inf = lfrd_influence(sup, dsg, est, lp, sel['lambda'])
    vr = lfrd_variance(sup, dsg, lp, inf, p, q, vce, nnmatch, nn_full_sample,
                       hc_lev, vce_residuals, report_all_vce)
    se_BC = np.sqrt(vr['var_BC'])
    se_conv = np.sqrt(vr['var_conv'])

    ###################################
    #### INFERENCE AND DIAGNOSTICS ####
    ###################################

    w1 = lfrd_wald(est['tau_lambda'], tau_BC, se_conv, se_BC, alpha, tau_0)
    dgn = lfrd_diagnostics(dsg, est, sel['a_trace'])
    bc = bias_constants(sup, h, p, kernel)

    ##################
    #### ASSEMBLE ####
    ##################

    # Rows: conventional, robust (bias-corrected)
    coef = np.array([est['tau_lambda'], tau_BC])
    se = np.array([se_conv, se_BC])
    ci = np.array([[w1['ci_lower_conv'], w1['ci_upper_conv']],
                   [w1['ci_lower_BC'], w1['ci_upper_BC']]])

    return LambdaFRDResults({
        'coef': coef, 'se': se, 'ci': ci,

        # Point estimates
        'tau_lambda': est['tau_lambda'],
        'tau_BC': tau_BC,
        'bias_hat': bias_hat,
        'lambda': sel['lambda'],

        # Standard errors
        'vce': vce,
        'se_conv': se_conv,
        'se_BC': se_BC,
        'var_conv': vr['var_conv'],
        'var_BC': vr['var_BC'],
        'var_all': vr['var_all'],
        'se_all': {k: np.sqrt(v) for k, v in vr['var_all'].items()},
        'var_conv_all': vr['var_conv_all'],
        'se_conv_all': {k: np.sqrt(v) for k, v in vr['var_conv_all'].items()},
        'n_clusters': vr['n_clusters'],

        # Intervals and t-statistics
        'alpha': alpha,
        'tau_0': tau_0,
        't_stat': w1['t_stat'],
        'ci_lower_BC': w1['ci_lower_BC'],
        'ci_upper_BC': w1['ci_upper_BC'],
        't_stat_conv': w1['t_stat_conv'],
        'ci_lower_conv': w1['ci_lower_conv'],
        'ci_upper_conv': w1['ci_upper_conv'],

        # Lambda selection
        'alpha_rule': alpha_rule,
        'psi': psi,
        'trace_weight': trace_weight,
        'trace_target': trace_target,
        'a_trace': sel['a_trace'],
        'a_trace_raw': sel['a_trace_raw'],
        'alpha_eff': sel['alpha_eff'],
        'trace_ratio_K': sel['ratio_K'],
        'trace_ratio': sel['ratio_use'],
        'CP_over_CQ': sel['CP_CQ'],
        'rho_cross': sel['rho_cross'],
        'bc_factor': sel['bc_factor'],
        'target_factor': sel['target_factor'],
        'kappa_N': sel['kappa_N'],
        'psi_star': sel['psi_star'],

        # Traces and first-stage diagnostics
        'C_P': dsg['C_P'],
        'C_Q': dsg['C_Q'],
        'L_eff': dsg['L_eff'],
        'q_eff': dsg['q_eff'],
        'G_M': dgn['G_M'],
        'G_P': dgn['G_P'],
        'G_R': dgn['G_R'],
        'F_stat': dgn['F_stat'],
        'F_can': dgn['F_can'],
        'sigma2_U': dgn['sigma2_U'],
        'sigma2_V': dgn['sigma2_V'],
        'sigma_UV': dgn['sigma_UV'],
        'rho2_UV': dgn['rho2_UV'],
        'ols_share': dgn['ols_share'],

        # IV moments and lambda-class components
        'tau_IV_Y': est['tau_IV_Y'],
        'tau_IV_D': est['tau_IV_D'],
        'pi_hat': est['pi_hat'],
        'ZtZ': dsg['ZtZ'],
        'tau_Y_lambda': est['tau_Y_lambda'],
        'tau_D_lambda': est['tau_D_lambda'],

        # Bias estimation
        'a_Y': lp['a_Y'],
        'a_D': lp['a_D'],
        'a_Y_raw': lp['a_Y_raw'],
        'a_D_raw': lp['a_D_raw'],
        'aY_plus': lp['aY_plus'],
        'aY_minus': lp['aY_minus'],
        'aD_plus': lp['aD_plus'],
        'aD_minus': lp['aD_minus'],
        'gamma_p': lp['gamma_p'],
        'c_bias_used': bc['used'],
        'c_bias_plus': bc['plus'],
        'c_bias_minus': bc['minus'],
        'c_bias_pop': bc['pop'],
        'c_bias_emp_plus': bc['emp_plus'],
        'c_bias_emp_minus': bc['emp_minus'],
        'side_sign': bc['side_sign'],

        # Influence functions and gradient
        'g': inf['g'],
        'coef_Y': inf['cY'],
        'coef_D': inf['cD'],
        'coef_Y_conv': inf['cY_conv'],
        'coef_D_conv': inf['cD_conv'],
        'influence': vr['xi_var'],
        'influence_conv': inf['xi_conv'],

        # Specification and sample sizes
        'c': c,
        'p': p,
        'q': q,
        'h': h,
        'b': b,
        'rho_bh': b / h,
        'kernel': kernel,
        'N': len(y),
        'N_h': {'left': int(np.sum(sup['Z'] == 0)), 'right': int(np.sum(sup['Z'] == 1))},
        'M': sup['M'],
        'M_bias': sup['M_b'],
        'M_union': sup['M_u'],
        'df': dsg['df'],
        'nnmatch': nnmatch,
        'hc_lev': hc_lev
    })
