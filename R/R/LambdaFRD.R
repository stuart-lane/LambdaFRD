#' \eqn{\lambda}-class estimation and inference for fuzzy regression discontinuity
#'
#' Estimates the FRD effect at the cutoff with a \eqn{\lambda}-class
#' estimator and provides robust bias-corrected confidence intervals. 
#' This estimator class has finite moments up to the order
#' of the error distribution, and is shown to have good estimator and 
#' inference properties in Lane (2025). \eqn{\lambda = 1} is the conventional
#' FRD estimator. As in \code{rdrobust}, a single call returns the conventional
#' estimate with its standard error and the bias-corrected estimate with 
#' its robust standard error and confidence interval. Consult "The moment is
#' here: a generalized class of estimators for fuzzy regression discontinuity
#' designs" Lane (2025) for more details.
#'
#' @param y Outcome vector.
#' @param x Running variable.
#' @param fuzzy Treatment receipt vector (the fuzzy-RD treatment indicator).
#' @param c Cutoff. Default 0.
#' @param p Order of the local polynomial used for the point estimate.
#'   Default 1.
#' @param q Order of the local polynomial used for bias correction. Must be
#'   greater than \code{p}. Default \code{p + 1}.
#' @param h Main bandwidth (required).
#' @param b Bias bandwidth. Default \code{h}.
#' @param covs Optional matrix or data frame of covariates, included linearly
#'   in the lambda-class fit and partialled out of the bias fits as in
#'   \code{rdrobust}.
#' @param kernel One of \code{"triangular"} (default), \code{"uniform"},
#'   \code{"epanechnikov"}.
#' @param lambda Optional fixed value of \eqn{\lambda}.  When \code{NULL}
#'   (default) \eqn{\lambda} is chosen by \code{alpha_rule}. When supplied,
#'   the rule-based quantities are still reported but \code{lambda} is used as
#'   given (e.g. \code{lambda = 1} for the conventional FRD estimator and
#'   \code{lambda = 0} for the conventional sharp RD estimator).
#' @param alpha_rule Rule for selecting \code{lambda}.
#'   \eqn{1 - \lambda = \alpha \times} trace ratio: \code{"trace"} (default,
#'   \eqn{\alpha = \psi}) or \code{"psi_df"} (\eqn{1 - \lambda = \psi / (M -
#'   2(p+1))}).
#' @param psi Multiplier for the \code{"trace"} and \code{"psi_df"} rules.
#'   Default 1.
#' @param trace_weight Traces converting \eqn{\alpha} into \eqn{1 - \lambda}:
#'   \code{"kernel"} (default, \eqn{tr(PK)/tr(QK)}) or \code{"variance"}
#'   (first-stage-variance weighted).
#' @param trace_target Bias target: \code{"uncorrected"} (default, removes the
#'   leading ratio bias of the conventional lambda-class estimator), \code{"bc"}
#'   (removes it for the bias-corrected estimator, factor
#'   \eqn{\max(2\rho_x - 1, 1)}) or \code{"bc_raw"} (the same factor without
#'   the floor at 1).
#' @param vce Variance estimator: \code{"nn"} (default), \code{"hc0"},
#'   \code{"hc1"}, \code{"hc2"}, \code{"hc3"}, \code{"cluster"}, \code{"homo"}.
#' @param nnmatch Minimum number of nearest-neighbour matches for
#'   \code{vce = "nn"}. Default 3.
#' @param cluster Cluster identifiers, required for \code{vce = "cluster"}.
#' @param nn_full_sample If \code{TRUE} (default) nearest-neighbour residuals
#'   are built from the full sample on each side, as in \code{rdrobust}.
#' @param hc_lev Leverages for the HC2/HC3 corrections when
#'   \code{vce_residuals = "p"}: \code{"controls"} (default) or \code{"full"}.
#' @param vce_residuals Residuals entering the bias-corrected variance:
#'   \code{"q"} (default, order-q fit, as in \code{rdrobust}) or \code{"p"}.
#' @param report_all_vce If \code{TRUE}, compute every variance estimator
#'   (returned in \code{var_all} / \code{var_conv_all}).
#' @param alpha Significance level for the confidence intervals. Default 0.05.
#' @param tau_0 Null value for the reported t-statistics. Default 0.
#'
#' @return An object of class \code{"LambdaFRD"}: a list containing, among
#'   others,
#'   \item{coef, se, ci}{matrices with rows \code{"Conventional"} and
#'     \code{"Robust"} (bias-corrected estimate, robust standard error).}
#'   \item{tau_lambda, se_conv, ci_lower_conv, ci_upper_conv}{Conventional
#'     lambda-class estimate, standard error and interval.}
#'   \item{tau_BC, se_BC, ci_lower_BC, ci_upper_BC}{Bias-corrected estimate,
#'     robust standard error and interval.}
#'   \item{lambda}{The value of \eqn{\lambda} used.}
#'   \item{F_stat, F_can, G_P, C_P, kappa_N, psi_star, rho_cross}{First-stage
#'     and trace diagnostics.}
#'
#' @examples
#' set.seed(1)
#' n <- 1000
#' x <- runif(n, -1, 1)
#' d <- as.numeric(rnorm(n) < -0.5 + 1.0 * (x >= 0))
#' y <- 0.5 * x + 1 * d + rnorm(n, sd = 0.3)
#' fit <- LambdaFRD(y, x, fuzzy = d, h = 0.4)
#' fit
#' summary(fit)
#'
#' @export
LambdaFRD <- function(
    y, x, fuzzy, c = 0,
    p = 1, q = p + 1, h, b = h,
    covs = NULL, kernel = "triangular",
    ## ---- lambda selection ---------------------------------------------------
    lambda = NULL,
    alpha_rule = "trace", psi = 1,
    trace_weight = "kernel", trace_target = "uncorrected",
    ## ---- variance -----------------------------------------------------------
    vce = "nn", nnmatch = 3, cluster = NULL, nn_full_sample = TRUE,
    hc_lev = "controls", vce_residuals = "q", report_all_vce = FALSE,
    ## ---- inference ----------------------------------------------------------
    alpha = 0.05, tau_0 = 0
) {

  cl <- match.call()

  ## ---------------------------------------------------------------------------
  ## 0. INPUT CHECKS
  ## ---------------------------------------------------------------------------

  if (missing(y) || missing(x) || missing(fuzzy))
    stop("Please supply the outcome y, running variable x and treatment fuzzy.", call. = FALSE)
  if (missing(h) || is.null(h))
    stop("Please supply a bandwidth h.", call. = FALSE)

  y <- as.numeric(y); x <- as.numeric(x); d <- as.numeric(fuzzy)
  if (length(y) != length(d) || length(y) != length(x))
    stop("y, x and fuzzy must be the same length.", call. = FALSE)
  if (!is.null(covs)) {
    covs <- as.matrix(covs)
    if (nrow(covs) != length(y)) stop("covs must have one row per observation.", call. = FALSE)
  }
  if (!is.null(cluster) && length(cluster) != length(y))
    stop("cluster must be the same length as y.", call. = FALSE)

  if (p %% 1 != 0 || p < 1)
    stop("p must be a positive integer (p >= 1 required for bias correction).", call. = FALSE)
  if (q %% 1 != 0 || q <= p)
    stop("q must be an integer greater than p.", call. = FALSE)
  if (!is.numeric(h) || length(h) != 1L || !is.finite(h) || h <= 0)
    stop("h must be a positive, finite number.", call. = FALSE)
  if (!is.numeric(b) || length(b) != 1L || !is.finite(b) || b <= 0)
    stop("b must be a positive, finite number.", call. = FALSE)
  if (b < h)
    warning("b < h, so bias estimates will use a narrower support than the main fit.", call. = FALSE)

  if (!is.numeric(alpha) || length(alpha) != 1L || alpha <= 0 || alpha >= 1)
    stop("alpha must be a single number strictly between 0 and 1.", call. = FALSE)

  kernel        <- match.arg(kernel, c("triangular", "uniform", "epanechnikov"))
  alpha_rule    <- match.arg(alpha_rule, c("trace", "psi_df"))
  trace_weight  <- match.arg(trace_weight, c("kernel", "variance"))
  trace_target  <- match.arg(trace_target, c("uncorrected", "bc", "bc_raw"))
  vce           <- match.arg(vce, c("nn", "hc0", "hc1", "hc2", "hc3", "cluster", "homo"))
  hc_lev        <- match.arg(hc_lev, c("controls", "full"))
  vce_residuals <- match.arg(vce_residuals, c("q", "p"))

  if (vce == "cluster" && is.null(cluster))
    stop("vce = 'cluster' requires a cluster ID vector.", call. = FALSE)
  if (!is.null(cluster) && vce != "cluster" && !report_all_vce)
    warning("cluster supplied but vce != 'cluster'; it will be ignored.", call. = FALSE)
  if (nnmatch < 1) stop("nnmatch must be at least 1.", call. = FALSE)

  ## ---- drop incomplete observations ----
  ok <- is.finite(y) & is.finite(x) & is.finite(d)
  if (!is.null(covs))    ok <- ok & rowSums(!is.finite(covs)) == 0
  if (!is.null(cluster)) ok <- ok & !is.na(cluster)
  if (!all(ok)) {
    y <- y[ok]; x <- x[ok]; d <- d[ok]
    if (!is.null(covs))    covs <- covs[ok, , drop = FALSE]
    if (!is.null(cluster)) cluster <- cluster[ok]
  }

  ## ---------------------------------------------------------------------------
  ## 1. SUPPORTS, LAMBDA-CLASS DESIGN AND SIDE-SPECIFIC FITS
  ## ---------------------------------------------------------------------------

  sup <- lfrd_supports(y, d, x, c, h, b, kernel, covs, cluster)
  dsg <- lambda_class_design(sup, p)
  lp  <- lp_fits(sup, p, q, h, b)

  ## ---------------------------------------------------------------------------
  ## 2. LAMBDA-CLASS ESTIMATE
  ## ---------------------------------------------------------------------------

  sel <- select_lambda(dsg, lp, sup, p, lambda, alpha_rule, psi,
                       trace_weight, trace_target)
  est <- lambda_class_estimate(dsg, sel$lambda)

  ## ---------------------------------------------------------------------------
  ## 3. BIAS CORRECTION
  ##
  ## The lambda-class has the same first-rder smoothing bias as the
  ## conventional estimator, so implement the bias estimator
  ##   bias = lambda (tau_IV_D a_Y - tau_IV_Y a_D) / tau_D_lambda.
  ## ---------------------------------------------------------------------------

  bias_hat <- sel$lambda * (est$tau_IV_D * lp$a_Y - est$tau_IV_Y * lp$a_D) / est$tau_D_lambda
  tau_BC <- est$tau_lambda - bias_hat

  ## ---------------------------------------------------------------------------
  ## 4. INFLUENCE FUNCTIONS AND VARIANCE
  ## ---------------------------------------------------------------------------

  inf <- lfrd_influence(sup, dsg, est, lp, sel$lambda)
  vr  <- lfrd_variance(sup, dsg, lp, inf, p, q, vce, nnmatch, nn_full_sample,
                       hc_lev, vce_residuals, report_all_vce)
  se_BC   <- sqrt(vr$var_BC)
  se_conv <- sqrt(vr$var_conv)

  ## ---------------------------------------------------------------------------
  ## 5. INFERENCE AND DIAGNOSTICS
  ## ---------------------------------------------------------------------------

  w1  <- lfrd_wald(est$tau_lambda, tau_BC, se_conv, se_BC, alpha, tau_0)
  dgn <- lfrd_diagnostics(dsg, est, sel$a_trace)
  bc  <- bias_constants(sup, h, p, kernel)

  ## ---------------------------------------------------------------------------
  ## 6. ASSEMBLE
  ## ---------------------------------------------------------------------------

  coef_mat <- matrix(c(est$tau_lambda, tau_BC), ncol = 1,
                     dimnames = list(c("Conventional", "Robust"), "Coeff"))
  se_mat   <- matrix(c(se_conv, se_BC), ncol = 1,
                     dimnames = list(c("Conventional", "Robust"), "Std. Err."))
  ci_mat   <- matrix(c(w1$ci_lower_conv, w1$ci_lower_BC, w1$ci_upper_conv, w1$ci_upper_BC),
                     ncol = 2, dimnames = list(c("Conventional", "Robust"),
                                               c("CI Lower", "CI Upper")))

  out <- list(
    call = cl,

    coef = coef_mat, se = se_mat, ci = ci_mat,

    # Point estimates
    tau_lambda = as.numeric(est$tau_lambda),
    tau_BC     = as.numeric(tau_BC),
    bias_hat   = as.numeric(bias_hat),
    lambda     = as.numeric(sel$lambda),

    # Standard errors
    vce          = vce,
    se_conv      = se_conv,
    se_BC        = se_BC,
    var_conv     = vr$var_conv,
    var_BC       = vr$var_BC,
    var_all      = vr$var_all,
    se_all       = sqrt(vr$var_all),
    var_conv_all = vr$var_conv_all,
    se_conv_all  = sqrt(vr$var_conv_all),
    n_clusters   = vr$n_clusters,

    # Intervals and t-statistics
    alpha         = alpha,
    tau_0         = tau_0,
    t_stat        = w1$t_stat,
    ci_lower_BC   = w1$ci_lower_BC,
    ci_upper_BC   = w1$ci_upper_BC,
    t_stat_conv   = w1$t_stat_conv,
    ci_lower_conv = w1$ci_lower_conv,
    ci_upper_conv = w1$ci_upper_conv,

    # Lambda selection
    alpha_rule    = alpha_rule,
    psi           = psi,
    trace_weight  = trace_weight,
    trace_target  = trace_target,
    a_trace       = as.numeric(sel$a_trace),
    a_trace_raw   = as.numeric(sel$a_trace_raw),
    alpha_eff     = as.numeric(sel$alpha_eff),
    trace_ratio_K = as.numeric(sel$ratio_K),
    trace_ratio   = as.numeric(sel$ratio_use),
    CP_over_CQ    = as.numeric(sel$CP_CQ),
    rho_cross     = as.numeric(sel$rho_cross),
    bc_factor     = as.numeric(sel$bc_factor),
    target_factor = as.numeric(sel$target_factor),
    kappa_N       = as.numeric(sel$kappa_N),
    psi_star      = as.numeric(sel$psi_star),

    # Traces and first-stage diagnostics
    C_P       = dsg$C_P,
    C_Q       = dsg$C_Q,
    L_eff     = dsg$L_eff,
    q_eff     = dsg$q_eff,
    G_M       = dgn$G_M,
    G_P       = dgn$G_P,
    G_R       = dgn$G_R,
    F_stat    = dgn$F_stat,
    F_can     = dgn$F_can,
    sigma2_U  = dgn$sigma2_U,
    sigma2_V  = dgn$sigma2_V,
    sigma_UV  = dgn$sigma_UV,
    rho2_UV   = dgn$rho2_UV,
    ols_share = dgn$ols_share,

    # IV moments and lambda-class components
    tau_IV_Y     = est$tau_IV_Y,
    tau_IV_D     = est$tau_IV_D,
    pi_hat       = est$pi_hat,
    ZtZ          = dsg$ZtZ,
    tau_Y_lambda = est$tau_Y_lambda,
    tau_D_lambda = est$tau_D_lambda,

    # Bias estimation
    a_Y      = lp$a_Y,
    a_D      = lp$a_D,
    a_Y_raw  = lp$a_Y_raw,
    a_D_raw  = lp$a_D_raw,
    aY_plus  = lp$aY_plus,
    aY_minus = lp$aY_minus,
    aD_plus  = lp$aD_plus,
    aD_minus = lp$aD_minus,
    gamma_p  = lp$gamma_p,
    c_bias_used      = bc$used,
    c_bias_plus      = bc$plus,
    c_bias_minus     = bc$minus,
    c_bias_pop       = bc$pop,
    c_bias_emp_plus  = bc$emp_plus,
    c_bias_emp_minus = bc$emp_minus,
    side_sign        = bc$side_sign,

    # Influence functions and gradient
    g              = inf$g,
    coef_Y         = as.numeric(inf$cY),
    coef_D         = as.numeric(inf$cD),
    coef_Y_conv    = as.numeric(inf$cY_conv),
    coef_D_conv    = as.numeric(inf$cD_conv),
    influence      = as.numeric(vr$xi_var),
    influence_conv = as.numeric(inf$xi_conv),

    # Specification and sample sizes
    c         = c,
    p         = p,
    q         = q,
    h         = h,
    b         = b,
    rho_bh    = b / h,
    kernel    = kernel,
    N         = length(y),
    N_h       = c(left = sum(sup$Z == 0), right = sum(sup$Z == 1)),
    M         = sup$M,
    M_bias    = sup$M_b,
    M_union   = sup$M_u,
    df        = dsg$df,
    nnmatch   = nnmatch,
    hc_lev    = hc_lev
  )

  class(out) <- "LambdaFRD"
  out
}
