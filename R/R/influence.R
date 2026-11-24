## =============================================================================
## INFLUENCE FUNCTIONS
##
## Bias-corrected estimator
##   tau_BC = tau_Y_lambda / tau_D_lambda
##            - lambda (tau_IV_D a_Y - tau_IV_Y a_D) / tau_D_lambda,
## linearised by the delta method in
##   theta = (tau_Y_lambda, tau_D_lambda, a_Y, a_D, tau_IV_Y, tau_IV_D).
## The derivatives through the IV moments in the bias term are dropped
## (g5 = g6 = 0), matching the robust bias-corrected variance convention.
##
## All scores live on the UNION support (see supports.R).  The conventional
## (uncorrected) score lives on the main window.
## =============================================================================

lfrd_influence <- function(sup, dsg, est, lp, lambda_val) {

  M_u   <- sup$M_u
  idx_m <- sup$idx_main_in_union
  dZ    <- lp$dZ

  ## ---- gradient vector for the delta method ----
  tYl     <- est$tau_Y_lambda
  tDl     <- est$tau_D_lambda
  C_const <- lambda_val

  g1 <- 1 / tDl
  g2 <- -tYl / tDl^2
  g3 <- -C_const * est$tau_IV_D / tDl
  g4 <-  C_const * est$tau_IV_Y / tDl
  g5 <- 0
  g6 <- 0

  ## ---- side-specific residuals, leverages and jump weights on the union ----
  ncolD <- ncol(lp$Dmat_u)
  res_p_u <- matrix(0, nrow = M_u, ncol = ncolD)
  res_q_u <- matrix(0, nrow = M_u, ncol = ncolD)
  ell_p_u <- ell_bc_u <- hlev_p_u <- hlev_q_u <- numeric(M_u)
  for (s in list(lp$left, lp$right)) {
    ii <- s$ii
    res_p_u[ii, ] <- s$res_p
    res_q_u[ii, ] <- s$res_q
    ell_p_u[ii]   <- s$ell_p
    ell_bc_u[ii]  <- s$ell_bc
    hlev_p_u[ii]  <- s$hii_p
    hlev_q_u[ii]  <- s$hii_q
  }

  bias_weight_u <- ell_p_u - ell_bc_u
  bias_weight_u[sup$Z_u == 0] <- -bias_weight_u[sup$Z_u == 0]

  sw_u            <- numeric(M_u)
  sw_u[idx_m]     <- dsg$sw
  ImLMD_u         <- numeric(M_u)
  ImLMD_u[idx_m]  <- as.numeric(est$ImLMD)
  Ztilde_u        <- numeric(M_u)
  Ztilde_u[idx_m] <- as.numeric(dsg$Z_tilde)

  ## ---- bias-corrected score coefficients ----
  cY <- (g1 * ImLMD_u + g5 * Ztilde_u) * sw_u + g3 * bias_weight_u
  cD <- (g2 * ImLMD_u + g6 * Ztilde_u) * sw_u + g4 * bias_weight_u
  cW <- if (dZ > 0L) {
    -outer(cY, lp$gamma_Y) - outer(cD, lp$gamma_D)
  } else {
    matrix(numeric(0), nrow = M_u, ncol = 0)
  }

  xi <- cY * res_q_u[, 1] + cD * res_q_u[, 2]
  if (dZ > 0L) xi <- xi + rowSums(cW * res_q_u[, 3:ncolD, drop = FALSE])

  eY_p <- res_p_u[idx_m, 1]
  eD_p <- res_p_u[idx_m, 2]

  ## ---- conventional lambda-class score, with the pooled covariate adjustment ----
  Pz_Y <- as.numeric(dsg$Z_tilde) *
    (dsg$ZtZ_inv * as.numeric(crossprod(dsg$Z_tilde, dsg$Y_tilde)))
  K_Y <- (1 - lambda_val) * as.numeric(dsg$Y_tilde) + lambda_val * Pz_Y
  cY_conv <- dsg$sw * as.numeric(est$ImLMD) / tDl
  cD_conv <- dsg$sw * (K_Y - 2 * est$tau_lambda * as.numeric(est$ImLMD)) / tDl
  xi_conv <- cY_conv * eY_p + cD_conv * eD_p
  cW_conv <- NULL
  if (dZ > 0L) {
    eW_p <- res_p_u[idx_m, 3:ncolD, drop = FALSE]
    cW_conv <- -outer(cY_conv, lp$gamma_Y) - outer(cD_conv, lp$gamma_D)
    xi_conv <- xi_conv + rowSums(cW_conv * eW_p)
  }

  ## ---- residuals entering the BC variance: order-q (default) or order-p ----
  xi_var_q <- xi
  xi_var_p <- cY * res_p_u[, 1] + cD * res_p_u[, 2]
  if (dZ > 0L) xi_var_p <- xi_var_p + rowSums(cW * res_p_u[, 3:ncolD, drop = FALSE])

  ## ---- lambda = 1: exact covariate-adjusted fuzzy-RD score (rdrobust) ----
  ## Inference-only nesting branch: tau_lambda and the IV moment structure are
  ## not replaced or re-estimated.
  rdrobust_nesting <- isTRUE(all.equal(lambda_val, 1, tolerance = 1e-12))
  s_rd <- c(1 / lp$tau_D_cl,
            -lp$tau_Y_cl / lp$tau_D_cl^2,
            -lp$gamma_Y / lp$tau_D_cl + (lp$tau_Y_cl / lp$tau_D_cl^2) * lp$gamma_D)
  score_p_u <- as.numeric(res_p_u %*% s_rd)
  score_q_u <- as.numeric(res_q_u %*% s_rd)
  if (rdrobust_nesting) {
    xi_conv  <- (ell_p_u * score_p_u)[idx_m]
    xi_var_q <- xi_var_p <- ell_bc_u * score_q_u
  }

  list(g = c(g1, g2, g3, g4, g5, g6),
       cY = cY, cD = cD, cW = cW,
       cY_conv = cY_conv, cD_conv = cD_conv, cW_conv = cW_conv,
       xi_var_q = xi_var_q, xi_var_p = xi_var_p, xi_conv = xi_conv,
       res_p_u = res_p_u, res_q_u = res_q_u, eY_p = eY_p, eD_p = eD_p,
       ell_p_u = ell_p_u, ell_bc_u = ell_bc_u,
       hlev_q = pmin(hlev_q_u, 0.99),
       hlev_p = pmin(hlev_p_u[idx_m], 0.99),
       rdrobust_nesting = rdrobust_nesting, s_rd = s_rd)
}
