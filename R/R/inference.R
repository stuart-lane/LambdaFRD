## =============================================================================
## INFERENCE: Wald intervals and first-stage diagnostics
## =============================================================================


## -----------------------------------------------------------------------------
## Delta-method (Wald) intervals and t-statistics for tau = tau_0
## -----------------------------------------------------------------------------

lfrd_wald <- function(tau_lambda, tau_BC, se_conv, se_BC, alpha, tau_0) {
  z <- qnorm(1 - alpha / 2)
  list(t_stat        = (tau_BC - tau_0) / se_BC,
       ci_lower_BC   = tau_BC - z * se_BC,
       ci_upper_BC   = tau_BC + z * se_BC,
       t_stat_conv   = (tau_lambda - tau_0) / se_conv,
       ci_lower_conv = tau_lambda - z * se_conv,
       ci_upper_conv = tau_lambda + z * se_conv)
}


## -----------------------------------------------------------------------------
## First-stage / trace diagnostics
##   F_stat : first-stage F with the homoskedastic M_W variance
##   F_can  : G_P / C_P, the F-statistic in canonical (trace) units
## -----------------------------------------------------------------------------

lfrd_diagnostics <- function(dsg, est, a_trace) {

  df <- dsg$df
  U_hat <- dsg$Y_tilde - est$tau_lambda * dsg$D_tilde
  V_hat <- est$Mz_D

  G_M <- as.numeric(crossprod(est$Mz_D))
  G_P <- est$tau_IV_D^2 / dsg$ZtZ
  G_R <- G_M + G_P

  sigma2_U <- as.numeric(crossprod(U_hat)) / df
  sigma2_V <- G_M / df
  sigma_UV <- as.numeric(crossprod(U_hat, V_hat)) / df
  rho2_UV  <- sigma_UV^2 / (sigma2_U * sigma2_V + 1e-300)
  F_stat   <- G_P / (sigma2_V + 1e-300)

  list(G_M = G_M, G_P = G_P, G_R = G_R,
       sigma2_U = sigma2_U, sigma2_V = sigma2_V, sigma_UV = sigma_UV,
       rho2_UV = rho2_UV, F_stat = F_stat, F_can = G_P / dsg$C_P,
       ols_share = as.numeric(a_trace * G_M / (G_P + a_trace * G_M)))
}
