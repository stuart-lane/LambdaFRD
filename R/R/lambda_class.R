## =============================================================================
## lambda-CLASS DESIGN AND ESTIMATOR
##
## The local fuzzy-RD model on the main window is written as a just-identified
## IV regression
##
##   Y = tau D + V gamma + u,      D = pi Z + V delta + v,
##
## with V = [1, (1-Z) Xc^j, Z Xc^j (j = 1..p), covariates] and instrument
## Z = 1{x >= c}, all weighted by the kernel.  V is partialled out
## (Frisch-Waugh), after which the lambda-class estimator is
##
##   tau_lambda = D'(I - lambda M_z) Y / D'(I - lambda M_z) D.
##
## on the partialled data. lambda = 1 is the conventional fuzzy-RD estimator 
## and lambda = 0 is local OLS sharp RD estimator.
## =============================================================================


## -----------------------------------------------------------------------------
## Partialled design, projection diagonals and trace quantities
## -----------------------------------------------------------------------------

lambda_class_design <- function(sup, p) {

  Y  <- sup$Y 
  D  <- sup$D
  Z  <- sup$Z
  Xc <- sup$Xc
  w  <- sup$w 
  M  <- sup$M

  ## ---- V matrix ----
  V <- matrix(1, nrow = M, ncol = 1)
  for (j in seq_len(p)) V <- cbind(V, (1 - Z) * Xc^j, Z * Xc^j)
  if (!is.null(sup$W)) V <- cbind(V, sup$W)
  df <- M - ncol(V) - 1
  
  if (df <= 0)
    stop("Too few observations in the bandwidth.", call. = FALSE)

  ## ---- kernel weights (sqrt-scaling convention) ----
  sw <- sqrt(w)
  Vs <- sw * V
  Zs <- sw * Z
  Ys <- sw * Y
  Ds <- sw * D

  ## ---- partial out V ----
  VtV_inv <- solve(crossprod(Vs))
  partial_out <- function(v) v - Vs %*% (VtV_inv %*% crossprod(Vs, v))

  Y_tilde <- partial_out(Ys)
  D_tilde <- partial_out(Ds)
  Z_tilde <- partial_out(Zs)

  ZtZ <- as.numeric(crossprod(Z_tilde))
  
  if (!is.finite(ZtZ) || ZtZ <= 0)
    stop("Partialled instrument has no variation.", call. = FALSE)
  ZtZ_inv <- 1 / ZtZ

  ## ---- projection diagonals: H (polynomial controls), P (instrument), Q (residual) ----
  H_diag   <- rowSums(Vs %*% VtV_inv * Vs)
  P_ii     <- as.numeric(Z_tilde)^2 * ZtZ_inv
  Q_ii_raw <- 1 - H_diag - P_ii
  
  if (min(Q_ii_raw) < -1e-8)
    warning("Some Q diagonal elements are materially negative.", call. = FALSE)
  Q_ii <- pmax(Q_ii_raw, 0)

  ## ---- linear first-stage fitted probabilities, for variance traces ----
  fs_fit <- tryCatch(
    qr.fitted(qr(cbind(Vs, Zs)), Ds),
    error = function(e) rep(NA_real_, M)
  )
  
  pi_i <- if (anyNA(fs_fit)) rep(mean(D), M) else fs_fit / pmax(sw, 1e-12)
  pi_i <- pmin(pmax(pi_i, 1e-4), 1 - 1e-4)

  ## ---- traces: C_P = tr(P Omega), C_Q = tr(Q Omega), L = tr(PK), q = tr(QK) ----
  c_i   <- w * pi_i * (1 - pi_i)
  C_P   <- sum(P_ii * c_i)
  C_Q   <- sum(Q_ii * c_i)
  L_eff <- sum(P_ii * w)
  q_eff <- sum(Q_ii * w)

  if (!is.finite(C_P) || C_P < 0)
    stop("C_P is invalid.", call. = FALSE)
  if (!is.finite(C_Q) || C_Q <= 0)
    stop("C_Q <= 0: trace-Fuller lambda cannot be formed.", call. = FALSE)
  if (!is.finite(L_eff) || L_eff <= 0 || !is.finite(q_eff) || q_eff <= 0)
    stop("Kernel traces tr(PK), tr(QK) are not positive.", call. = FALSE)

  list(sw = sw, df = df, Y_tilde = Y_tilde, D_tilde = D_tilde, Z_tilde = Z_tilde,
       ZtZ = ZtZ, ZtZ_inv = ZtZ_inv, H_diag = H_diag, P_ii = P_ii, Q_ii = Q_ii,
       pi_i = pi_i, C_P = C_P, C_Q = C_Q, L_eff = L_eff, q_eff = q_eff)
}


## -----------------------------------------------------------------------------
## lambda-class estimate, plus the IV moments
## -----------------------------------------------------------------------------

lambda_class_estimate <- function(dsg, lambda) {

  Z_tilde <- dsg$Z_tilde
  D_tilde <- dsg$D_tilde
  Y_tilde <- dsg$Y_tilde

  Mz_D  <- D_tilde - Z_tilde * (dsg$ZtZ_inv * sum(Z_tilde * D_tilde))
  ImLMD <- D_tilde - lambda * Mz_D

  tau_Y_lambda <- as.numeric(crossprod(ImLMD, Y_tilde))
  tau_D_lambda <- as.numeric(crossprod(ImLMD, D_tilde))
  if (!is.finite(tau_D_lambda) || abs(tau_D_lambda) < .Machine$double.eps)
    stop("lambda-class denominator is numerically zero.", call. = FALSE)

  tau_IV_D <- as.numeric(crossprod(Z_tilde, D_tilde))
  tau_IV_Y <- as.numeric(crossprod(Z_tilde, Y_tilde))

  list(Mz_D = Mz_D, ImLMD = ImLMD,
       tau_Y_lambda = tau_Y_lambda, tau_D_lambda = tau_D_lambda,
       tau_lambda = tau_Y_lambda / tau_D_lambda,
       tau_IV_Y = tau_IV_Y, tau_IV_D = tau_IV_D,
       pi_hat = tau_IV_D / dsg$ZtZ)
}
