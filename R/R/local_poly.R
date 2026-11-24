## =============================================================================
## SIDE-SPECIFIC LOCAL POLYNOMIAL FITS FOR BIAS CORRECTION
##
## The lambda-class estimator is left unchanged.  This builds side-specific order-p
## (bandwidth h) and order-q (bandwidth b) local-polynomial objects only for
## bias correction and inference, following the CCT / rdrobust
## equivalent-kernel algebra, including any pooled covariate adjustment.
##
## The bias-corrected estimates uses the (p+1)-th coefficient of the order-q fit:
##
##   Q_eq = (R_p W_h)' - h^{p+1} L_p e_{p+1}' G_q^{-1} (R_q W_b)',
##   L_p  = (R_p W_h)' (x / h)^{p+1},
##
## so q = p + 1 is the usual choice and q > p + 1 is also supported.  
## Everything is built on the UNION support (see supports.R).
## =============================================================================


## -----------------------------------------------------------------------------
## One side of the cutoff
## -----------------------------------------------------------------------------

lp_side <- function(sup, side, Dmat_u, p, q, h) {

  ii   <- (sup$Z_u == side)
  xc_s <- sup$Xc_u[ii]
  wh_s <- sup$w_h_u[ii]
  wb_s <- sup$w_b_u[ii]
  data_s <- Dmat_u[ii, , drop = FALSE]

  Rp <- build_poly(xc_s, p)
  Rq <- build_poly(xc_s, q)
  invGp <- solve(crossprod(Rp, Rp * wh_s))
  invGq <- solve(crossprod(Rq, Rq * wb_s))

  beta_p <- invGp %*% crossprod(Rp * wh_s, data_s)
  beta_q <- invGq %*% crossprod(Rq * wb_s, data_s)

  Lp <- crossprod(Rp * wh_s, (xc_s / h)^(p + 1))
  e_p1 <- numeric(q + 1); e_p1[p + 2] <- 1
  q_score <- invGq %*% (t(Rq) * rep(wb_s, each = q + 1))
  Qeq <- t(t(Rp * wh_s) - h^(p + 1) * (Lp %*% t(e_p1)) %*% q_score)
  beta_bc <- invGp %*% crossprod(Qeq, data_s)

  res_p <- data_s - Rp %*% beta_p
  res_q <- data_s - Rq %*% beta_q

  hii_p <- rowSums((Rp %*% invGp) * (Rp * wh_s))
  hii_q <- rowSums((Rq %*% invGq) * (Rq * wb_s))
  e0 <- numeric(p + 1); e0[1] <- 1
  ell_p  <- as.numeric((Rp * wh_s) %*% (invGp %*% e0))   # conventional jump weights
  ell_bc <- as.numeric(Qeq %*% (invGp %*% e0))           # bias-corrected jump weights

  list(ii = ii, Rp = Rp, invGp = invGp,
       beta_p = beta_p, beta_q = beta_q, beta_bc = beta_bc,
       res_p = res_p, res_q = res_q, hii_p = hii_p, hii_q = hii_q,
       ell_p = ell_p, ell_bc = ell_bc)
}


## -----------------------------------------------------------------------------
## Both sides, covariate adjustment and the estimated bias of the jumps
## -----------------------------------------------------------------------------

lp_fits <- function(sup, p, q, h, b) {

  W_u <- sup$W_u
  if (is.null(W_u)) {
    Dmat_u <- cbind(sup$Y_u, sup$D_u)
    dZ <- 0L
  } else {
    Dmat_u <- cbind(sup$Y_u, sup$D_u, W_u)
    dZ <- ncol(W_u)
  }

  left  <- lp_side(sup, 0, Dmat_u, p, q, h)
  right <- lp_side(sup, 1, Dmat_u, p, q, h)
  beta_p_jump  <- right$beta_p  - left$beta_p
  beta_bc_jump <- right$beta_bc - left$beta_bc

  ## ---- pooled covariate adjustment (rdrobust) ----
  if (dZ > 0L) {
    covariate_cross <- function(lp) {
      ii <- lp$ii
      Wcov <- W_u[ii, , drop = FALSE]
      data_s <- Dmat_u[ii, , drop = FALSE]
      wh_s <- sup$w_h_u[ii]
      Up <- crossprod(lp$Rp * wh_s, data_s)
      ZWD <- crossprod(Wcov * wh_s, data_s)
      UpZ <- Up[, 3:ncol(Up), drop = FALSE]
      UiGU <- t(UpZ) %*% lp$invGp %*% Up
      list(ZWZ = ZWD[, 3:ncol(ZWD), drop = FALSE] - UiGU[, 3:ncol(UiGU), drop = FALSE],
           ZWY = ZWD[, 1:2, drop = FALSE] - UiGU[, 1:2, drop = FALSE])
    }
    cc_l <- covariate_cross(left)
    cc_r <- covariate_cross(right)
    gamma_p <- qr.solve(cc_l$ZWZ + cc_r$ZWZ, cc_l$ZWY + cc_r$ZWY, tol = 1e-20)
    gamma_Y <- gamma_p[, 1]
    gamma_D <- gamma_p[, 2]
  } else {
    gamma_p <- matrix(numeric(0), nrow = 0, ncol = 2)
    gamma_Y <- numeric(0)
    gamma_D <- numeric(0)
  }

  adj_jump <- function(jump, col, gam) {
    as.numeric(jump[1, col] -
                 if (dZ > 0L) crossprod(gam, jump[1, 3:ncol(jump)]) else 0)
  }
  tau_Y_cl <- adj_jump(beta_p_jump,  1, gamma_Y)
  tau_D_cl <- adj_jump(beta_p_jump,  2, gamma_D)
  tau_Y_bc <- adj_jump(beta_bc_jump, 1, gamma_Y)
  tau_D_bc <- adj_jump(beta_bc_jump, 2, gamma_D)

  ## ---- leading (p+1)-th coefficients of the order-q fit, in b^(p+1) units ----
  k1 <- p + 2
  lead <- function(lp, col) as.numeric(lp$beta_q[k1, col] * b^(p + 1))

  list(left = left, right = right, Dmat_u = Dmat_u, dZ = dZ,
       gamma_p = gamma_p, gamma_Y = gamma_Y, gamma_D = gamma_D,
       tau_Y_cl = tau_Y_cl, tau_D_cl = tau_D_cl,
       ## estimated bias of the reduced-form and first-stage jumps
       a_Y = tau_Y_cl - tau_Y_bc,
       a_D = tau_D_cl - tau_D_bc,
       a_Y_raw = as.numeric((right$beta_q[k1, 1] - left$beta_q[k1, 1]) * b^(p + 1)),
       a_D_raw = as.numeric((right$beta_q[k1, 2] - left$beta_q[k1, 2]) * b^(p + 1)),
       aY_plus = lead(right, 1), aY_minus = lead(left, 1),
       aD_plus = lead(right, 2), aD_minus = lead(left, 2))
}
