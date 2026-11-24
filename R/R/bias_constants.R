## -----------------------------------------------------------------------------
## Population constant
##
##   c_pop = e_0' Gamma^{-1} vartheta,
##   Gamma_{jk}   = int_0^1 K(u) u^{j+k} du,      j,k = 0..p
##   vartheta_j   = int_0^1 K(u) u^{j+p+1} du
##
## Kernel normalising constants cancel.  Known values (right side, p = 1):
## uniform -1/6, triangular -1/10, Epanechnikov -11/95.
##
## Left side: c_pop_minus = (-1)^{p+1} * c_pop_plus.  For odd p the two
## coincide, so the bias is a constant times the JUMP in the (p+1)-th
## derivative; for even p it is a constant times the SUM.
## -----------------------------------------------------------------------------

pop_bias_const <- function(kernel, p) {
  nu <- function(j) {
    if (kernel == "uniform") {
      1 / (j + 1)
    } else if (kernel == "triangular") {
      1 / (j + 1) - 1 / (j + 2)
    } else {                       # epanechnikov
      1 / (j + 1) - 1 / (j + 3)
    }
  }
  G  <- outer(0:p, 0:p, function(a, b) nu(a + b))
  th <- vapply(0:p, function(a) nu(a + p + 1), numeric(1))
  as.numeric(solve(G, th)[1])
}


## -----------------------------------------------------------------------------
## Empirical constant (one side), on the MAIN bandwidth window
##
##   c_hat = e_0' [ sum_i w_i r_p(u_i) r_p(u_i)' ]^{-1}
##                [ sum_i w_i r_p(u_i) u_i^{p+1} ],     u_i = (X_i - c)/h
##
## Converges to pop_bias_const() but differs in finite samples through the
## local design density.  Covariates are not partialled out.
## -----------------------------------------------------------------------------

emp_bias_const_side <- function(u, wt, p) {
  if (length(u) < p + 2L) return(NA_real_)
  R  <- build_poly(u, p)
  RW <- R * wt
  G  <- crossprod(R, RW)
  th <- crossprod(RW, u^(p + 1))
  tryCatch(as.numeric(solve(G, th)[1]), error = function(e) NA_real_)
}


## -----------------------------------------------------------------------------
## Both sides, with a population fallback when the empirical constant cannot
## be formed on one side.
## -----------------------------------------------------------------------------

bias_constants <- function(sup, h, p, kernel) {
  U_h <- sup$Xc / h
  c_pop     <- pop_bias_const(kernel, p)
  side_sign <- (-1)^(p + 1)
  c_emp_plus  <- emp_bias_const_side(U_h[sup$Z == 1], sup$w[sup$Z == 1], p)
  c_emp_minus <- emp_bias_const_side(U_h[sup$Z == 0], sup$w[sup$Z == 0], p)

  if (is.finite(c_emp_plus) && is.finite(c_emp_minus)) {
    c_plus <- c_emp_plus; c_minus <- c_emp_minus; used <- "empirical"
  } else {
    c_plus <- c_pop; c_minus <- side_sign * c_pop; used <- "population (fallback)"
  }

  list(used = used, plus = c_plus, minus = c_minus, pop = c_pop,
       emp_plus = c_emp_plus, emp_minus = c_emp_minus, side_sign = side_sign)
}
