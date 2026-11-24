## =============================================================================
## SAMPLE SUPPORTS
##
## Three windows are used:
##   main  : observations with K((x - c)/h) > 0  (the lambda-class fit)
##   bias  : observations with K((x - c)/b) > 0  (the order-q bias fit)
##   union : main OR bias.
##
## When b > h the order-q fit uses observations outside the main window.  They
## contribute sampling variability to the bias estimate, so every influence
## quantity is built on the UNION support.  When b <= h the union is the main
## window.
## =============================================================================

lfrd_supports <- function(y, d, x, c, h, b, kernel, covs, cluster) {

  z   <- as.numeric(x >= c)
  xc  <- x - c
  w_h <- kern_weights(xc, h, kernel)
  w_b <- kern_weights(xc, b, kernel)

  in_h <- (w_h > 0)
  in_b <- (w_b > 0)
  in_u <- in_h | in_b
  ids_h <- which(in_h)
  ids_u <- which(in_u)

  list(
    ## ---- full sample ----
    y = y, d = d, x = x, z = z, xc = xc, covs = covs, cluster = cluster,
    w_h = w_h, w_b = w_b, in_h = in_h, in_b = in_b, in_u = in_u,

    ## ---- main window ----
    Y  = y[in_h],  D = d[in_h], Z = z[in_h], Xc = xc[in_h], w = w_h[in_h],
    W  = if (is.null(covs)) NULL else covs[in_h, , drop = FALSE],
    M  = sum(in_h),

    ## ---- bias window ----
    M_b = sum(in_b),

    ## ---- union ----
    Y_u = y[in_u], D_u = d[in_u], Z_u = z[in_u], Xc_u = xc[in_u],
    w_h_u = w_h[in_u], w_b_u = w_b[in_u],
    W_u = if (is.null(covs)) NULL else covs[in_u, , drop = FALSE],
    M_u = length(ids_u),
    idx_main_in_union = match(ids_h, ids_u),
    ids_u = ids_u
  )
}
