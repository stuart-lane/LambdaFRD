## =============================================================================
## Kernel weights and polynomial design matrices
## =============================================================================

## Kernel weights K((x - c) / h)
kern_weights <- function(xc, h, kernel) {
  if (kernel == "uniform") {
    as.numeric(abs(xc) <= h) * 0.5
  } else if (kernel == "triangular") {
    pmax(0, 1 - abs(xc) / h)
  } else {
    pmax(0, 0.75 * (1 - (xc / h)^2))
  }
}

## Polynomial design matrix in u:  [1, u, u^2, ..., u^order]
build_poly <- function(u, order) {
  mat <- matrix(1, nrow = length(u), ncol = order + 1)
  if (order >= 1L) for (j in 1:order) mat[, j + 1] <- u^j
  mat
}
