## =============================================================================
## rdrobust-style same-side nearest-neighbour residuals for all columns
##
## For each i, take the `J` nearest observations in x ON THE SAME SIDE of the
## cutoff (the caller splits by side) and return
##
##   sqrt(J_i / (J_i + 1)) * ( data_i - mean_{j in NN(i)} data_j ),
##
## where J_i >= J is the realised number of matches: all observations tied at
## the J-th nearest distance are included, which matters when x has mass points.
## Returns rows in the ORIGINAL input order.
## =============================================================================

nn_residuals_side <- function(x, data, J) {

  data <- as.matrix(data)
  n <- length(x)
  if (n < 2L) return(matrix(NA_real_, nrow = n, ncol = ncol(data)))

  ord <- order(x, method = "radix")
  xs <- x[ord]; 
  ds <- data[ord, , drop = FALSE]
  tab <- table(xs)
  dups <- rep(as.integer(tab), as.integer(tab))
  dupsid <- unlist(lapply(as.integer(tab), seq_len), use.names = FALSE)
  matches <- min(as.integer(J), n - 1L)
  res <- matrix(0, nrow = n, ncol = ncol(ds))
  nn_tol_eps <- sqrt(.Machine$double.eps)

  for (pos in seq_len(n)) {
    rpos <- dups[pos] - dupsid[pos]
    lpos <- dupsid[pos] - 1L
    while (lpos + rpos < matches) {
      if (pos - lpos - 1L < 1L) {
        rpos <- rpos + dups[pos + rpos + 1L]
      } else if (pos + rpos + 1L > n) {
        lpos <- lpos + dups[pos - lpos - 1L]
      } else {
        dleft <- xs[pos] - xs[pos - lpos - 1L]
        dright <- xs[pos + rpos + 1L] - xs[pos]
        nn_tol <- max(dleft, dright) * nn_tol_eps
        if (dleft - dright > nn_tol) {
          rpos <- rpos + dups[pos + rpos + 1L]
        } else if (dright - dleft > nn_tol) {
          lpos <- lpos + dups[pos - lpos - 1L]
        } else {
          rpos <- rpos + dups[pos + rpos + 1L]
          lpos <- lpos + dups[pos - lpos - 1L]
        }
      }
    }

    lo <- pos - lpos
    hi <- pos + rpos
    Ji <- (hi - lo + 1L) - 1L
    sf <- sqrt(Ji / (Ji + 1))
    neigh_sum <- colSums(ds[lo:hi, , drop = FALSE]) - ds[pos, ]
    res[pos, ] <- sf * (ds[pos, ] - neigh_sum / Ji)
  }

  out <- matrix(0, nrow = n, ncol = ncol(res))
  out[ord, ] <- res
  out
}
