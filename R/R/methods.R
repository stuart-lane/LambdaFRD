## =============================================================================
## S3 METHODS
## =============================================================================

#' Print a LambdaFRD fit
#'
#' @param x A \code{"LambdaFRD"} object.
#' @param ... Unused.
#' @export
print.LambdaFRD <- function(x, ...) {
  cat("Call: ")
  print(x$call)
  cat("\n")
  cat(sprintf("%-22s %12d\n", "Number of Obs.", as.integer(x$N)))
  cat(sprintf("%-22s %12s\n", "Kernel", x$kernel))
  cat(sprintf("%-22s %12s\n", "VCE method", toupper(x$vce)))
  cat(sprintf("%-22s %12.4f\n", "lambda", x$lambda))
  cat("\n")
  invisible(x)
}


#' Summarise a LambdaFRD fit
#'
#' @param object A \code{"LambdaFRD"} object.
#' @param ... Unused.
#' @export
summary.LambdaFRD <- function(object, ...) {
  x <- object
  cat("Fuzzy RD estimates using the lambda-class estimator.\n\n")

  cat(sprintf("%-22s %12d\n", "Number of Obs.", as.integer(x$N)))
  cat(sprintf("%-22s %12s\n", "Kernel", x$kernel))
  cat(sprintf("%-22s %12s\n", "VCE method", toupper(x$vce)))
  cat(sprintf("%-22s %12.4f\n", "lambda", x$lambda))
  cat(sprintf("%-22s %12.3f\n", "First-stage F", x$F_stat))
  cat("\n")

  cat(sprintf("%-22s %12s %12s\n", "", "Left", "Right"))
  cat(sprintf("%-22s %12d %12d\n", "Eff. Number of Obs.",
              as.integer(x$N_h[["left"]]), as.integer(x$N_h[["right"]])))
  cat(sprintf("%-22s %12d %12d\n", "Order est. (p)", as.integer(x$p), as.integer(x$p)))
  cat(sprintf("%-22s %12d %12d\n", "Order bias (q)", as.integer(x$q), as.integer(x$q)))
  cat(sprintf("%-22s %12.3f %12.3f\n", "BW est. (h)", x$h, x$h))
  cat(sprintf("%-22s %12.3f %12.3f\n", "BW bias (b)", x$b, x$b))
  cat("\n")

  z  <- (x$coef[, 1] - x$tau_0) / x$se[, 1]
  pv <- 2 * pnorm(-abs(z))
  lvl <- round(100 * (1 - x$alpha))
  line <- strrep("=", 77)
  cat(line, "\n", sep = "")
  cat(sprintf("%14s %10s %10s %10s %10s %18s\n", "Method", "Coef.", "Std. Err.",
              "z", "P>|z|", sprintf("[ %d%% C.I. ]", lvl)))
  cat(line, "\n", sep = "")
  for (i in 1:2) {
    cat(sprintf("%14s %10.3f %10.3f %10.3f %10.3f   [%6.3f , %6.3f]\n",
                rownames(x$coef)[i], x$coef[i, 1], x$se[i, 1], z[i], pv[i],
                x$ci[i, 1], x$ci[i, 2]))
  }
  cat(line, "\n", sep = "")
  invisible(object)
}


#' Extract LambdaFRD coefficients
#'
#' @param object A \code{"LambdaFRD"} object.
#' @param ... Unused.
#' @return Named vector: conventional and bias-corrected estimates.
#' @export
coef.LambdaFRD <- function(object, ...) {
  c(Conventional = object$tau_lambda, Robust = object$tau_BC)
}
