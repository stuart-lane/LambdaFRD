sim_frd <- function(seed = 1, n = 1500) {
  set.seed(seed)
  x <- runif(n, -1, 1)
  d <- as.numeric(rnorm(n) < -0.5 + 1 * (x >= 0))
  y <- x + 2 * x^2 + d + rnorm(n, sd = 0.3)
  list(y = y, x = x, d = d, w = cbind(rnorm(n), runif(n)))
}

test_that("lambda = 1 nests rdrobust's fuzzy RD estimator", {
  skip_if_not_installed("rdrobust")
  s <- sim_frd()
  for (pq in list(c(1, 2), c(1, 3), c(2, 3))) {
    for (v in c("hc0", "hc3", "nn")) {
      r <- rdrobust::rdrobust(s$y, s$x, fuzzy = s$d, p = pq[1], q = pq[2],
                              h = 0.5, b = 0.7, vce = v)
      m <- suppressWarnings(LambdaFRD(s$y, s$x, s$d, p = pq[1], q = pq[2],
                                      h = 0.5, b = 0.7, lambda = 1, vce = v))
      expect_equal(m$tau_lambda, unname(r$coef[1]), tolerance = 1e-10)
      expect_equal(m$tau_BC,     unname(r$coef[2]), tolerance = 1e-10)
      expect_equal(m$se_conv,    unname(r$se[1]),   tolerance = 1e-10)
      expect_equal(m$se_BC,      unname(r$se[3]),   tolerance = 1e-10)
    }
  }
})

test_that("default q is p + 1", {
  s <- sim_frd(2)
  a <- LambdaFRD(s$y, s$x, s$d, h = 0.4, p = 2)
  b <- LambdaFRD(s$y, s$x, s$d, h = 0.4, p = 2, q = 3)
  expect_identical(a$tau_BC, b$tau_BC)
  expect_identical(a$se_BC, b$se_BC)
})

test_that("fields used by the simulation scripts are returned and finite", {
  s <- sim_frd(3)
  f <- LambdaFRD(s$y, s$x, s$d, h = 0.4, trace_target = "bc")
  for (nm in c("lambda", "F_stat", "G_P", "C_P", "kappa_N", "psi_star", "rho_cross",
               "tau_lambda", "ci_lower_conv", "ci_upper_conv",
               "tau_BC", "ci_lower_BC", "ci_upper_BC")) {
    expect_true(is.numeric(f[[nm]]) && length(f[[nm]]) == 1L && is.finite(f[[nm]]), info = nm)
  }
  expect_true(f$lambda > 0 && f$lambda < 1)
  expect_s3_class(f, "LambdaFRD")
  expect_named(coef(f), c("Conventional", "Robust"))
})

test_that("covariates and clustering run", {
  s <- sim_frd(4)
  cl <- sample(1:50, length(s$y), replace = TRUE)
  f <- LambdaFRD(s$y, s$x, s$d, h = 0.4, b = 0.6, covs = s$w,
                 vce = "cluster", cluster = cl)
  expect_true(is.finite(f$se_BC))
  expect_equal(f$n_clusters, 50L)
  expect_output(print(f), "lambda")
  expect_output(summary(f), "Robust")
})

test_that("invalid inputs are rejected", {
  s <- sim_frd(6)
  expect_error(LambdaFRD(s$y, s$x, s$d), "bandwidth")
  expect_error(LambdaFRD(s$y, s$x, s$d, h = 0.4, p = 1, q = 1), "q must be")
  expect_error(LambdaFRD(s$y, s$x, s$d, h = 0.4, vce = "cluster"), "cluster")
  expect_error(LambdaFRD(s$y[-1], s$x, s$d, h = 0.4), "same length")
})

test_that("missing values are dropped", {
  s <- sim_frd(7)
  y <- s$y; y[c(3, 10)] <- NA
  f <- LambdaFRD(y, s$x, s$d, h = 0.4)
  g <- LambdaFRD(s$y[-c(3, 10)], s$x[-c(3, 10)], s$d[-c(3, 10)], h = 0.4)
  expect_identical(f$tau_BC, g$tau_BC)
  expect_equal(f$N, length(y) - 2L)
})
