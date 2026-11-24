## =============================================================================
## VARIANCE ESTIMATORS
##
##   hc0     sum xi_i^2                          (no leverage correction)
##   hc1     finite-sample rescaling of xi_i
##   hc2     sum (xi_i / sqrt(1 - h_ii))^2
##   hc3     sum (xi_i / (1 - h_ii))^2
##   nn      nearest-neighbour residuals in place of fitted residuals (as in rdrobust)
##   cluster sum_g (sum_{i in g} xi_i)^2, with the CR1 finite-sample factor
##   homo    constant (co)variances from pooled residuals
##
## Each is computed for the bias-corrected (robust) and the conventional score.
## Only the selected `vce` is computed unless report_all_vce = TRUE (hc0-hc3 are
## always cheap and always returned).
## =============================================================================

lfrd_variance <- function(sup, dsg, lp, inf, p, q, vce, nnmatch, nn_full_sample,
                          hc_lev, vce_residuals, report_all_vce) {

  M  <- sup$M
  df <- dsg$df 
  w <- sup$w
  Z_u <- sup$Z_u
  idx_m <- sup$idx_main_in_union
  
  dZ <- lp$dZ
  cY <- inf$cY
  cD <- inf$cD
  cW <- inf$cW
  
  xi_var  <- if (vce_residuals == "q") inf$xi_var_q else inf$xi_var_p
  xi_conv <- inf$xi_conv

  ## ---- leverages ----
  hlev_main <- if (hc_lev == "full") dsg$H_diag + dsg$P_ii else dsg$H_diag
  hlev_main <- pmin(hlev_main, 0.99)
  hlev_ii <- numeric(sup$M_u)
  hlev_ii[idx_m] <- hlev_main
  if (vce_residuals == "q") hlev_ii <- inf$hlev_q

  ## ---- HC family (bias-corrected) ----
  var_hc0 <- sum(xi_var^2)
  if (vce_residuals == "q") {
    n_above <- sum((Z_u == 1) & (sup$w_b_u > 0))
    n_below <- sum((Z_u == 0) & (sup$w_b_u > 0))
    f_hc1 <- ifelse(
      Z_u == 1,
      sqrt(n_above / max(n_above - (q + 1), 1)),
      sqrt(n_below / max(n_below - (q + 1), 1))
    )
  } else {
    f_hc1 <- rep(sqrt(M / df), sup$M_u)
  }
  var_hc1 <- sum((f_hc1 * xi_var)^2)
  var_hc2 <- sum((xi_var / sqrt(1 - hlev_ii))^2)
  var_hc3 <- sum((xi_var / (1 - hlev_ii))^2)

  ## ---- nearest neighbour ----
  var_nn <- var_conv_nn <- NA_real_
  if (vce == "nn" || report_all_vce) {
    if (nn_full_sample && !inf$rdrobust_nesting) {
      nn_data_full <- if (dZ > 0L) cbind(sup$y, sup$d, sup$covs) else cbind(sup$y, sup$d)
      nn_res_full <- matrix(0, nrow = nrow(nn_data_full), ncol = ncol(nn_data_full))
      
      for (side in c(0, 1)) {
        ii <- which(sup$z == side)
        if (length(ii) >= 2L)
          nn_res_full[ii, ] <- nn_residuals_side(sup$x[ii], nn_data_full[ii, , drop = FALSE], nnmatch)
      }
      nn_res_u <- nn_res_full[sup$ids_u, , drop = FALSE]
    } else {
      nn_data_u <- lp$Dmat_u
      nn_res_u <- matrix(0, nrow = nrow(nn_data_u), ncol = ncol(nn_data_u))
      
      for (side in c(0, 1)) {
        ii <- which(Z_u == side)
        if (length(ii) >= 2L)
          nn_res_u[ii, ] <- nn_residuals_side(sup$Xc_u[ii], nn_data_u[ii, , drop = FALSE], nnmatch)
      }
    }

    if (inf$rdrobust_nesting) {
      nn_score_u  <- as.numeric(nn_res_u %*% inf$s_rd)
      var_conv_nn <- sum((inf$ell_p_u * nn_score_u)^2)
      var_nn      <- sum((inf$ell_bc_u * nn_score_u)^2)
    } else {
      nn_Y <- nn_res_u[, 1]; nn_D <- nn_res_u[, 2]
      nn_score_bc <- cY * nn_Y + cD * nn_D
      if (dZ > 0L)
        nn_score_bc <- nn_score_bc + rowSums(cW * nn_res_u[, 3:ncol(nn_res_u), drop = FALSE])
      var_nn <- sum(nn_score_bc^2)

      nn_score_conv <- inf$cY_conv * nn_Y[idx_m] + inf$cD_conv * nn_D[idx_m]
      if (dZ > 0L)
        nn_score_conv <- nn_score_conv +
          rowSums(inf$cW_conv * nn_res_u[idx_m, 3:ncol(nn_res_u), drop = FALSE])
      var_conv_nn <- sum(nn_score_conv^2)
    }
  }

  ## ---- homoskedastic ----
  var_homo <- var_conv_homo <- NA_real_
  if (vce == "homo" || report_all_vce) {
    eY <- inf$eY_p; eD <- inf$eD_p
    s_YY <- sum(eY^2 * w) / df
    s_DD <- sum(eD^2 * w) / df
    s_YD <- sum(eY * eD * w) / df
    var_homo <- s_YY * sum(cY^2) + 2 * s_YD * sum(cY * cD) + s_DD * sum(cD^2)
    var_conv_homo <- s_YY * sum(inf$cY_conv^2) +
      2 * s_YD * sum(inf$cY_conv * inf$cD_conv) + s_DD * sum(inf$cD_conv^2)
  }

  ## ---- cluster (CR1) ----
  var_cl <- var_conv_cl <- NA_real_; n_clusters <- NA_integer_
  if ((vce == "cluster" || report_all_vce) && !is.null(sup$cluster)) {
    g_sums <- tapply(xi_var, sup$cluster[sup$in_u], sum)
    G <- length(g_sums)
    n_clusters <- G
    var_cl <- (G / max(G - 1, 1)) * ((M - 1) / df) * sum(g_sums^2)

    g_sums_m <- tapply(xi_conv, sup$cluster[sup$in_h], sum)
    Gm <- length(g_sums_m)
    var_conv_cl <- (Gm / max(Gm - 1, 1)) * ((M - 1) / df) * sum(g_sums_m^2)
  }

  ## ---- conventional HC family, on the order-p residual score ----
  hlev_p <- inf$hlev_p
  n_above_p <- sum(sup$Z == 1); n_below_p <- sum(sup$Z == 0)
  f_conv_hc1 <- ifelse(
    sup$Z == 1,
    sqrt(n_above_p / max(n_above_p - (p + 1), 1)),
    sqrt(n_below_p / max(n_below_p - (p + 1), 1))
  )
  var_conv_hc0 <- sum(xi_conv^2)
  var_conv_hc1 <- sum((f_conv_hc1 * xi_conv)^2)
  var_conv_hc2 <- sum((xi_conv / sqrt(1 - hlev_p))^2)
  var_conv_hc3 <- sum((xi_conv / (1 - hlev_p))^2)

  var_all <- c(hc0 = var_hc0, hc1 = var_hc1, hc2 = var_hc2, hc3 = var_hc3,
               nn = var_nn, cluster = var_cl, homo = var_homo)
  var_conv_all <- c(hc0 = var_conv_hc0, hc1 = var_conv_hc1, hc2 = var_conv_hc2,
                    hc3 = var_conv_hc3, nn = var_conv_nn, cluster = var_conv_cl,
                    homo = var_conv_homo)

  var_BC <- var_all[[vce]]
  if (!is.finite(var_BC) || var_BC <= 0)
    stop("Selected variance estimator (vce = '", vce, "') is not positive.", call. = FALSE)
  var_conv <- var_conv_all[[vce]]
  if (!is.finite(var_conv) || var_conv <= 0)
    stop("Selected conventional variance estimator (vce = '", vce, "') is not positive.",
         call. = FALSE)

  list(var_BC = var_BC, var_conv = var_conv,
       var_all = var_all, var_conv_all = var_conv_all,
       n_clusters = n_clusters, xi_var = xi_var)
}
