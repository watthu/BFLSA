library(Rcpp)
library(coda)
library(Matrix)
library(igraph)
library(mclust)
library(RcppArmadillo)
Rcpp::sourceCpp("MAP_subgroup.cpp", env = environment())
Rcpp::sourceCpp("Gibbs_subgroup.cpp", env = environment())

#' Pointwise log-likelihood of the Gaussian regression model
#'
#' @param y A length-n numeric vector, response variable.
#' @param X An n by np numeric matrix, block-diagonal design of the heterogeneous covariates (row i
#'   contains \code{X_[i, ]} in columns \code{(i - 1) * p + 1:p}).
#' @param Z An n by q numeric matrix, covariates with common effects.
#' @param beta A length-np numeric vector, individual coefficients stacked by individual,
#'   \eqn{(\beta_1^T, \ldots, \beta_n^T)^T}.
#' @param alpha A length-q numeric vector, common coefficients \eqn{\alpha}.
#' @param sigma2 A positive numeric scalar, the error variance \eqn{\sigma^2}.
#'
#' @return \code{logf} returns an n by 1 numeric matrix, the values of
#'   \eqn{\log N(y_i; x_i^T \beta_i + z_i^T \alpha, \sigma^2)}, used to compute the lppd.
logf <- function(y, X, Z, beta, alpha, sigma2) {
  dnorm(y, X %*% beta + Z %*% alpha, sqrt(sigma2), log = T)
}

#' Initial values of individual coefficients
#'
#' Initialize individual-level regression coefficients using OLS residuals (heterogeneous intercept
#' only, \eqn{p = 1}) or a projected ridge fit followed by a local regression refinement
#' (\eqn{p > 1}).
#'
#' @param y A length-n numeric vector, response variable.
#' @param X An n by np numeric matrix, block-diagonal design built from \code{X_}.
#' @param X_ An n by p numeric matrix, covariates with heterogeneous effects.
#' @param Z An n by q numeric matrix, covariates with common effects.
#'
#' @return \code{get_init_mbeta} returns an n by p numeric matrix whose i-th row is the initial
#'   value of \eqn{\beta_i}.
get_init_mbeta <- function(y, X, X_, Z) {
  n <- dim(X_)[1]; p <- dim(X_)[2]; q <- dim(Z)[2]
  X_Z <- cbind(X_, Z)
  
  # Estimate combined coefficients via OLS
  betaalpha <- solve(t(X_Z) %*% X_Z) %*% t(X_Z) %*% y
  alphaR <- betaalpha[1:q + p]
  newy <- y - Z %*% alphaR
  
  if (p == 1) {
    mbeta0 <- newy
  } else {
    index <- t(combn(n, 2))
    Delta <- matrix(0, choose(n, 2), n)
    for (l in 1:choose(n, 2)) {
      i <- index[l, 1]; j <- index[l, 2]
      Delta[l, i] <- 1; Delta[l, j] <- -1
    }
    AA <- kronecker(t(Delta) %*% Delta, diag(p))

    # Use a smooth initialization strategy inspired by Ma et al. (2020)
    Pz <- diag(n) - Z %*% solve(t(Z) %*% Z) %*% t(Z)
    betaR <- solve(t(X) %*% Pz %*% X  + 0.0001 * AA) %*% t(X) %*% Pz %*% y
    mbetaR <- matrix(betaR, nrow = n, byrow = T)
    mbetaR_med <- apply(mbetaR, 1, median)
    dist_med <- as.matrix(dist(mbetaR_med))

    # Localized regression refinement using its ten nearest neighbors
    mbeta0 <- t(sapply(1:n, function(i) {
      subID <- order(dist_med[i, ])[1:max(p, 10)]
      Xsub <- X_[subID, ]; ysub <- newy[subID]
      solve(t(Xsub) %*% Xsub + 0.0001 * diag(p)) %*% t(Xsub) %*% ysub
    }))
  }
  return(mbeta0)
}

#' Neighborhood graph and effective resistance
#'
#' Build a neighborhood (knn + MST) graph based on the Manhattan distances between the initial
#' coefficient vectors, and compute the pairwise effective resistance of the graph. The MST edges
#' guarantee that the graph is connected. If a prior network \code{A_prior} is given, neighbors are
#' only chosen among its edges, and disconnected components of the prior network are bridged by
#' their closest pairs.
#'
#' @param mbeta An n by p numeric matrix, initial values of \eqn{\beta_i} (e.g., from
#'   \code{get_init_mbeta}).
#' @param n,p Integer scalars, the sample size and the number of heterogeneous covariates.
#' @param kn An integer scalar, the maximum number of nearest neighbors of each individual.
#'   Individuals with higher eigenvector centrality receive fewer neighbors (down to
#'   \code{min(kn, 10)}).
#' @param r_coef A positive numeric scalar, the multiplier controlling the separation between spike
#'   and slab through \code{log_scale_factor}.
#' @param A_prior An optional symmetric n by n numeric matrix, the adjacency matrix of a prior
#'   network; positive entries are candidate edges. Default is \code{NULL} (no prior network).
#'
#' @return \code{get_knn_features} returns a named list containing the following components. All
#'   pairwise quantities follow the order of \code{combn(n, 2)}, i.e., the lower triangle of an n by
#'   n matrix.
#'
#' \tabular{ll}{
#'   \code{r_edge_all} \tab A length-\code{choose(n, 2)} numeric vector, effective resistances of
#'                  all pairs. \cr
#'   \code{r_edge} \tab A numeric vector, effective resistances of the neighborhood edges. \cr
#'   \code{log_scale_factor} \tab A numeric scalar, \code{r_coef * log(n) / min(r_edge) / p}, the
#'                  log ratio of slab to spike scales. \cr
#'   \code{knn_zeta} \tab A length-\code{choose(n, 2)} logical vector indicating whether each pair
#'                  is a neighborhood edge. \cr
#'   \code{mst_zeta} \tab An integer vector, positions (in all pairs) of the MST edges. \cr
#' }
get_knn_features <- function(mbeta, n, p, kn, r_coef, A_prior = NULL) {
  # Construct minimum spanning tree (MST) connections based on pairwise distances of beta
  dist <- as.matrix(dist(mbeta, method = "manhattan"))
  dist[dist < 1e-6] <- 1e-6; diag(dist) <- 0
  base_mat <- matrix(0, n, n)
  if (!is.null(A_prior)) {
    base_mat <- (A_prior > 0) * 1; diag(base_mat) <- 0
    prior_graph <- graph_from_adjacency_matrix(base_mat, mode = "undirected")
    prior_components <- components(prior_graph)
    prior_w_graph <- graph_from_adjacency_matrix(base_mat * dist, mode = "undirected", weighted = T)
    eigenvector_centrality <- eigen_centrality(prior_w_graph)$vector
    
    if (prior_components$no == 1) {
      zeta_mat <- as.matrix(as_adjacency_matrix(mst(prior_w_graph), attr = "weight") > 0)
    } else {
      comp_label <- prior_components$membership
      n_comp <- prior_components$no
      zeta_mat <- matrix(0, n, n)
      for (cc in 1:n_comp) {
        idx <- which(comp_label == cc)
        sub_dist <- dist[idx, idx, drop = FALSE]
        sub_graph <- graph_from_adjacency_matrix(sub_dist, mode = "undirected", weighted = T)
        sub_mst <- mst(sub_graph)
        zeta_mat[idx, idx] <- as.matrix(as_adjacency_matrix(sub_mst, attr = "weight") > 0)
      }
      comp_dist <- matrix(Inf, n_comp, n_comp)
      comp_bridge <- vector("list", n_comp * n_comp)
      for (i in 1:(n - 1)) {
        for (j in (i + 1):n) {
          ci <- comp_label[i]; cj <- comp_label[j]
          if (ci != cj && dist[i,j] < comp_dist[ci, cj]) {
            comp_dist[ci, cj] <- comp_dist[cj, ci] <- dist[i, j]
            comp_bridge[[ci + (cj - 1) * n_comp]] <- c(i, j)
            comp_bridge[[cj + (ci - 1) * n_comp]] <- c(i, j)
          }
        }
      }
      diag(comp_dist) <- 0
      comp_mst_edges <- as_edgelist(mst(
        graph_from_adjacency_matrix(comp_dist, mode = "undirected", weighted = T)
      ))
      for (e in 1:nrow(comp_mst_edges)) {
        ci <- comp_mst_edges[e, 1]; cj <- comp_mst_edges[e, 2]
        bn <- comp_bridge[[ci + (cj - 1) * n_comp]]
        zeta_mat[bn[1], bn[2]] <- zeta_mat[bn[2], bn[1]] <- 1
      }
      zeta_mat <- zeta_mat > 0
    }
  } else {
    graph <- graph_from_adjacency_matrix(dist, mode = "undirected", weighted = T)
    mst_g <- mst(graph)
    zeta_mat <- as.matrix(as_adjacency_matrix(mst_g, attr = "weight") > 0)
    eigenvector_centrality <- eigen_centrality(graph)$vector
  }
  logi <- lower.tri(zeta_mat)
  mst_zeta <- which(zeta_mat[logi])  # Must link in MST
  
  # The existing schedule gives fewer neighbors to higher-centrality nodes.
  kns <- numeric(n)
  kns[order(eigenvector_centrality)] <- floor(seq(from = kn, to = min(kn, 10), length.out = n))
  knn_mat <- sapply(1:n, function(i) {
    neighbor_vector <- numeric(n)
    if (!is.null(A_prior)) {
      candidates <- which(base_mat[i, ] == 1)
      k <- min(kns[i], length(candidates))
      nearest <- candidates[order(dist[i, candidates])[seq_len(k)]]
    } else {
      candidates <- setdiff(seq_len(n), i)
      nearest <- candidates[order(dist[i, candidates])[seq_len(min(kns[i], n - 1L))]]
    }
    neighbor_vector[nearest] <- 1
    return(neighbor_vector)
  })
  knn_mat <- pmin(knn_mat + t(knn_mat) + zeta_mat, 1); diag(knn_mat) <- 0
  
  # Calculating pairwise effective resistance
  L <- -knn_mat; diag(L) <- -rowSums(L); L_inv <- MASS::ginv(L)
  r_mat <- rep(1, n) %*% t(diag(L_inv)) + diag(L_inv) %*% t(rep(1, n)) - 2 * L_inv
  logi <- lower.tri(r_mat)
  knn_zeta <- knn_mat[logi] == 1
  r_edge_all <- r_mat[logi]
  r_edge <- r_edge_all[knn_zeta]
  log_scale_factor <- r_coef * log(n) / min(r_edge) / p
  return(list(r_edge_all = r_edge_all, r_edge = r_edge, log_scale_factor = log_scale_factor,
              knn_zeta = knn_zeta, mst_zeta = mst_zeta))
}

#' Indices of selected pairs
#'
#' @param n An integer scalar, the sample size.
#' @param knn_zeta A length-\code{choose(n, 2)} logical vector indicating the selected pairs in the
#'   order of \code{combn(n, 2)}.
#'
#' @return \code{get_knn_pairs} returns a named list with integer vectors \code{i} and \code{j}, the
#'   0-based indices (for C++) of the two individuals of each selected pair, where \code{i < j}.
get_knn_pairs <- function(n, knn_zeta) {
  idx <- which(knn_zeta)
  comb <- t(combn(n, 2))
  list(i = comb[idx, 1] - 1, j = comb[idx, 2] - 1)
}

#' Subgroup detection from the estimated edge indicators
#'
#' Select fused edges by thresholding the posterior means of \eqn{\delta_{ij}} with an FDR-type
#' rule, find the connected components of the fused graph, merge components with (numerically)
#' identical centers, and relabel subgroups by the first coordinate of their centers.
#'
#' @param beta A length-np numeric vector, the estimate of \eqn{\beta}.
#' @param delta A numeric vector, posterior means of \eqn{\delta_{ij}} on the neighborhood edges
#'   (small values indicate fusion).
#' @param sigma2,lambda0 Numeric scalars, the estimates of \eqn{\sigma^2} and \eqn{\lambda_0}.
#' @param n An integer scalar, the sample size.
#' @param r_coef A positive numeric scalar, see \code{get_knn_features}.
#' @param knn_features A list returned by \code{get_knn_features}.
#'
#' @return \code{subgroup_res} returns a named list containing the following components:
#'
#' \tabular{ll}{
#'   \code{Khat} \tab An integer scalar, the estimated number of subgroups. \cr
#'   \code{label_hat} \tab A length-n integer vector, the estimated subgroup labels. \cr
#'   \code{eta_hat} \tab A length-(p * \code{Khat}) numeric vector, the subgroup centers stacked by
#'                  subgroup. \cr
#'   \code{flag} \tab A logical scalar indicating whether some connected components were merged. \cr
#' }
subgroup_res <- function(beta, delta, sigma2, n, r_coef, lambda0, knn_features) {
  # Compute False Discovery Rate (FDR) control threshold for edge inclusion
  threds <- seq(0, 0.5, 0.001)
  FDR_control <- 1 / (1 + exp(r_coef * log(n) / 2 + sqrt(lambda0) * 
                                expm1(-knn_features$log_scale_factor / 2) * 1e-2 / sqrt(sigma2)))
  FDR <- sapply(threds, function(nu) sum(delta * (delta <= nu)) / sum(delta <= nu))
  eligible <- threds[is.finite(FDR) & FDR <= FDR_control]
  threshold <- if (length(eligible)) max(eligible) else -Inf
  delta <- as.numeric(delta <= threshold)
  knn_pairs <- get_knn_pairs(n, knn_features$knn_zeta)
  mdelta <- as.matrix(sparseMatrix(i = c(knn_pairs$i + 1, knn_pairs$j + 1),
                                   j = c(knn_pairs$j + 1, knn_pairs$i + 1),
                                   x = c(delta, delta), dims = c(n, n)))
  
  # Find connected components (subgroups) in the graph
  graph <- graph_from_adjacency_matrix(mdelta, mode = "undirected")
  components <- components(graph)
  K <- components$no
  label <- components$membership
  
  # Compute subgroup centroids by averaging beta coefficients
  p <- as.integer(length(beta) / n)
  mbeta <- matrix(beta, nrow = n, byrow = T)
  meta <- matrix(0, ncol(mbeta), K)
  for (i in 1:K) {
    if (sum(label == i) == 1) {
      meta[, i] <- mbeta[label == i,]
    } else {
      if (p == 1) {
        meta[, i] <- mean(mbeta[label == i,])
      } else {
        meta[, i] <- colMeans(mbeta[label == i,])
      }
    }
  }
  # Construct a new graph with edges between clusters and combine close clusters
  adj_label <- as.matrix(dist(t(meta))) < 1e-4
  graph_label <- graph_from_adjacency_matrix(adj_label, mode = "undirected")
  components_label <- components(graph_label)
  K_final <- components_label$no
  label_label <- components_label$membership
  label_final <- unname(label_label[label])
  meta_final <- matrix(0, ncol(mbeta), K_final)
  for (i in 1:K_final) {
    if (sum(label_final == i) == 1) {
      meta_final[, i] <- mbeta[label_final == i,]
    } else {
      if (p == 1) {
        meta_final[, i] <- mean(mbeta[label_final == i,])
      } else {
        meta_final[, i] <- colMeans(mbeta[label_final == i,])
      }
    }
  }
  eta_final <- as.vector(meta_final)
  
  # Reorder clusters and reassign final labels
  meta <- matrix(eta_final, ncol = K_final)
  order_of_centers <- order(meta[1, ])
  label <- label_final
  for (i in 1:K_final) {
    label[label_final == order_of_centers[i]] <- i
  }
  meta <- meta[, order_of_centers, drop = F]
  eta <- as.vector(meta)
  
  return(list(Khat = K_final, label_hat = label, eta_hat = eta, flag = K_final != K))
}

#' MAP estimation for subgroup analysis
#'
#' An R wrapper of \code{MAP_full_cpp}. It performs Maximum A Posteriori (MAP) updates with a
#' decaying \eqn{\lambda_0} restart schedule, which provides warm starts for the Gibbs sampler.
#'
#' @param y A length-n numeric vector, response variable.
#' @param X An n by np numeric matrix, block-diagonal design built from \code{X_}.
#' @param Z An n by q numeric matrix, covariates with common effects.
#' @param beta0,alpha0 Numeric vectors, the initial values of \eqn{\beta} and \eqn{\alpha},
#'   respectively.
#' @param sigma20,lambda00 Positive numeric scalars, the initial values of \eqn{\sigma^2} and
#'   \eqn{\lambda_0}.
#' @param knn_features A list returned by \code{get_knn_features}.
#' @param model A character string, \code{"SS"} (spike-and-slab) or \code{"SSL"} (spike-and-slab
#'   lasso).
#' @param a_lambda A positive numeric scalar, the rate parameter of the Gamma prior of
#'   \eqn{\lambda_0}.
#' @param threshold A positive numeric scalar, the stop threshold.
#' @param maxiter An integer scalar, the maximum number of iterations of both inner and outer loops.
#'
#' @return \code{MAP_subgroup} returns a named list mainly containing the following components:
#'
#' \tabular{ll}{
#'   \code{beta},\code{alpha} \tab Numeric vectors, the MAP estimates of \eqn{\beta} and
#'                  \eqn{\alpha}. \cr
#'   \code{sigma2},\code{lambda0} \tab Numeric scalars, the MAP estimates of \eqn{\sigma^2} and
#'                  \eqn{\lambda_0}. \cr
#'   \code{prob_delta} \tab A numeric vector, conditional probabilities of \eqn{\delta_{ij} = 1} on
#'                  the neighborhood edges. \cr
#'   \code{pesudo_log_post} \tab A numeric scalar, the best pseudo log-posterior value. \cr
#'   \code{iteration},\code{pesudo_log_posts} \tab Numeric vectors, the number of inner iterations
#'                  of each outer round and the history of accepted pseudo log-posterior values. \cr
#' }
MAP_subgroup <- function(y, X, Z, beta0, alpha0, sigma20, lambda00, knn_features, model = "SS",
                         a_lambda, threshold = 1e-4, maxiter = 1e2) {
  n <- nrow(X)
  r_edge <- knn_features$r_edge
  log_scale_factor <- knn_features$log_scale_factor
  knn_zeta <- knn_features$knn_zeta
  mst_zeta <- knn_features$mst_zeta
  knn_pairs <- get_knn_pairs(n, knn_zeta)
  mst_idx_in_knn <- as.integer(match(mst_zeta, which(knn_zeta)) - 1L)
  
  result <- MAP_full_cpp(y, X, Z, beta0, alpha0, sigma20, lambda00, r_edge, log_scale_factor,
                         ei = knn_pairs$i, ej = knn_pairs$j, mst_idx_in_knn, model,
                         a_lambda, threshold, maxiter = as.integer(maxiter))
  return(result)
}

#' Gibbs sampler function for subgroup analysis
#'
#' An R wrapper of \code{Gibbs_sampler_cpp}. It draws posterior samples of \eqn{\beta},
#' \eqn{\alpha}, \eqn{\sigma^2}, \eqn{\lambda_0} and the edge indicators \eqn{\delta_{ij}}.
#'
#' @inheritParams MAP_subgroup
#' @param nburn,niter Integer scalars indicate the numbers of burn-in samples and posterior samples,
#'   respectively.
#' @param return.burn A logical scalar indicates whether to return the burn-in samples.
#' @param edge_mode A character string. \code{"all"} samples \eqn{\delta_{ij}} for all pairs, while
#'   \code{"nbd"} samples them only for the neighborhood edges. In both cases, only neighborhood
#'   edges enter the update of \eqn{\beta}.
#'
#' @return \code{Gibbs_subgroup} returns a named list mainly containing the following components:
#'
#' \tabular{ll}{
#'   \code{beta} \tab A \code{niter} by np numeric matrix, posterior samples of \eqn{\beta}. \cr
#'   \code{alpha} \tab A \code{niter} by q numeric matrix, posterior samples of \eqn{\alpha}. \cr
#'   \code{delta} \tab A \code{niter} by (number of sampled pairs) numeric matrix, posterior samples
#'                  of \eqn{\delta_{ij}}. \cr
#'   \code{sigma2},\code{lambda0} \tab length-\code{niter} numeric vectors, posterior samples of
#'                  \eqn{\sigma^2} and \eqn{\lambda_0}. \cr
#'   \code{post} \tab A length-\code{niter} numeric vector, log-posterior values (up to a constant)
#'                  of the samples. \cr
#'   \code{delta_hat} \tab A numeric vector, posterior means of \eqn{\delta_{ij}}. \cr
#'   \code{*_burn} \tab Burn-in samples, returned only if \code{return.burn = TRUE}. \cr
#' }
Gibbs_subgroup <- function(y, X, Z, beta0, alpha0, sigma20, lambda00, knn_features, model = "SS",
                           a_lambda, nburn = 2000, niter = 5000, return.burn = FALSE,
                           edge_mode = "all") {
  n <- nrow(X); m <- choose(n, 2)
  r_edge <- knn_features$r_edge
  r_edge_all <- knn_features$r_edge_all
  log_scale_factor <- knn_features$log_scale_factor
  knn_zeta <- knn_features$knn_zeta
  mst_zeta <- knn_features$mst_zeta
  knn_pairs <- get_knn_pairs(n, knn_zeta)
  all_pairs <- get_knn_pairs(n, rep(TRUE, m))
  mst_idx_in_knn <- as.integer(match(mst_zeta, which(knn_zeta)) - 1L)
  knn_idx_in_all <- as.integer(which(knn_zeta) - 1L)
  
  result <- Gibbs_sampler_cpp(y, X, Z, beta0, alpha0, sigma20, lambda00, log_scale_factor,
                              r_edge, ei = knn_pairs$i, ej = knn_pairs$j, mst_idx_in_knn,
                              r_edge_all, ei_all = all_pairs$i, ej_all = all_pairs$j,
                              knn_idx_in_all, model = model, a_lambda = a_lambda,
                              nburn = as.integer(nburn), niter = as.integer(niter),
                              return_burn = return.burn, edge_mode = edge_mode)
  return(result)
}

#' BFLSA algorithm
#'
#' Bayesian Fusion Learning for Subgroup Analysis. It initializes the model parameters, runs MAP
#' estimation from several initial values of \eqn{\sigma^2}, and then uses Gibbs sampling started at
#' the best MAP estimate to obtain the posterior inference and the subgroups. If the true values are
#' given, it also evaluates the performance through EstErr and ARI. The model is
#' \deqn{y_i = x_i^T \beta_i + z_i^T \alpha + \epsilon_i, \epsilon_i \sim N(0, \sigma^2).}
#'
#' @param y A length-n numeric vector, response variable.
#' @param X_ An n by p numeric matrix, covariates with heterogeneous effects \eqn{\beta_i} (e.g., a
#'   column of ones for heterogeneous intercepts).
#' @param Z An n by q numeric matrix, covariates with common effects \eqn{\alpha}.
#'   \code{cbind(X_, Z)} should have full column rank.
#' @param beta_true,alpha_true Optional numeric vectors, the true \eqn{\beta} (length np, stacked by
#'   individual) and \eqn{\alpha} to obtain EstErr.
#' @param label_true An optional length-n vector, the true subgroup labels to obtain ARI.
#' @param nburn,niter Integer scalars indicate the numbers of burn-in samples and posterior samples,
#'   respectively. \code{niter} should be at least 20.
#' @param a_lambda A positive numeric scalar, the rate parameter of the Gamma prior of
#'   \eqn{\lambda_0}, divided internally by the number of neighborhood edges.
#' @param lambda0_init A positive numeric scalar, the initial value of \eqn{\lambda_0}.
#' @param kn An integer scalar in \code{[1, n - 1]}, the maximum number of nearest neighbors, see
#'   \code{get_knn_features}.
#' @param r_coef A positive numeric scalar, the multiplier controlling the separation between spike
#'   and slab.
#' @param model A character string, \code{"SS"} (spike-and-slab) or \code{"SSL"} (spike-and-slab
#'   lasso).
#' @param return.chain A logical scalar indicates whether to return the posterior samples and all
#'   MAP estimates.
#' @param A_prior An optional symmetric n by n numeric matrix, the adjacency matrix of a prior
#'   network. Default is \code{NULL}.
#' @param mc.cores An integer scalar, the number of cores used for the MAP estimates (passed to
#'   \code{parallel::mclapply}; use 1 on Windows).
#' @param edge_mode A character string, \code{"all"} or \code{"nbd"}, see \code{Gibbs_subgroup}.
#'
#' @return \code{BFLSA} returns a named list mainly containing the following components:
#'
#' \tabular{ll}{
#'   \code{beta_hat},\code{alpha_hat} \tab Numeric vectors, posterior medians of \eqn{\beta} and
#'                  \eqn{\alpha}. \cr
#'   \code{sigma2_hat},\code{lambda0_hat} \tab Numeric scalars, posterior medians of \eqn{\sigma^2}
#'                  and \eqn{\lambda_0}. \cr
#'   \code{delta_hat} \tab A numeric vector, posterior means of \eqn{\delta_{ij}} (all pairs if
#'                  \code{edge_mode = "all"}, neighborhood edges if \code{edge_mode = "nbd"}). \cr
#'   \code{post_hat} \tab A numeric scalar, posterior median of the log-posterior. \cr
#'   \code{Khat},\code{label_hat},\code{eta_hat} \tab The estimated number of subgroups, subgroup
#'                  labels and subgroup centers, see \code{subgroup_res}. \cr
#'   \code{EstErr_beta},\code{EstErr_alpha},\code{ARI_hat} \tab Numeric scalars, estimation errors
#'                  of \eqn{\beta} and \eqn{\alpha}, and the adjusted Rand index (\code{NA} if the
#'                  truth is not given). \cr
#'   \code{check_converge} \tab A list of the Geweke, Heidelberger-Welch and autocorrelation
#'                  diagnostics of \eqn{\beta}, and an integer \code{converged} (1 if all checks
#'                  pass). \cr
#'   \code{MAP_estimate} \tab A list, the best MAP estimate, see \code{MAP_subgroup}. \cr
#'   \code{lppd},\code{IC} \tab Numeric scalars, the log pointwise predictive density and the
#'                  information criterion. \cr
#'   \code{time} \tab A length-3 numeric vector, the run time (in seconds) of MAP, MCMC and the
#'                  whole procedure. \cr
#'   \code{chain},\code{MAP_estimates} \tab The posterior samples and all MAP estimates, returned
#'                  only if \code{return.chain = TRUE}. \cr
#' }
BFLSA <- function(y, X_, Z, beta_true = NULL, alpha_true = NULL, label_true = NULL, nburn, niter,
                  a_lambda, lambda0_init, kn, r_coef, model = "SS", return.chain = FALSE,
                  A_prior = NULL, mc.cores = 1, edge_mode = "all") {
  stopifnot(is.matrix(X_), is.matrix(Z), is.numeric(y),
            nrow(X_) == length(y), nrow(Z) == length(y),
            nrow(X_) >= 3, ncol(X_) >= 1, ncol(Z) >= 1,
            all(is.finite(y)), all(is.finite(X_)), all(is.finite(Z)),
            length(nburn) == 1, nburn >= 0, nburn == as.integer(nburn),
            length(niter) == 1, niter >= 20, niter == as.integer(niter),
            length(kn) == 1, kn >= 1, kn < length(y), kn == as.integer(kn),
            length(a_lambda) == 1, is.finite(a_lambda), a_lambda > 0,
            length(lambda0_init) == 1, is.finite(lambda0_init), lambda0_init > 0,
            length(r_coef) == 1, is.finite(r_coef), r_coef > 0)
  if (qr(cbind(X_, Z))$rank < ncol(X_) + ncol(Z))
    stop("The combined design cbind(X_, Z) must have full column rank.")
  
  t_start <- proc.time()
  n <- dim(X_)[1]; p <- dim(X_)[2]; q <- dim(Z)[2]
  Cn <- ifelse(p == 1, 12.5 * log(log(n * p + q)), log(n * p + q))
  
  # Create the block-diagonal matrix X from X_
  X <- as.matrix(bdiag(apply(X_, 1, function(x) Matrix(x, nrow = 1))))
  if (model != "SS" & model != "SSL") {
    stop("Incorrect model type! Please choose either 'SS' or 'SSL'.")
  }
  if (edge_mode != "all" & edge_mode != "nbd") {
    stop("Incorrect edge mode! Please choose either 'all' or 'nbd'.")
  }
  
  # Get initial values for beta, alpha, sigma2, and distances
  mbeta0 <- get_init_mbeta(y, X, X_, Z)
  beta0 <- as.vector(t(mbeta0))
  alpha0 <- solve(t(Z) %*% Z) %*% t(Z) %*% (y - X %*% beta0)
  dist0 <- as.matrix(dist(mbeta0, method = "euclidean"))
  sigma20s <- max(mean(dist0^2) / n, mean((y - X %*% beta0 - Z %*% alpha0)^2)) * 2^(-2:7)
  knn_features <- get_knn_features(mbeta0, n, p, kn, r_coef, A_prior)
  a_lambda0 <- a_lambda / length(knn_features$r_edge)
  
  t0 <- proc.time()
  MAP_estimates <- parallel::mclapply(sigma20s, function(sigma20)
    MAP_subgroup(y, X, Z, beta0, alpha0, sigma20, lambda0_init, knn_features, model, a_lambda0),
    mc.cores = mc.cores)
  pesudo_log_posts <- sapply(MAP_estimates, function(MAP) MAP$pesudo_log_post)
  best_idx <- which.max(pesudo_log_posts)
  best_MAP <- MAP_estimates[[best_idx]]
  time.MAP <- unname((proc.time() - t0)[3])
  
  beta0 <- best_MAP$beta
  alpha0 <- best_MAP$alpha
  sigma20 <- best_MAP$sigma2
  lambda00 <- best_MAP$lambda0
  
  t0 <- proc.time()
  chain <- Gibbs_subgroup(y, X, Z, beta0, alpha0, sigma20, lambda00, knn_features, model,
                               a_lambda0, nburn, niter, edge_mode = edge_mode)
  beta_samples <- chain$beta; beta_hat <- apply(beta_samples, 2, median)
  alpha_samples <- chain$alpha; alpha_hat <- apply(alpha_samples, 2, median)
  sigma2_samples <- chain$sigma2; sigma2_hat <- median(sigma2_samples)
  lambda0_samples <- chain$lambda0; lambda0_hat <- median(lambda0_samples)
  delta_samples <- chain$delta; delta_hat <- chain$delta_hat
  post_samples <- chain$post; post_hat <- median(post_samples)
  time.MCMC <- unname((proc.time() - t0)[3])
  
  # Diagnostics for convergence
  geweke <- geweke.diag(beta_samples)$z  # Geweke's diagnostic
  heidel <- heidel.diag(beta_samples)  # Heidelberger-Welch's diagnostic
  heidel.test <- isTRUE(all(heidel[, 1] == 1))
  acf_beta <- sapply(1:ncol(beta_samples), function(i)
    max(abs(acf(beta_samples[, i], plot = F)[[1]][,,1][-(1:6)])))  # Autocorrelation check
  converged <- as.integer(isTRUE(max(acf_beta) <= 0.125 & max(abs(geweke)) <= 1.96 & heidel.test))
  
  # Subgroup results
  delta_nbd <- if (edge_mode == "all") delta_hat[knn_features$knn_zeta] else delta_hat
  res_subgroup <- subgroup_res(beta_hat, delta_nbd, sigma2_hat,
                               n, r_coef, lambda0_hat, knn_features)
  Khat <- res_subgroup$Khat
  label_hat <- res_subgroup$label_hat
  eta_hat <- res_subgroup$eta_hat
  
  # Compute performance metrics (EstErr and ARI)
  EstErr_beta <- if (is.null(beta_true)) NA_real_ else 
    mean(apply(matrix(beta_hat, nrow = p) - matrix(beta_true, nrow = p), 2,
               function(x) norm(x,"2") / sqrt(p)))
  EstErr_alpha <- if (is.null(alpha_true)) NA_real_ else norm(alpha_hat - alpha_true, "2") / sqrt(q)
  ARI_hat <- if (is.null(label_true)) NA_real_ else adjustedRandIndex(label_hat, label_true)
  logfs <- sapply(1:niter, function(i) {
    logf(y, X, Z, beta_samples[i,], alpha_samples[i,], sigma2_samples[i])
  })
  row_max <- apply(logfs, 1, max)
  lppd <- sum(row_max + log(rowMeans(exp(logfs - row_max))))
  IC <- -2 * lppd + Cn * log(n) * (Khat * p + q)
  time.final <- unname((proc.time() - t_start)[3])
  
  result <- list(beta_hat = beta_hat, alpha_hat = alpha_hat, sigma2_hat = sigma2_hat,
                 lambda0_hat = lambda0_hat, delta_hat = delta_hat, post_hat = post_hat,
                 EstErr_beta = EstErr_beta, EstErr_alpha = EstErr_alpha, Khat = Khat,
                 eta_hat = eta_hat, label_hat = label_hat, ARI_hat = ARI_hat,
                 check_converge = list(geweke = geweke, heidel = heidel,
                                       acf_beta = acf_beta, converged = converged),
                 MAP_estimate = best_MAP, lppd = lppd, IC = IC, edge_mode = edge_mode,
                 time = c(time.MAP, time.MCMC, time.final), a_lambda = a_lambda,
                 lambda0_init = lambda0_init, kn = kn, r_coef = r_coef, model = model)
  if (return.chain) {
    result$MAP_estimates <- MAP_estimates
    result$chain <- list(beta = beta_samples, alpha = alpha_samples,
                         sigma2 = sigma2_samples, delta = delta_samples,
                         lambda0 = lambda0_samples, post = post_samples)
  }
  return(result)
}
