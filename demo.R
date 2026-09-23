source("BFLSA_functions.R")

# Example 1
load("simdata_itcp.RData"); set.seed(22)
result_SS <- BFLSA(y = simdata$y, X_ = simdata$X_, Z = simdata$Z, beta_true = simdata$beta_true, 
                   alpha_true = simdata$alpha_true, label_true = simdata$label_true, 
                   nburn = 2000, niter = 5000, a_lambda = 1, lambda0_init = 20,
                   kn = 20, r_coef = 1, model = "SS", return.chain = FALSE, edge_mode = "all")
result_SS$beta_hat       # Estimated individualized coefficients
result_SS$EstErr_beta    # EstErr for individualized coefficients
result_SS$Khat           # Estimated number of subgroups
result_SS$ARI_hat        # ARI between true and estimated labels

result_SSL <- BFLSA(y = simdata$y, X_ = simdata$X_, Z = simdata$Z, beta_true = simdata$beta_true, 
                   alpha_true = simdata$alpha_true, label_true = simdata$label_true, 
                   nburn = 2000, niter = 5000, a_lambda = 1, lambda0_init = 20,
                   kn = 20, r_coef = 1, model = "SSL", return.chain = TRUE, edge_mode = "nbd")
result_SSL$alpha_hat     # Estimated common coefficients
result_SSL$EstErr_alpha  # EstErr for common coefficients
result_SSL$delta_hat     # Estimated pairwise indicators
result_SSL$chain         # Full sampling chain

# Example 2
load("simdata_slope.RData"); set.seed(22)
result_SS <- BFLSA(y = simdata$y, X_ = simdata$X_, Z = simdata$Z, beta_true = simdata$beta_true, 
                   alpha_true = simdata$alpha_true, label_true = simdata$label_true, 
                   nburn = 2000, niter = 5000, a_lambda = 1, lambda0_init = 20,
                   kn = 20, r_coef = 0.25, model = "SS", return.chain = TRUE, edge_mode = "nbd")
result_SS$beta_hat       # Estimated individualized coefficients
result_SS$EstErr_beta    # EstErr for individualized coefficients
result_SS$delta_hat      # Estimated pairwise indicators
result_SS$chain          # Full sampling chain

result_SSL <- BFLSA(y = simdata$y, X_ = simdata$X_, Z = simdata$Z, beta_true = simdata$beta_true, 
                    alpha_true = simdata$alpha_true, label_true = simdata$label_true, 
                    nburn = 2000, niter = 5000, a_lambda = 1, lambda0_init = 20,
                    kn = 20, r_coef = 0.25, model = "SSL", return.chain = FALSE, edge_mode = "all")
result_SSL$alpha_hat     # Estimated common coefficients
result_SSL$EstErr_alpha  # EstErr for common coefficients
result_SSL$Khat          # Estimated number of subgroups
result_SSL$ARI_hat       # ARI between true and estimated labels

# Example 3
load("simdata_itcp.RData")
label_true <- simdata$label_true
n <- length(label_true)

set.seed(22)
p_w <- 1; p_b <- 0.7
Adjacency_matrix <- matrix(0, n, n)
for (i in 1:(n - 1)) {
  for (j in (i + 1):n) {
    p_edge <- ifelse(label_true[i] == label_true[j], p_w, p_b)
    Adjacency_matrix[i, j] <- rbinom(1, 1, p_edge)
    Adjacency_matrix[j, i] <- Adjacency_matrix[i, j]
  }
}
result_SS <- BFLSA(y = simdata$y, X_ = simdata$X_, Z = simdata$Z, beta_true = simdata$beta_true, 
                   alpha_true = simdata$alpha_true, label_true = simdata$label_true, 
                   nburn = 2000, niter = 5000, a_lambda = 1, lambda0_init = 20,
                   kn = 20, r_coef = 0.5, model = "SS", A_prior = Adjacency_matrix, edge_mode = "all")
result_SS$beta_hat       # Estimated individualized coefficients
result_SS$EstErr_beta    # EstErr for individualized coefficients
result_SS$delta_hat      # Estimated pairwise indicators
result_SS$ARI_hat        # ARI between true and estimated labels

set.seed(22)
p_w <- 0.7; p_b <- 1
Adjacency_matrix <- matrix(0, n, n)
for (i in 1:(n - 1)) {
  for (j in (i + 1):n) {
    p_edge <- ifelse(label_true[i] == label_true[j], p_w, p_b)
    Adjacency_matrix[i, j] <- rbinom(1, 1, p_edge)
    Adjacency_matrix[j, i] <- Adjacency_matrix[i, j]
  }
}
result_SSL <- BFLSA(y = simdata$y, X_ = simdata$X_, Z = simdata$Z, beta_true = simdata$beta_true, 
                    alpha_true = simdata$alpha_true, label_true = simdata$label_true, 
                    nburn = 2000, niter = 5000, a_lambda = 1, lambda0_init = 20,
                    kn = 20, r_coef = 0.5, model = "SSL", A_prior = Adjacency_matrix, edge_mode = "nbd")
result_SSL$alpha_hat     # Estimated common coefficients
result_SSL$EstErr_alpha  # EstErr for common coefficients
result_SSL$delta_hat     # Estimated pairwise indicators
result_SSL$ARI_hat       # ARI between true and estimated labels
