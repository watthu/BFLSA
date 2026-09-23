// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp14)]]

#include "BFLSA.h"

// Gibbs update for beta (individual coefficients)
//
// Draw beta ~ N(A^{-1} X^T (y - Z alpha), sigma2 * A^{-1}), where A = X^T X + Kron(Q, I_p) and Q is
// the Laplacian with edge weights lambda0 * {1 - delta + exp(-log_scale_factor) * delta} * s_inv.
//
// @param delta,s_inv Numeric vectors, current delta_ij and inverse auxiliary variables on the
//   neighborhood edges (s_inv = 1 for "SS").
// @param XtX An np by np matrix, pre-computed X^T X.
//
// @return A length-np vector, one draw of beta.
static vec Gibbs_beta(const vec& y, const mat& X, const mat& Z, const vec& alpha,
                      const vec& delta, const vec& s_inv, double sigma2, double lambda0,
                      int n, int p, const ivec& ei, const ivec& ej,
                      double log_scale_factor, const ivec& mst_idx_in_knn, const mat& XtX)
{
  int ne = ei.n_elem;
  int np = n * p;
  vec temp_delta = cap_mst_delta(delta, mst_idx_in_knn);

  vec Qijs(ne);
  for (int e = 0; e < ne; e++) {
    if (temp_delta[e] == 1) {
      Qijs[e] = std::exp(std::log(lambda0) + std::log(s_inv[e]) - log_scale_factor);
    } else {
      double mix = 1.0 - temp_delta[e] + std::exp(-log_scale_factor) * temp_delta[e];
      Qijs[e] = lambda0 * mix * s_inv[e];
    }
  }
  sp_mat Q = build_Q(n, ei, ej, Qijs);
  mat A = build_A(XtX, Q, n, p);
  vec rhs = X.t() * (y - Z * alpha);
  
  mat L;
  if (!chol(L, A, "lower")) {
    mat Ainv = inv_sympd(A);
    mat Sigma_beta = sigma2 * Ainv;
    mat Lc; chol(Lc, Sigma_beta, "lower");
    return Ainv * rhs + Lc * randn<vec>(np);
  }
  
  vec mu_beta = solve(trimatu(L.t()), solve(trimatl(L), rhs));
  return mu_beta + std::sqrt(sigma2) * solve(trimatu(L.t()), randn<vec>(np));
}

// Gibbs update for alpha (common coefficients)
//
// Draw alpha ~ N((Z^T Z)^{-1} Z^T (y - X beta), sigma2 * (Z^T Z)^{-1}).
//
// @return A length-q vector, one draw of alpha.
static vec Gibbs_alpha(const vec& y, const mat& X, const mat& Z, const vec& beta,
                       double sigma2, const mat& ZtZ)
{
  mat ZtZ_inv = inv_sympd(ZtZ);
  mat Sigma_alpha = sigma2 * ZtZ_inv;
  vec mu_alpha = ZtZ_inv * Z.t() * (y - X * beta);
  mat L; chol(L, Sigma_alpha, "lower");
  return mu_alpha + L * randn<vec>(Z.n_cols);
}

// Gibbs update for delta (edge linkage indicators)
//
// @param prob_delta A numeric vector, P(delta_ij = 1 | rest) of each edge.
//
// @return A 0/1 numeric vector, one draw of delta (1 = slab, i.e. separated).
static vec Gibbs_delta(const vec& prob_delta)
{
  int ne = prob_delta.n_elem;
  vec delta(ne);
  for (int e = 0; e < ne; e++)
    delta[e] = R::rbinom(1, prob_delta[e]);
  return delta;
}

// Gibbs update for s_inv (inverse for auxiliary variables)
//
// Only used by the "SSL" model, where the Laplace prior is written as a scale mixture of normals;
// 1 / s_ij follows an inverse Gaussian law.
//
// @param dist A numeric vector, current distances ||beta_i - beta_j||.
//
// @return A numeric vector, one draw of s_inv on the neighborhood edges.
static vec Gibbs_s_inv(const vec& dist, const vec& delta, double sigma2, double lambda0,
                       double log_scale_factor, const ivec& mst_idx_in_knn)
{
  vec temp_delta = cap_mst_delta(delta, mst_idx_in_knn);
  int ne = delta.n_elem;
  vec s_inv(ne);
  for (int e = 0; e < ne; e++) {
    double log_temp;
    if (temp_delta[e] == 1) {
    	 log_temp = std::log(sigma2) - std::log(lambda0) + log_scale_factor;
    } else {
      double mix = 1.0 - temp_delta[e] + std::exp(-log_scale_factor) * temp_delta[e];
      log_temp = std::log(sigma2) - std::log(lambda0) - std::log(mix);
    }
    double log_mu_e = log_temp / 2.0 - std::log(dist[e]);
    s_inv[e] = rinvgauss_scalar(log_mu_e);
  }
  return s_inv;
}

// Gibbs update for sigma2 (residual variance)
//
// Draw sigma2 ~ InvGamma((n(p + 1) - p) / 2, RSS / 2 + penalty / 2).
//
// @param dist2 A numeric vector, squared distances on the neighborhood edges.
//
// @return A numeric scalar, one draw of sigma2.
static double Gibbs_sigma2(const vec& y, const mat& X, const mat& Z,
                           const vec& beta, const vec& alpha, const vec& dist2,
                           const vec& delta, const vec& s_inv, double lambda0,
                           double log_scale_factor, int n, int p)
{
  double a_sigma2 = (n * (p + 1.0) - p) / 2.0;
  vec resid = y - X * beta - Z * alpha;
  double pen = 0.0;
  int ne = delta.n_elem;
  for (int e = 0; e < ne; e++) {
    if (delta[e] == 1) {
      pen += std::exp(std::log(lambda0) - log_scale_factor
                      + std::log(s_inv[e]) + std::log(dist2[e]));
    } else {
      double mix = 1.0 - delta[e] + std::exp(-log_scale_factor) * delta[e];
      pen += lambda0 * mix * s_inv[e] * dist2[e];
    }
  }
  double b_sigma2 = dot(resid, resid) / 2.0 + pen / 2.0;
  return 1.0 / R::rgamma(a_sigma2, 1.0 / b_sigma2);
}

// Gibbs update for lambda0 (spike parameter)
//
// Draw lambda0 ~ Gamma(p(n - 1), a_lambda + penalty / (2 sigma2)).
//
// @param a_lambda A positive numeric scalar, the prior rate of lambda0.
//
// @return A numeric scalar, one draw of lambda0.
static double Gibbs_lambda0(const vec& dist2, const vec& delta, const vec& s_inv,
                            double sigma2, double log_scale_factor,
                            double a_lambda, int n, int p)
{
  double a_lambda0 = p * (n - 1.0);
  double pen = 0.0;
  int ne = delta.n_elem;
  for (int e = 0; e < ne; e++) {
  	if (delta[e] == 1) {
    	 pen += std::exp(-log_scale_factor + std::log(s_inv[e]) + std::log(dist2[e]));
    } else {
      double mix = 1.0 - delta[e] + std::exp(-log_scale_factor) * delta[e];
      pen += mix * s_inv[e] * dist2[e];
    }
  }
  double b_lambda0 = pen / sigma2 / 2.0 + a_lambda;
  return R::rgamma(a_lambda0, 1.0 / b_lambda0);
}

// Compute the log-posterior up to a constant.
//
// @return A numeric scalar, the log-posterior of the current state, used to monitor the chain
//   (returned as "post").
static double log_posterior(const vec& y, const mat& X, const mat& Z,
                            const vec& beta, const vec& alpha, const vec& delta, const vec& s_inv,
                            double sigma2, double lambda0, const ivec& ei, const ivec& ej,
                            const vec& r_edge, double log_scale_factor, double a_lambda,
                            int n, int p, bool is_SS)
{
  mat mbeta = reshape(beta, p, n).t();
  int ne = ei.n_elem;
  double penalty = 0.0;
  for (int e = 0; e < ne; e++) {
    rowvec diff = mbeta.row(ei[e]) - mbeta.row(ej[e]);
    double d2 = std::max(dot(diff, diff), 1e-12);
    if (delta[e] == 1) {
    	 penalty += std::exp(std::log(lambda0) - log_scale_factor + std::log(s_inv[e]) + std::log(d2));
    } else {
      double mix = 1.0 - delta[e] + std::exp(-log_scale_factor) * delta[e];
      penalty += lambda0 * mix * s_inv[e] * d2;
    }
  }
  penalty /= (sigma2 * 2.0);

  vec resid = y - X * beta - Z * alpha;
  double ss = dot(resid, resid) / sigma2 / 2.0;
  double lp = -((n * (p + 1.0) - p) / 2.0 + 1.0) * std::log(sigma2) - ss - penalty;
  for (int e = 0; e < ne; e++)
    lp += (std::log(lambda0) - delta[e] * log_scale_factor) * r_edge[e] * p / 2.0;
  lp += (p * (n - 1.0) / 2.0 - 1.0) * std::log(lambda0) - a_lambda * lambda0;
  if (!is_SS)
    for (int e = 0; e < ne; e++)
      lp += 0.5 * std::log(s_inv[e]) - 0.5 / s_inv[e];
  return lp;
}

// Gibbs sampler algorithm: Iterate over specified number of iterations
//
// @param y,X,Z Response, block-diagonal design (n by np) and common design.
// @param beta0,alpha0,sigma20,lambda00 Initial values (e.g., MAP estimates).
// @param log_scale_factor,r_edge,ei,ej,mst_idx_in_knn Neighborhood edges and their features, see
//   get_knn_features() and get_knn_pairs() in R.
// @param r_edge_all,ei_all,ej_all,knn_idx_in_all The same for all pairs, and 0-based positions of
//   the neighborhood edges in all pairs.
// @param model "SS" or "SSL".
// @param a_lambda The prior rate of lambda0.
// @param nburn,niter Numbers of burn-in and posterior samples.
// @param return_burn Whether to return the burn-in samples.
// @param edge_mode "all" (sample delta for all pairs) or "nbd" (neighborhood edges only).
//
// @return A List with posterior samples of beta, alpha, delta, sigma2, lambda0, the log-posterior
//   "post" and posterior means "delta_hat".
// [[Rcpp::export]]
List Gibbs_sampler_cpp(const arma::vec& y, const arma::mat& X, const arma::mat& Z,
                       arma::vec beta0, arma::vec alpha0, double sigma20, double lambda00,
                       double log_scale_factor, const arma::vec& r_edge, const arma::ivec& ei,
                       const arma::ivec& ej, const arma::ivec& mst_idx_in_knn, 
                       const arma::vec& r_edge_all, const arma::ivec& ei_all,
                       const arma::ivec& ej_all, const arma::ivec& knn_idx_in_all,
                       std::string model, double a_lambda, int nburn, int niter, bool return_burn,
                       std::string edge_mode = "all")
{
  if (nburn < 0 || niter < 1) stop("Require nburn >= 0 and niter >= 1.");
  if (model != "SS" && model != "SSL") stop("Unknown model.");
  if (edge_mode != "all" && edge_mode != "nbd")
    stop("edge_mode must be 'all' or 'nbd'.");
  const bool use_all_edges = (edge_mode == "all");
  const bool is_SS = (model == "SS");
  int n = X.n_rows;
  int np = X.n_cols;
  int p = np / n;
  int q = Z.n_cols;
  int ne = ei.n_elem;
  int ne_all = ei_all.n_elem;
  const int n_edges = use_all_edges ? ne_all : ne;
  
  // Pre-compute XtX and ZtZ once
  const mat XtX = X.t() * X;
  const mat ZtZ = Z.t() * Z;

  // Storage matrices for burn-in and sampling chains
  mat beta_burn, beta_iter(niter, np);
  mat alpha_burn, alpha_iter(niter, q);
  vec sigma2_burn, sigma2_iter(niter);
  vec lambda0_burn, lambda0_iter(niter);
  vec posts_burn, posts_iter(niter);
  vec delta_iter_sum = arma::zeros<arma::vec>(n_edges);
  mat delta_iter(niter, n_edges);  

  // Running state - initialise and record row 0
  vec beta = beta0;
  vec alpha = alpha0;
  double sigma2 = sigma20;
  double lambda0 = lambda00;
  vec delta_selected(n_edges, fill::zeros);
  vec s_inv(ne, fill::ones);
  vec delta_knn(ne, fill::zeros);
  if (return_burn) {
    beta_burn.set_size(nburn, np);
    alpha_burn.set_size(nburn, q);
    sigma2_burn.set_size(nburn);
    lambda0_burn.set_size(nburn);
    posts_burn.set_size(nburn);
  }

  // One full Gibbs step - shared between burn-in and sampling loops
  auto gibbs_step = [&]() {
    mat mbeta = reshape(beta, p, n).t();
    DistResult res1_knn = knn_dist(mbeta, ei, ej, r_edge,
                                   log_scale_factor, lambda0, sigma2, is_SS);
    if (use_all_edges) {
      DistResult res1_all = knn_dist(mbeta, ei_all, ej_all, r_edge_all,
                                    log_scale_factor, lambda0, sigma2, is_SS);
      delta_selected = Gibbs_delta(res1_all.prob_delta);
      for (int e = 0; e < ne; e++)
        delta_knn[e] = delta_selected[knn_idx_in_all[e]];
    } else {
      delta_knn = Gibbs_delta(res1_knn.prob_delta);
      delta_selected = delta_knn;
    }
    s_inv = is_SS ? vec(ne, fill::ones)
                  : Gibbs_s_inv(res1_knn.dist, delta_knn, sigma2, lambda0, log_scale_factor,
                                mst_idx_in_knn);
    beta = Gibbs_beta(y, X, Z, alpha, delta_knn, s_inv, sigma2, lambda0,
                      n, p, ei, ej, log_scale_factor, mst_idx_in_knn, XtX);
    alpha = Gibbs_alpha(y, X, Z, beta, sigma2, ZtZ);
    DistResult res2_knn = knn_dist(reshape(beta, p, n).t(), ei, ej, r_edge,
                                   log_scale_factor, lambda0, sigma2, is_SS);
    sigma2 = Gibbs_sigma2(y, X, Z, beta, alpha, res2_knn.dist2, delta_knn, s_inv,
                           lambda0, log_scale_factor, n, p);
    lambda0 = Gibbs_lambda0(res2_knn.dist2, delta_knn, s_inv, sigma2, log_scale_factor,
                            a_lambda, n, p);
  };

  // ----- Burn-in -----
  for (int brn = 0; brn < nburn; brn++) {
    Rcpp::checkUserInterrupt();
    gibbs_step();
    if (return_burn) {
      beta_burn.row(brn) = beta.t();
      alpha_burn.row(brn) = alpha.t();
      sigma2_burn[brn] = sigma2;
      lambda0_burn[brn] = lambda0;
      posts_burn[brn] = log_posterior(y, X, Z, beta, alpha, delta_knn, s_inv, sigma2, lambda0,
                                      ei, ej, r_edge, log_scale_factor, a_lambda, n, p, is_SS);
    }
  }

  // ----- Sampling -----
  for (int itr = 0; itr < niter; itr++) {
    Rcpp::checkUserInterrupt();
    gibbs_step();
    beta_iter.row(itr) = beta.t();
    alpha_iter.row(itr) = alpha.t();
    sigma2_iter[itr] = sigma2;
    lambda0_iter[itr] = lambda0;
    delta_iter.row(itr) = delta_selected.t();
    delta_iter_sum += delta_selected;
    posts_iter[itr] = log_posterior(y, X, Z, beta, alpha, delta_knn, s_inv, sigma2, lambda0, ei, ej,
                                    r_edge, log_scale_factor, a_lambda, n, p, is_SS);
  }
  vec delta_iter_mean = delta_iter_sum / niter;

  List result = List::create(
    Named("beta") = beta_iter,
    Named("alpha") = alpha_iter,
    Named("delta") = delta_iter,
    Named("sigma2") = NumericVector(sigma2_iter.begin(), sigma2_iter.end()),
    Named("lambda0") = NumericVector(lambda0_iter.begin(), lambda0_iter.end()),
    Named("post") = NumericVector(posts_iter.begin(), posts_iter.end()),
    Named("delta_hat") = NumericVector(delta_iter_mean.begin(), delta_iter_mean.end()),
    Named("edge_mode") = edge_mode
  );
  if (return_burn) {
    result["beta_burn"] = beta_burn;
    result["alpha_burn"] = alpha_burn;
    result["sigma2_burn"] = NumericVector(sigma2_burn.begin(), sigma2_burn.end());
    result["lambda0_burn"] = NumericVector(lambda0_burn.begin(), lambda0_burn.end());
    result["posts_burn"] = NumericVector(posts_burn.begin(), posts_burn.end());
  }
  return result;
}
