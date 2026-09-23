// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp14)]]

#include "BFLSA.h"

// Compute the pseudo log-posterior up to a constant.
//
// The indicators delta_ij are replaced by their conditional probabilities prob_delta (an EM-type
// expectation); "SSL" uses the Laplace penalty directly.
//
// @return A numeric scalar, used to compare MAP solutions.
static double pesudo_log_post(const vec& y, const mat& X, const mat& Z,
                              const vec& beta, const vec& alpha, const vec& prob_delta,
                              double sigma2, double lambda0, const ivec& ei, const ivec& ej,
                              const vec& r_edge, double log_scale_factor, double a_lambda,
                              int n, int p, bool is_SS)
{
  mat mbeta = reshape(beta, p, n).t();
  int ne = ei.n_elem;
  
  // Calculate the penalty term
  double penalty = 0.0;
  for (int e = 0; e < ne; e++) {
    rowvec diff = mbeta.row(ei[e]) - mbeta.row(ej[e]);
    double d2 = std::max(dot(diff, diff), 1e-12);
    double d = std::sqrt(d2);
    if (is_SS) {
      double mix = 1.0 - prob_delta[e] + std::exp(-log_scale_factor) * prob_delta[e];
      penalty += lambda0 * mix * d2;
    } else {
      double mix = 1.0 - prob_delta[e] + std::exp(-log_scale_factor / 2.0) * prob_delta[e];
      penalty += std::sqrt(lambda0) * mix * d;
    }
  }
  penalty = is_SS ? penalty / (sigma2 * 2.0) : penalty / std::sqrt(sigma2);
  
  // Calculate the rest terms
  vec resid = y - X * beta - Z * alpha;
  double ss = dot(resid, resid) / sigma2 / 2.0;
  double lp = -((n * (p + 1.0) - p) / 2.0 + 1.0) * std::log(sigma2) - ss - penalty;
  for (int e = 0; e < ne; e++)
    lp += (std::log(lambda0) - prob_delta[e] * log_scale_factor) * r_edge[e] * p / 2.0;
  lp += (p * (n - 1.0) / 2.0 - 1.0) * std::log(lambda0) - a_lambda * lambda0;
  return lp;
}

// Result struct returned by a single MAP inner loop run
//   i_conv  : index of the last inner iteration.
//   log_post: pseudo log-posterior at the returned solution.
struct MAP_Once {
  vec beta;
  vec alpha;
  vec prob_delta;
  double sigma2;
  double lambda0;
  int i_conv;
  double log_post;
};


// Perform MAP estimation once
//
// Iterate the conditional-mode updates (a)-(h) below until the change of (beta, alpha, sigma2,
// lambda0) is smaller than threshold or maxiter is hit.
//
// @param beta,alpha,sigma2,lambda0 Starting values of the inner loop.
// @param threshold,maxiter The stop threshold and the maximum iterations.
// @param XtX,ZtZ Pre-computed X^T X and Z^T Z.
//
// @return A MAP_Once struct with the final estimates.
static MAP_Once run_MAP(const vec& y, const mat& X, const mat& Z,
                        vec beta, vec alpha, double sigma2, double lambda0,
                        const ivec& ei, const ivec& ej, const vec& r_edge,
                        double log_scale_factor, const ivec& mst_idx_in_knn,
                        double a_lambda, double threshold, int maxiter,
                        int n, int p, int q, int ne, bool is_SS,
                        const mat& XtX, const mat& ZtZ)
{
  MAP_Once out;
  out.beta = beta;
  out.alpha = alpha;
  out.prob_delta = NumericVector(ne, 1.0);
  out.sigma2 = sigma2;
  out.lambda0 = lambda0;
  out.i_conv = 0;
  out.log_post = -datum::inf;

  // Initial knn result for the entry beta
  DistResult kres = knn_dist(reshape(beta, p, n).t(), ei, ej, r_edge,
                             log_scale_factor, lambda0, sigma2, is_SS);

  for (int i = 0; i < maxiter; i++) {
    // (a) Cap prob_delta at MST edges
    vec temp_delta = cap_mst_delta(kres.prob_delta, mst_idx_in_knn);

    // (b) Compute Qijs for the current model
    vec Qijs(ne);
    if (is_SS) {
      for (int e = 0; e < ne; e++) {
        if (1.0 - temp_delta[e] < 1e-13) {
        	Qijs[e] = std::exp(std::log(lambda0) - 13);
        } else {
        	double mix = 1.0 - temp_delta[e] + std::exp(-log_scale_factor) * temp_delta[e];
          Qijs[e] = lambda0 * mix;
        }
      }
    } else {
      for (int e = 0; e < ne; e++) {
        if (1.0 - temp_delta[e] < 1e-13) {
        	Qijs[e] = std::exp(std::log(lambda0 * sigma2) / 2.0 - 6.5 - std::log(kres.dist[e]));
        } else {
          double mix = 1.0 - temp_delta[e] + std::exp(-log_scale_factor / 2.0) * temp_delta[e];
          Qijs[e] = std::sqrt(lambda0) * mix * std::sqrt(sigma2) / kres.dist[e];
        }
      }
    }

    // (c) Update beta via Cholesky solve
    sp_mat Q = build_Q(n, ei, ej, Qijs);
    mat A = build_A(XtX, Q, n, p);
    vec beta1 = chol_solve(A, X.t() * (y - Z * alpha));
    
    // (d) Update alpha
    vec alpha1 = solve(ZtZ, Z.t() * (y - X * beta1));

    // (e) Recompute distances with updated beta
    mat mbeta1 = reshape(beta1, p, n).t();
    vec dist2_new(ne);
    for (int e = 0; e < ne; e++) {
      rowvec diff = mbeta1.row(ei[e]) - mbeta1.row(ej[e]);
      dist2_new[e] = std::max(dot(diff, diff), 1e-12);
    }

    // (f) Update sigma2
    double a_sigma2 = (n * (p + 1.0) - p) / 2.0;
    vec resid = y - X * beta1 - Z * alpha1;
    double b_sigma2 = dot(resid, resid) / 2.0 + dot(Qijs, dist2_new) / 2.0;
    double sigma21 = b_sigma2 / (a_sigma2 + 1.0);
    
    // (g) Update lambda0
    double a_lambda0 = p * (n - 1.0);
    double b_lambda0 = a_lambda;
    for (int e = 0; e < ne; e++)
      b_lambda0 += Qijs[e] / lambda0 * dist2_new[e] / sigma21 / 2.0;
    double lambda01 = (a_lambda0 - 1.0) / b_lambda0;
    
    // (h) Check convergence
    double diff_val = norm(beta1 - beta, 2) / std::sqrt((double)(n * p)) 
      + norm(alpha1 - alpha, 2) / std::sqrt((double)q) 
      + std::abs(sigma21 - sigma2) 
      + std::abs(lambda01 - lambda0);
    double lp = pesudo_log_post(y, X, Z, beta1, alpha1, kres.prob_delta, sigma21, lambda01,
                                ei, ej, r_edge, log_scale_factor, a_lambda, n, p, is_SS);

    // Advance local state
    beta = beta1;
    alpha = alpha1;
    sigma2 = sigma21;
    lambda0 = lambda01;
    kres = knn_dist(reshape(beta, p, n).t(), ei, ej, r_edge, log_scale_factor, lambda0, sigma2,
                    is_SS);
    
    // Write into output struct every iteration so final values are always set
    out.beta = beta;
    out.alpha = alpha;
    out.prob_delta = kres.prob_delta;
    out.sigma2 = sigma2;
    out.lambda0 = lambda0;
    out.i_conv = i;
    out.log_post = lp;
    
    if (diff_val < threshold) break;
  }
  return out;
}

// Full MAP exploration
//
// Repeatedly run run_MAP(), accepting a solution when its pseudo log-posterior does not decrease by
// more than 1e-2. When the last three accepted values plateau, lambda0 for the next restart is
// decayed by exp(-0.1); stop when it falls below 1e-2 or after maxiter rounds.
//
// @param y,X,Z Response, block-diagonal design (n by np) and common design.
// @param beta0,alpha0,sigma20,lambda00 Initial values.
// @param r_edge,log_scale_factor,ei,ej,mst_idx_in_knn Neighborhood edges and their features, see
//   get_knn_features() and get_knn_pairs() in R.
// @param model "SS" or "SSL".
// @param a_lambda The prior rate of lambda0.
// @param threshold,maxiter The stop threshold and the maximum iterations.
//
// @return A List with the best beta, alpha, prob_delta, sigma2, lambda0, the best pseudo
//   log-posterior, and the iteration / log-posterior histories.
// [[Rcpp::export]]
List MAP_full_cpp(const arma::vec& y, const arma::mat& X, const arma::mat& Z,
                  arma::vec beta0, arma::vec alpha0, double sigma20, double lambda00,
                  const arma::vec& r_edge, double log_scale_factor,
                  const arma::ivec& ei, const arma::ivec& ej,
                  const arma::ivec& mst_idx_in_knn, std::string model,
                  double a_lambda, double threshold, int maxiter)
{
  if (maxiter < 1 || threshold <= 0) stop("Invalid MAP iteration controls.");
  if (model != "SS" && model != "SSL") stop("Unknown model.");
  const bool is_SS = (model == "SS");
  int n = X.n_rows;
  int np = X.n_cols;
  int p = np / n;
  int q = Z.n_cols;
  int ne = ei.n_elem;

  // Pre-compute XtX and ZtZ once
  const mat XtX = X.t() * X;
  const mat ZtZ = Z.t() * Z;
  
  // Best-solution trackers
  double lp_best = -datum::inf;
  vec beta_hat = beta0;
  vec alpha_hat = alpha0;
  vec prob_delta_hat = NumericVector(ne, 1.0);
  double sigma2_hat = sigma20;
  double lambda0_hat = lambda00;

  int total_iters = 0;
  std::vector<double> lp_history;
  std::vector<int> iter_used;
  double lambda0_init = lambda00;
  
  // Current running params - passed by value into run_MAP each iteration
  vec beta_cur = beta0;
  vec alpha_cur = alpha0;
  double sigma2_cur = sigma20;
  double lambda0_cur = lambda00;
  
  while (true) {
    Rcpp::checkUserInterrupt();
    // Run inner MAP loop
    MAP_Once res = run_MAP(y, X, Z, beta_cur, alpha_cur, sigma2_cur, lambda0_cur,
                           ei, ej, r_edge, log_scale_factor, mst_idx_in_knn,
                           a_lambda, threshold, maxiter, n, p, q, ne, is_SS, XtX, ZtZ);

    int i_conv = res.i_conv + 1;
    double lp_cur = res.log_post;
    iter_used.push_back(i_conv);
    total_iters++;

    // Accept or reject the result
    if (lp_cur - lp_best > -1e-2) {
      // Accept: update best solution and advance running state
      lp_best = lp_cur;
      beta_hat = res.beta;
      alpha_hat = res.alpha;
      prob_delta_hat = res.prob_delta;
      sigma2_hat = res.sigma2;
      lambda0_hat = res.lambda0;
      lp_history.push_back(lp_cur);
      beta_cur = res.beta;
      alpha_cur = res.alpha;
      sigma2_cur = res.sigma2;
    }

    // Decay lambda0 when the last 3 accepted log-posteriors have plateaued
    if ((int)lp_history.size() >= 3) {
      auto b3 = lp_history.end() - 3;
      double hi = *std::max_element(b3, lp_history.end());
      double lo = *std::min_element(b3, lp_history.end());
      if (hi - lo < 1e-3) {
        lambda0_init *= std::exp(-0.1);
        if (lambda0_init < 1e-2) break;
      }
    }

    // Reset lambda0 for the next outer round
    lambda0_cur = lambda0_init;

    // Stop if total iteration budget is exhausted
    if (total_iters >= maxiter) break;
  }

  return List::create(
    Named("beta") = NumericVector(beta_hat.begin(), beta_hat.end()),
    Named("alpha") = NumericVector(alpha_hat.begin(), alpha_hat.end()),
    Named("prob_delta") = NumericVector(prob_delta_hat.begin(), prob_delta_hat.end()),
    Named("sigma2") = sigma2_hat,
    Named("lambda0") = lambda0_hat,
    Named("pesudo_log_post") = lp_best,
    Named("iteration") = IntegerVector(iter_used.begin(), iter_used.end()),
    Named("pesudo_log_posts") = NumericVector(lp_history.begin(), lp_history.end())
  );
}
