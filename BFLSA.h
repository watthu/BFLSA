#pragma once
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#include <cmath>
#include <algorithm>

using namespace Rcpp;
using namespace arma;

// Result of knn distance computation
//   prob_delta : conditional probabilities of delta_ij = 1 on the edges.
//   dist, dist2: Euclidean distances ||beta_i - beta_j|| and their squares.
struct DistResult {
  vec prob_delta;
  vec dist;
  vec dist2;
};
  
// Calculate distances and edge inclusion probabilities within knn network.
//
// @param mbeta An n by p matrix, the i-th row is beta_i.
// @param ei,ej Integer vectors, 0-based indices of the two ends of each edge.
// @param r_edge A numeric vector, effective resistances of the edges.
// @param log_scale_factor A numeric scalar, log ratio of slab to spike scales.
// @param lambda0,sigma2 Numeric scalars, current values of lambda0 and sigma2.
// @param is_SS A logical scalar, true for "SS" model and false for "SSL".
//
// @return A DistResult, where prob_delta[e] = 1 / (1 + exp(L_e)) and L_e is the log prior odds plus
//   log likelihood ratio of delta_e = 0 vs 1.
static inline DistResult knn_dist(const mat& mbeta, const ivec& ei, const ivec& ej,
                                  const vec& r_edge, double log_scale_factor, double lambda0,
                                  double sigma2, bool is_SS)
{
  int ne = ei.n_elem;
  int p = mbeta.n_cols;
  DistResult res;
  res.prob_delta.set_size(ne);
  res.dist.set_size(ne);
  res.dist2.set_size(ne);

  for (int e = 0; e < ne; e++) {
    rowvec diff = mbeta.row(ei[e]) - mbeta.row(ej[e]);
    double d2 = std::max(dot(diff, diff), 1e-12);
    double d = std::sqrt(d2);
    res.dist2[e] = d2;
    res.dist[e] = d;

    double L;
    if (is_SS) {
      L = log_scale_factor * p * r_edge[e] / 2.0
        - (1.0 - std::exp(-log_scale_factor)) * lambda0 * d2 / sigma2 / 2.0;
    } else {
      L = log_scale_factor * p * r_edge[e] / 2.0
        - (1.0 - std::exp(-log_scale_factor / 2.0)) * std::sqrt(lambda0) * d / std::sqrt(sigma2);
    }
    res.prob_delta[e] = 1.0 / (1.0 + std::exp(L));
  }
  return res;
}

// Build sparse Q from edge weights Qijs.
//
// @param n An integer scalar, the sample size.
// @param ei,ej Integer vectors, 0-based indices of the two ends of each edge.
// @param Qijs A numeric vector, the weight of each edge.
//
// @return An n by n sparse graph Laplacian Q with Q_ij = -Qijs and the diagonal equal to the
//   weighted degrees.
static inline sp_mat build_Q(int n, const ivec& ei, const ivec& ej, const vec& Qijs)
{
  int ne = ei.n_elem;
  vec dg = zeros<vec>(n);
  for (int e = 0; e < ne; e++) {
    dg[ei[e]] += Qijs[e];
    dg[ej[e]] += Qijs[e];
  }
  umat locs(2, 2 * ne + n);
  vec vals(2 * ne + n);
  int k = 0;
  for (int e = 0; e < ne; e++) {
    locs(0,k) = ei[e]; locs(1,k) = ej[e]; vals[k] = -Qijs[e]; k++;
    locs(0,k) = ej[e]; locs(1,k) = ei[e]; vals[k] = -Qijs[e]; k++;
  }
  for (int i = 0; i < n; i++) {
    locs(0,k) = i; locs(1,k) = i; vals[k] = dg[i]; k++;
  }
  return sp_mat(locs, vals, n, n);
}

// Build A = XtX + Kron(Q, I_p) as a dense matrix.
//
// @param XtX An np by np matrix, X^T X.
// @param Q An n by n sparse Laplacian returned by build_Q.
// @param n,p Integer scalars, the sample size and the dimension of beta_i.
//
// @return An np by np matrix, the precision matrix (up to sigma2) of beta.
static inline mat build_A(const mat& XtX, const sp_mat& Q, int n, int p)
{
  int np = n * p;
  umat locs(2, Q.n_nonzero * p);
  vec vals(Q.n_nonzero * p);
  int k = 0;
  for (sp_mat::const_iterator it = Q.begin(); it != Q.end(); ++it) {
    int qi = it.row(), qj = it.col();
    double qv = *it;
    for (int a = 0; a < p; a++) {
      locs(0,k) = qi * p + a; locs(1,k) = qj * p + a; vals[k] = qv; k++;
    }
  }
  sp_mat KronQI(locs, vals, np, np);
  mat A = XtX + mat(KronQI);
  return A;
}

// Cholesky solve: returns A^{-1} b via LL' decomposition. Falls back to a general solve if A is not
// numerically positive definite.
static inline vec chol_solve(const mat& A, const vec& b)
{
  mat L;
  if (!chol(L, A, "lower"))
    return solve(A, b);
  return solve(trimatu(L.t()), solve(trimatl(L), b));
}

// Cap prob_delta at MST edges to ensure graph connectivity.
//
// @param delta A numeric vector, probabilities (or indicators) on the edges.
// @param mst_idx_in_knn An integer vector, 0-based positions of MST edges in the neighborhood
//   edges.
//
// @return A copy of delta with MST entries capped at 1 - 1e-6, so that the MST edges always keep a
//   (small) spike weight.
static inline vec cap_mst_delta(const vec& delta, const ivec& mst_idx_in_knn)
{
  vec temp = delta;
  for (int k = 0; k < (int)mst_idx_in_knn.n_elem; k++)
    temp[mst_idx_in_knn[k]] = std::min(temp[mst_idx_in_knn[k]], 1.0 - 1e-6);
  return temp;
}

// Sample InvGauss(mu, lambda=1) via Michael-Schucany-Haas algorithm. Note that the input is the
// logarithm transformed version.
//
// @param log_mu A numeric scalar, log of the mean mu.
//
// @return A numeric scalar, one draw from InvGauss(mu, 1). Working on the log scale avoids overflow
//   when mu is very large (distance near 0).
static inline double rinvgauss_scalar(double log_mu)
{
  double v = R::rnorm(0.0, 1.0);
  double y = v * v;
  double log_y = std::log(y);
  double log_x = -std::log(std::exp(-log_mu) + y / 2.0
                            + std::sqrt(y * y / 4.0 + std::exp(log_y - log_mu)));
  double lu = std::log(R::runif(0.0, 1.0));
  double lp = -std::log1p(std::exp(log_x - log_mu));
  return (lu <= lp) ? std::exp(log_x) : std::exp(2.0 * log_mu - log_x);
}
