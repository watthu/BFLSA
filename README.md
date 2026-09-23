# BFLSA: Bayesian Fusion Learning for Subgroup Analysis

This repository implements Bayesian fusion learning for subgroup analysis with **spike-and-slab (SS)** and **spike-and-slab lasso (SSL)** shrinkage, as proposed in

> Hu, W., Zhang, W. and Li, C. (2026). *(Full title and journal to be added.)*

For individual $i = 1, \dots, n$, the model is 

```math
y_i = \boldsymbol{x}_i^\top \boldsymbol{\beta}_i + \boldsymbol{z}_i^\top \boldsymbol{\alpha} + \varepsilon_i, \qquad \varepsilon_i \sim N(0, \sigma^2)
```

where $`\boldsymbol{\beta}_i \in \mathbb{R}^p`$ are **heterogeneous** effects and $`\boldsymbol{\alpha} \in \mathbb{R}^q`$ are **common** effects. A spike-and-slab type prior is placed on the differences $`\boldsymbol{\beta}_i - \boldsymbol{\beta}_j`$ over the edges of a neighborhood graph, so that similar individuals are fused together. Subgroups are then identified from the posterior of the pairwise indicators $`\delta_{ij}`$.

R handles data preparation, graph construction and summaries; C++ (via Rcpp/RcppArmadillo) performs MAP estimation and Gibbs sampling. This is a collection of research scripts, not an installable R package.

## Requirements

Install the R dependencies once:

```r
install.packages(c("Rcpp", "RcppArmadillo", "Matrix", "igraph", "MASS", "mclust", "coda"))
```

Sourcing `BFLSA_functions.R` compiles the two C++ files with `Rcpp::sourceCpp()`, so a C++14 toolchain is required (Rtools on Windows, Xcode Command Line Tools on macOS). Keep `BFLSA.h` in the same folder as the `.cpp` files, and set the working directory to the project folder.

## Quick start

```r
source("BFLSA_functions.R")
load("simdata_itcp.RData")    # loads a list named `simdata`

set.seed(22)
result <- BFLSA(y = simdata$y, X_ = simdata$X_, Z = simdata$Z,
                beta_true = simdata$beta_true, alpha_true = simdata$alpha_true,
                label_true = simdata$label_true,
                nburn = 2000, niter = 5000, a_lambda = 1, lambda0_init = 20,
                kn = 20, r_coef = 1, model = "SS", edge_mode = "all")
result$beta_hat       # Estimated individualized coefficients
result$EstErr_beta    # EstErr for individualized coefficients
result$Khat           # Estimated number of subgroups
result$label_hat      # Estimated subgroup labels
result$ARI_hat        # ARI between true and estimated labels
```

For real data, omit `beta_true`, `alpha_true` and `label_true`; the corresponding `EstErr_*` and `ARI_hat` are returned as `NA`.

The full set of examples is in `demo.R`:

```r
source("demo.R")
```

| Example | Data | Fits |
| --- | --- | --- |
| 1 | `simdata_itcp.RData` (heterogeneous intercept) | `SS` + `edge_mode = "all"`, `SSL` + `edge_mode = "nbd"` |
| 2 | `simdata_slope.RData` (heterogeneous intercept and slope) | `SS` + `"nbd"`, `SSL` + `"all"` |
| 3 | `simdata_itcp.RData` with a simulated prior network `A_prior` | `SS` + `"all"`, `SSL` + `"nbd"` |


## Data

Each `.RData` file contains a list named `simdata` with

| Element | Description |
| --- | --- |
| `y` | Length-`n` response vector (`n = 100`). |
| `X_` | `n × p` heterogeneous covariates: a column of ones (`simdata_itcp`, `p = 1`), or ones plus a covariate (`simdata_slope`, `p = 2`). |
| `Z` | `n × 5` common covariates. |
| `beta_true` | True individual coefficients, stacked by individual (length `n * p`). |
| `alpha_true` | True common coefficients (length 5). |
| `label_true` | True subgroup labels (2 groups for `simdata_itcp`, 3 for `simdata_slope`). |

Loading both files into the same environment overwrites `simdata`.

## Functions

Functions in `BFLSA_functions.R` (each documented with roxygen-style comments):

- `BFLSA` is the main function: initialization, MAP estimation, Gibbs sampling, subgroup detection, diagnostics and evaluation.
- `get_init_mbeta` initializes the individual coefficients `beta_i` (OLS for a heterogeneous intercept; projected ridge plus local refinement for slopes).
- `get_knn_features` builds the neighborhood graph (kNN + minimum spanning tree, optionally restricted to a prior network) and computes the pairwise effective resistance.
- `get_knn_pairs` converts selected pairs to 0-based index vectors for C++.
- `MAP_subgroup` is the R wrapper of `MAP_full_cpp` (MAP estimation with `lambda0` decay).
- `Gibbs_subgroup` is the R wrapper of `Gibbs_sampler_cpp` (Gibbs sampler).
- `subgroup_res` detects subgroups from the estimated edge indicators by FDR-type thresholding and connected components.
- `logf` computes the pointwise log-likelihood (used for `lppd` and `IC`).

## Input of `BFLSA`

| Argument | Description |
| --- | --- |
| `y` | Numeric response vector of length `n`. |
| `X_` | `n × p` matrix of heterogeneous covariates (include a column of ones for a heterogeneous intercept). |
| `Z` | `n × q` matrix of common covariates, `q >= 1`. Do not duplicate an intercept already in `X_`; `cbind(X_, Z)` must have full column rank. |
| `beta_true`, `alpha_true`, `label_true` | Optional true values, only used for evaluation. `beta_true` is stacked by individual, i.e. `as.vector(t(beta_matrix))`. |
| `nburn`, `niter` | Numbers of burn-in and retained posterior samples (`niter >= 20`). |
| `a_lambda` | Positive rate of the Gamma prior on `lambda0`; divided internally by the number of neighborhood edges. |
| `lambda0_init` | Positive initial value of the spike parameter `lambda0`. |
| `kn` | Maximum number of nearest neighbors, between `1` and `n - 1`. |
| `r_coef` | Positive multiplier controlling the spike/slab separation via effective resistance. |
| `model` | `"SS"` (default) or `"SSL"`. |
| `edge_mode` | `"all"` (default) or `"nbd"`; see below. |
| `A_prior` | Optional symmetric `n × n` adjacency matrix of a prior network; neighbors are chosen among its edges. |
| `mc.cores` | Number of cores for the MAP starts (default `1`; use `1` on Windows). |
| `return.chain` | If `TRUE`, also return the posterior chains and all MAP estimates. |

### Edge modes

Both modes use only the neighborhood edges to update `beta` and to detect subgroups. They differ in which indicators `delta_ij` are sampled and returned:

| Mode | Length of `delta_hat` / columns of `chain$delta` |
| --- | --- |
| `"all"` | `choose(n, 2)` — all pairs |
| `"nbd"` | number of neighborhood edges |

Pairs are ordered as in `combn(n, 2)` (the lower-triangle order of `dist()`). Larger `delta_ij` means weaker fusion, i.e. stronger evidence that `i` and `j` are in different subgroups.

## Output of `BFLSA`

| Element | Description |
| --- | --- |
| `beta_hat`, `alpha_hat` | Posterior medians of individual and common coefficients. `matrix(beta_hat, nrow = n, byrow = TRUE)` gives an `n × p` matrix. |
| `sigma2_hat`, `lambda0_hat` | Posterior medians of the error variance and spike parameter. |
| `delta_hat` | Posterior means of the pairwise indicators. |
| `post_hat` | Posterior median of the log-posterior (up to a constant). |
| `Khat`, `label_hat`, `eta_hat` | Estimated number of subgroups, labels, and subgroup centers (`matrix(eta_hat, nrow = p)`). |
| `EstErr_beta` | Mean over individuals of `||beta_hat_i - beta_i|| / sqrt(p)`. |
| `EstErr_alpha` | `||alpha_hat - alpha|| / sqrt(q)`. |
| `ARI_hat` | Adjusted Rand index between estimated and true labels. |
| `lppd`, `IC` | Log pointwise predictive density and information criterion. |
| `check_converge` | Geweke, Heidelberger–Welch and autocorrelation diagnostics of `beta`; `converged = 1` if all pass. |
| `MAP_estimate` | The best MAP estimate used to start the sampler. |
| `time` | Elapsed seconds of MAP, MCMC and the whole run. |
| `chain`, `MAP_estimates` | Posterior samples (`beta`, `alpha`, `sigma2`, `lambda0`, `delta`, `post`) and all MAP fits, only when `return.chain = TRUE`. |


## Files

| File | Description |
| --- | --- |
| `BFLSA_functions.R` | R interface and helper functions. |
| `BFLSA.h` | Shared C++ helpers (distances, precision matrix, inverse-Gaussian sampler). |
| `MAP_subgroup.cpp` | MAP estimation. |
| `Gibbs_subgroup.cpp` | Gibbs sampler. |
| `demo.R` | Examples on the simulated data. |
| `simdata_itcp.RData`, `simdata_slope.RData` | Simulated data for heterogeneous intercept and heterogeneous slope cases. |

## Citation

If you use this code, please cite

> Hu, W., Zhang, W. and Li, C. (2026). *(Full title and journal to be added.)*
