#include "tsne_exact.h"
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <cfloat>
using namespace Rcpp;

// Matrices stored by row: (n, m) at n * N + m
static std::vector<double> to_row_major(const NumericMatrix& M) {
  const int N = M.nrow(), K = M.ncol();
  std::vector<double> out((size_t)N * K);
  for (int n = 0; n < N; n++) for (int m = 0; m < K; m++) out[(size_t)n * K + m] = M(n, m);
  return out;
}

static NumericMatrix from_row_major(const std::vector<double>& v, int N, int K) {
  NumericMatrix M(N, K);
  for (int n = 0; n < N; n++) for (int m = 0; m < K; m++) M(n, m) = v[(size_t)n * K + m];
  return M;
}

// [[Rcpp::export]]
NumericMatrix art_squared_distance(NumericMatrix Z) {
  const int N = Z.nrow(), D = Z.ncol();
  std::vector<double> X = to_row_major(Z);
  NumericMatrix DD(N, N);
  for (int n = 0; n < N; n++) {
    DD(n, n) = 0.0;
    for (int m = n + 1; m < N; m++) {
      double acc = 0.0;
      for (int d = 0; d < D; d++) {
        const double diff = X[(size_t)n * D + d] - X[(size_t)m * D + d];
        acc += diff * diff;
      }
      DD(n, m) = acc;
      DD(m, n) = acc;
    }
  }
  return DD;
}

// Entropy for one beta, as in tsne.cpp
static void row_at(const double* d, int N, int n, double beta, std::vector<double>& p, double& s, double& H) {
  for (int m = 0; m < N; m++) p[m] = exp(-beta * d[m]);
  p[n] = DBL_MIN;
  s = DBL_MIN;
  for (int m = 0; m < N; m++) s += p[m];
  H = 0.0;
  for (int m = 0; m < N; m++) H += beta * (d[m] * p[m]);
  H = (H / s) + log(s);
}

// Perplexity search, as in tsne.cpp
static void search_rows(const std::vector<double>& D, int N, double perplexity,
                        std::vector<double>& P, std::vector<double>& beta) {
  std::vector<double> p(N);
  for (int n = 0; n < N; n++) {
    const double* d = &D[(size_t)n * N];
    double b = 1.0, min_beta = -DBL_MAX, max_beta = DBL_MAX, s = 0.0, H = 0.0;
    bool found = false;
    for (int iter = 0; iter < 200; iter++) {
      row_at(d, N, n, b, p, s, H);
      const double Hdiff = H - log(perplexity);
      if (Hdiff < 1e-5 && -Hdiff < 1e-5) { found = true; break; }
      if (Hdiff > 0) {
        min_beta = b;
        if (max_beta == DBL_MAX || max_beta == -DBL_MAX) b *= 2.0; else b = (b + max_beta) / 2.0;
      } else {
        max_beta = b;
        if (min_beta == -DBL_MAX || min_beta == DBL_MAX) b /= 2.0; else b = (b + min_beta) / 2.0;
      }
    }
    if (!found) stop("row %d did not reach the target perplexity in 200 steps", n + 1);
    for (int m = 0; m < N; m++) P[(size_t)n * N + m] = p[m] / s;
    beta[n] = b;
  }
}

// Beta of each point
// [[Rcpp::export]]
NumericVector art_search_beta(NumericMatrix DD, double perplexity) {
  const int N = DD.nrow();
  std::vector<double> D = to_row_major(DD), P((size_t)N * N), beta(N);
  search_rows(D, N, perplexity, P, beta);
  return wrap(beta);
}

// Combine the two kernels and recalibrate
// [[Rcpp::export]]
List art_fuse(NumericMatrix D_ang, NumericMatrix D_rad, NumericVector beta_ang, NumericVector beta_rad,
              double alpha, double perplexity, bool recalibrate) {
  const int N = D_ang.nrow();
  std::vector<double> Da = to_row_major(D_ang), Dr = to_row_major(D_rad);
  std::vector<double> D((size_t)N * N), P((size_t)N * N), beta(N);
  for (int n = 0; n < N; n++) {
    const double a = (1 - alpha) * beta_ang[n];
    const double b = alpha * beta_rad[n];
    for (int m = 0; m < N; m++) D[(size_t)n * N + m] = a * Da[(size_t)n * N + m] + b * Dr[(size_t)n * N + m];
  }
  if (recalibrate) {
    search_rows(D, N, perplexity, P, beta);
  } else {
    std::vector<double> p(N);
    double s, H;
    for (int n = 0; n < N; n++) {
      row_at(&D[(size_t)n * N], N, n, 1.0, p, s, H);
      for (int m = 0; m < N; m++) P[(size_t)n * N + m] = p[m] / s;
      beta[n] = 1.0;
    }
  }
  NumericVector row_perplexity(N);
  for (int n = 0; n < N; n++) {
    double acc = 0.0;
    for (int m = 0; m < N; m++) {
      const double v = P[(size_t)n * N + m];
      if (m != n && v > 0) acc += v * log(v);
    }
    row_perplexity[n] = exp(-acc);
  }
  std::vector<double> S((size_t)N * N);
  for (int n = 0; n < N; n++) {
    S[(size_t)n * N + n] = P[(size_t)n * N + n];
    for (int m = 0; m < N; m++) if (m != n) S[(size_t)n * N + m] = P[(size_t)n * N + m] + P[(size_t)m * N + n];
  }
  double total = 0.0;
  for (size_t i = 0; i < S.size(); i++) total += S[i];
  for (size_t i = 0; i < S.size(); i++) S[i] /= total;
  return List::create(_["P"] = from_row_major(S, N, N), _["perplexity"] = row_perplexity,
                      _["beta"] = wrap(beta));
}

// t-SNE layout
// [[Rcpp::export]]
List art_layout(NumericMatrix P, NumericMatrix start, int max_iter = 1000, double eta = 200,
                double exaggeration = 12, int stop_lying_iter = 250, int mom_switch_iter = 250,
                double momentum = 0.5, double final_momentum = 0.8) {
  const int N = P.nrow();
  TSNE<2> tsne(30.0, 0.0, false, max_iter, true, stop_lying_iter, mom_switch_iter,
               momentum, final_momentum, eta, exaggeration, 1);   // exact t-SNE
  tsne.P = to_row_major(P);
  std::vector<double> Y = to_row_major(start);
  std::vector<double> costs(N), itercost((size_t)std::ceil(max_iter / 50.0));
  tsne.trainIterations(N, Y.data(), costs.data(), itercost.data());
  return List::create(_["Y"] = from_row_major(Y, N, 2), _["kl"] = wrap(itercost));
}
