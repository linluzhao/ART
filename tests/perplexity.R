library(ART)

# Case 1
d <- readRDS("case1.rds")
X <- as.matrix(d$X)
perplexity <- 50
alphas <- c(0, 0.33, 0.66, 1)

row_entropy <- function(D, beta) {
  vapply(seq_len(nrow(D)), function(n) {
    p <- exp(-beta[n] * D[n, -n])
    p <- p / sum(p)
    p <- p[p > 0]
    -sum(p * log(p))
  }, 0)
}

prep <- art_prepare(X, perplexity = perplexity)

stopifnot(max(abs(row_entropy(prep$D_ang, prep$beta_ang) - log(perplexity))) < 1e-5)
stopifnot(max(abs(row_entropy(prep$D_rad, prep$beta_rad) - log(perplexity))) < 1e-5)

for (alpha in alphas) {
  fused <- fuse(prep, alpha)
  D <- (1 - alpha) * prep$beta_ang * prep$D_ang + alpha * prep$beta_rad * prep$D_rad
  stopifnot(max(abs(row_entropy(D, fused$beta) - log(perplexity))) < 1e-5)
}
