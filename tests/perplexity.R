library(ART)

case1 <- read.csv("Case1_400.csv", row.names = 1)
X <- as.matrix(case1[, 1:10])
perplexity <- 50
alphas <- c(0, 0.33, 0.66, 1)

row_entropy <- function(D, beta) {
  H <- c()
  for (n in 1:nrow(D)) {
    p <- exp(-beta[n] * D[n, -n])
    p <- p / sum(p)
    p <- p[p > 0]
    H[n] <- -sum(p * log(p))
  }
  H
}

prep <- art_prepare(X, perplexity = perplexity)

stopifnot(max(abs(row_entropy(prep$D_ang, prep$beta_ang) - log(perplexity))) < 1e-5)
stopifnot(max(abs(row_entropy(prep$D_rad, prep$beta_rad) - log(perplexity))) < 1e-5)

for (alpha in alphas) {
  fused <- fuse(prep, alpha)
  D <- (1 - alpha) * prep$beta_ang * prep$D_ang + alpha * prep$beta_rad * prep$D_rad
  stopifnot(max(abs(row_entropy(D, fused$beta) - log(perplexity))) < 1e-5)
}
