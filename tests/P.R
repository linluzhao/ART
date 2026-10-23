library(ART)

case1 <- read.csv("Case1_400.csv", row.names = 1)
X <- as.matrix(case1[, 1:10])
alphas <- c(0, 0.33, 0.66, 1)

reference <- readRDS("reference.rds")

prep <- art_prepare(X, perplexity = 50)

for (k in 1:4) {
  fused <- fuse(prep, alphas[k])
  P <- fused$P
  stopifnot(all(is.finite(P)))
  stopifnot(all(P >= 0))
  stopifnot(isSymmetric(P))
  stopifnot(all.equal(sum(P), 1))
  stopifnot(all.equal(fused$beta, reference$beta[[k]]))
}
