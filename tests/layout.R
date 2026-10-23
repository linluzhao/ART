library(ART)

case1 <- read.csv("Case1_400.csv", row.names = 1)
X <- as.matrix(case1[, 1:10])
alphas <- c(0, 0.33, 0.66, 1)

reference <- readRDS("reference.rds")

prep <- art_prepare(X, perplexity = 50)

for (k in 1:4) {
  P <- fuse(prep, alphas[k])$P
  Y <- art_layout(P, prep$start, max_iter = 10)$Y
  stopifnot(all.equal(Y, reference$Y[[k]], tolerance = 1e-6))
}
