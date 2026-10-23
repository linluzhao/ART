library(ART)

case1 <- read.csv("Case1_400.csv", row.names = 1)
X <- as.matrix(case1[, 1:10])
alphas <- c(0, 0.33, 0.66, 1)

res  <- art_path(X, alphas = alphas, perplexity = 50)
res2 <- art_path(X, alphas = alphas, perplexity = 50)

stopifnot(length(res$frames) == 4)

for (Y in res$frames) {
  stopifnot(nrow(Y) == 400, ncol(Y) == 2)
  stopifnot(all(is.finite(Y)))
}

stopifnot(identical(res$frames, res2$frames))
