library(ART)

# Case 1
d <- readRDS("case1.rds")
X <- as.matrix(d$X)
alphas <- c(0, 0.33, 0.66, 1)

res  <- art_path(X, alphas = alphas, perplexity = 50)
res2 <- art_path(X, alphas = alphas, perplexity = 50)

stopifnot(length(res$frames) == length(alphas))
stopifnot(nrow(res$diagnostics) == length(alphas))
for (Y in res$frames) stopifnot(dim(Y) == c(nrow(X), 2), is.finite(Y))
stopifnot(identical(res$frames, res2$frames))
