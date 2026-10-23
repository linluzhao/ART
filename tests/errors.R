library(ART)

case1 <- read.csv("Case1_400.csv", row.names = 1)
X <- as.matrix(case1[, 1:10])
prep <- art_prepare(X, perplexity = 50)

error_message <- function(code) {
  tryCatch({
    code
    "no error"
  }, error = function(e) conditionMessage(e))
}

X_na <- X
X_na[1, 1] <- NA

X_origin <- X
X_origin[1, ] <- 0

stopifnot(error_message(art_prepare(X_na, perplexity = 50)) == "X has missing or non-finite values")
stopifnot(error_message(art_prepare(X_origin, perplexity = 50)) == "X has points at the origin")
stopifnot(error_message(art_prepare(X, perplexity = 200)) == "perplexity is too large for the number of samples")
stopifnot(error_message(fuse(prep, 1.5)) == "alpha must be between 0 and 1")
