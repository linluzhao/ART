# Direction and magnitude
art_decompose <- function(X) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  if (any(!is.finite(X))) stop("X has missing or non-finite values")
  r <- sqrt(rowSums(X^2))
  if (any(r == 0)) stop(sum(r == 0), " points are at the origin")
  list(u = X / r, r = r)
}

# Random start
shared_start <- function(n_points, seed = 1L) {
  had_seed <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
  on.exit(if (had_seed) assign(".Random.seed", old_seed, envir = globalenv())
          else rm(".Random.seed", envir = globalenv()))
  set.seed(seed)
  matrix(rnorm(2 * n_points), ncol = 2) * 1e-4
}

# Angular and radial kernels
art_prepare <- function(X, perplexity = 30, seed = 1L) {
  parts <- art_decompose(X)
  N     <- length(parts$r)
  if (N - 1 < 3 * perplexity) stop("perplexity is too large for the number of samples")
  D_ang <- art_squared_distance(parts$u)
  D_rad <- art_squared_distance(matrix(parts$r))
  list(D_ang = D_ang, D_rad = D_rad,
       beta_ang = art_search_beta(D_ang, perplexity),
       beta_rad = art_search_beta(D_rad, perplexity),
       perplexity = perplexity, start = shared_start(N, seed))
}

# Combine the two kernels
fuse <- function(prep, alpha, recalibrate = TRUE) {
  if (alpha < 0 || alpha > 1) stop("alpha must be between 0 and 1")
  art_fuse(prep$D_ang, prep$D_rad, prep$beta_ang, prep$beta_rad, alpha, prep$perplexity, recalibrate)
}

# Alignment
procrustes_onto <- function(Y, reference) {
  Y         <- sweep(Y, 2, colMeans(Y), "-")
  reference <- sweep(reference, 2, colMeans(reference), "-")
  s <- svd(crossprod(Y, reference))
  Y %*% (s$u %*% t(s$v))
}

align_to_first <- function(frames) {
  first <- sweep(frames[[1]], 2, colMeans(frames[[1]]), "-")
  c(list(first), lapply(frames[-1], procrustes_onto, reference = first))
}

# One alpha
art_frame <- function(prep, alpha, recalibrate = TRUE, ...) {
  fused  <- fuse(prep, alpha, recalibrate)
  layout <- art_layout(fused$P, prep$start, ...)
  list(Y = layout$Y, perplexity = fused$perplexity, kl = layout$kl)
}

# All alphas
art_path <- function(X, alphas = seq(0, 1, by = 0.02), perplexity = 30, seed = 1L,
                     recalibrate = TRUE, ...) {
  prep <- art_prepare(X, perplexity, seed)
  runs <- lapply(alphas, function(a) art_frame(prep, a, recalibrate, ...))
  raw  <- lapply(runs, `[[`, "Y")

  diagnostics <- data.frame(
    alpha           = alphas,
    mean_perplexity = vapply(runs, function(x) mean(x$perplexity), 0),
    min_perplexity  = vapply(runs, function(x) min(x$perplexity), 0),
    max_perplexity  = vapply(runs, function(x) max(x$perplexity), 0),
    kl              = vapply(runs, function(x) x$kl[length(x$kl)], 0))

  invisible(list(frames = align_to_first(raw), unaligned = raw, diagnostics = diagnostics))
}
