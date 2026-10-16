library(ART)


# Data: 2031 cells from Kang et al. (2018), one row per cell and one column per gene
file <- "cells1488.rds"
if (!file.exists(file))
  download.file("https://github.com/linluzhao/ART/releases/download/v0.1.0/cells1488.rds", file, mode = "wb")

cells  <- readRDS(file)
labels <- cells$cell_type
X      <- as.matrix(cells[, colnames(cells) != "cell_type"])


# Five alphas, about one minute each on a laptop
alphas <- c(0, 0.25, 0.5, 0.75, 1)


# ART
res <- art_path(X, alphas = alphas, perplexity = 50, seed = 1)


# Figure
types  <- c("CD4 T cells", "CD8 T cells", "B cells", "NK cells",
            "CD14+ Monocytes", "FCGR3A+ Monocytes", "Dendritic cells", "Megakaryocytes")
colors <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f")

png("cells_art.png", width = 3000, height = 800, res = 150)
layout(matrix(c(1:5, rep(6, 5)), nrow = 2, byrow = TRUE), heights = c(5, 1))
par(mar = c(1, 1, 3, 1))

for (k in seq_along(alphas)) {
  plot(res$frames[[k]], col = colors[match(labels, types)], pch = 16, cex = 0.6, asp = 1,
       axes = FALSE, xlab = "", ylab = "", main = bquote("ART: " * alpha == .(alphas[k])))
  box(col = "grey60")
}

plot.new()
legend("center", legend = types, col = colors, pch = 16, pt.cex = 1.5, horiz = TRUE, bty = "n")
invisible(dev.off())
