install.packages("torch")
library(torch)
nrow = ncol = 3000
# --- Base R Benchmarks ---
t_create_r <- system.time({
  matrix_r <- matrix(rnorm(nrow * ncol), nrow, ncol)
})[["elapsed"]]

t_create_t <- system.time({
  matrix_t <- torch_randn(nrow, ncol)
})[["elapsed"]]

print(paste("R creation time:", t_create_r, ";torch creation time:", t_create_t))

t_elemwise_r <- system.time({
  elemwise_r <- matrix_r * matrix_r
})[["elapsed"]]

t_elemwise_t <- system.time({
  elemwise_t <- matrix_t * matrix_t
})[["elapsed"]]

print(paste("R Element-wise product time:", t_elemwise_r, ";torch Element-wise product time:", t_elemwise_t))

t_matmul_r <- system.time({
  matmul_r <- matrix_r %*% t(matrix_r)
})[["elapsed"]]

t_matmul_t <- system.time({
  matmul_t <- torch_matmul(matrix_t, matrix_t$t())
})[["elapsed"]]

print(paste("R matrix multiplication time:", t_matmul_r, ";torch matrix multiplication time:", t_matmul_t))

###################---- next session

t_rowsum_r <- system.time({
  row_sums_r <- rowSums(as.array(matrix_r)^2)
})[["elapsed"]]

t_rowsum_t <- system.time({
  row_sums_t <- (matrix_t^2)$sum(dim = 2)
})[["elapsed"]]

t_norm_r <- system.time({
  norm_r <- scale(matrix_r)
})[["elapsed"]]

t_norm_t <- system.time({
  mean_t <- matrix_t$mean(dim = 1)
  std_t <- matrix_t$std(dim = 1)
  norm_t <- (matrix_t - mean_t) / std_t
})[["elapsed"]]

t_colmean_r <- system.time({
  colmean_r <- colMeans(matrix_r)
})[["elapsed"]]

t_colmean_t <- system.time({
  colmean_t <- matrix_t$mean(dim = 1)
})[["elapsed"]]

