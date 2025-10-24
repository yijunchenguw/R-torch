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


