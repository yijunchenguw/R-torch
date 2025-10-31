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

#Appendix ----
# Generate a matrix Y with 20 columns and 500 rows (total 10000 values)
# The first 5000 values are sampled from 0:3, the next 5000 from 0:5
set.seed(233)
torch_manual_seed(123)
Y <- matrix(c(sample(c(0:3), 5000, T), sample(c(0:5), 5000, T)), ncol = 20)
N = 500; J = 20; D = 3; K = 5 
a <- torch_rand(J, D)*(0.5)+0.5 # J = 20, D = 3 
b <- torch_rand(J, K) # K = 5
theta_matrix <- torch_randn(N,D)
# Introduce a missing value into Y to simulate real-world data issues
Y[1] <- NA
# Identify which entries in Y are missing (i.e., NA)
Y.na <- is.na(Y)

# Replace all missing entries in Y with 0.5 (neutral placeholder)
# This avoids errors in tensor computations such as broadcasting or masking
Y[Y.na] <- 0.5
Y <- torch_tensor(Y)
K_list <- apply(Y, 2, max)            # Determine max score for each item
K <- max(K_list)                      # Global max score
K_range_max <- torch_arange(0, K, dtype = torch_int())
# K_range_max = [0, 1, ..., K]

K_matrix <- K_range_max$expand(c(N, J, K+1))  
# Expand k range across examinees and items

mask <- K_range_max <= torch_tensor(K_list)$view(c(1, length(K_list), -1))$expand(c(N, J, 1))

K_matrix <- K_matrix * mask           # Apply the mask to remove invalid values
Y_matrix <- Y$unsqueeze(3)$expand(c(N, J, K+1))
# Expand Y to [N, J, K+1] by duplicating scores across all categories

Response_mask <- torch_where(K_range_max == Y_matrix, 1, 0)
# Create binary tensor: 1 where Y matches category k, 0 elsewhere
# Concatenate the column of zeros to the left of b
b_expanded <- torch_cat(list(torch_zeros(b$size(1), 1), b), dim = 2)*mask

xi <- K_matrix * 
  torch_matmul(theta_matrix, a$t())$unsqueeze(3)$expand(c(N, J, K+1)) -
  b_expanded

Xi <- torch_where(mask, exp(xi), 0)

Pr <- (Xi*Response_mask)$sum(3) / (Xi$sum(3))
# Numerator retains only the exponential logits corresponding to the
# observed response (using Response_mask)
# Denominator normalizes by the sum of exponential logits across all k-values
# $sum(3) sums across the k-values dimension


# -----------------------------------------------------------
# Convert torch tensors to base R arrays
# -----------------------------------------------------------
Y_r      <- as.array(Y)
a_r      <- as.array(a)
b_r      <- cbind(0, as.array(b))   # add a zero column for baseline category
theta_r  <- as.array(theta_matrix)

# -----------------------------------------------------------
# Initialize probability holder
# -----------------------------------------------------------
Pr_r <- array(0, dim = c(N, J))     # N x J matrix of probabilities

# -----------------------------------------------------------
# Compute response probabilities for each person-item pair
# -----------------------------------------------------------
for (i in 1:N) {
  for (j in 1:J) {
    
    # Linear predictor: discrimination × ability
    q <- a_r[j, ] %*% theta_r[i, ]
    
    # Observed response category index (1-based)
    kij <- Y_r[i, j] + 1
    
    # Initialize numerator and denominator
    denom <- 0
    num <- 0
    
    # Iterate over all possible response categories
    for (k in 1:(K_list[j] + 1)) {
      dw <- (k - kij) * q
      db <- b_r[j, k]
      denom <- denom + exp(dw - db)
      
      # When k matches the observed response, store numerator
      if (kij == k) {
        num <- exp(dw - db)
      }
    }
    
    # Compute probability for person i, item j
    Pr_r[i, j] <- num / denom
  }
}

# -----------------------------------------------------------
# Handle missing responses (represented as 0.5 here)
# -----------------------------------------------------------
Pr_r[as.array(Y) == 0.5] <- 0

# Monte Carlo ----

N <- 1e8L
p <- torch_rand(N, 2) * 2 - 1
n <- sum((p ^ 2)$sum(2) <= 1)
n / N * 4


#Case Study 3 ----
torch_manual_seed(123)
N <- 1000
x <- torch_rand(N, 1) * 4 - 2
y <- exp(x ^ 3 - x ^ 2)$clip(max = 2) + torch_normal(0, 0.2, x$shape)

#Model Architecture and Loss
model <- nn_sequential(
  nn_linear(1, 16),
  nn_relu(),
  nn_linear(16, 16),
  nn_relu(),
  nn_linear(16, 1)
)
model(1)
# torch_tensor
#  0.2075
# [ CPUFloatType{1} ][ grad_fn = <ViewBackward0> ]
nnf_mse_loss(torch_tensor(c(0, 1)), torch_tensor(c(1, -2)))
# torch_tensor
# 5
# [ CPUFloatType{} ]
opt <- optim_adam(model$parameters, 0.1)
for (i in 1:1000) {
  y_pred <- model(x)
  loss <- nnf_mse_loss(y_pred, y)
  opt$zero_grad()
  loss$backward()
  opt$step()
  if (i %% 200 == 0)
    cat(sprintf("Epoch %d: Loss = %.4f\n", i, loss$item()))
}
# Epoch 200: Loss = 0.0503
# Epoch 400: Loss = 0.0404
# Epoch 600: Loss = 0.0398
# Epoch 800: Loss = 0.0399
# Epoch 1000: Loss = 0.0399
