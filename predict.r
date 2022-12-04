## require(devtools)
## install_version("lle", version = "1.1", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(Matrix)
library(xgboost)

## load labels
train_data <- read.csv("./mnist/train.csv")

## load lower dimension representations
## train_x <- as.matrix(train_data[, -1])
embedding_autoencoder <- read.csv("./output/mnist_autoencoder.csv")

## resplit train/test
train_x_low_data <- embedding_autoencoder[1:42000, ]
test_x_low_data <- embedding_autoencoder[42001:70000, ]
train_x <- as.matrix(train_x_low_data)
train_y <- train_data$label
test_x <- as.matrix(test_x_low_data)

## split train into train_new and validation
train_idx <- sample(seq_len(nrow(train_x)), round(nrow(train_x) * 0.8))
val_idx  <- (seq_len(nrow(train_x)))[-train_idx]
train_train_x <- as.matrix(train_x[train_idx, ])
train_train_y <- train_y[train_idx]
train_val_x <- as.matrix(train_x[val_idx, ])
train_val_y <- train_y[val_idx]

## parameter tuning
mnist_model_tune <-
  xgboost(
    data = train_train_x,
    label = train_train_y,
    nrounds = 2000,
    objective = "multi:softmax",
    num_class = 10,
    early_stopping_rounds = 5,
    max_depth = 2,
    eta = 1)
val_y_hat <- predict(mnist_model_tune, train_val_x)
mean(val_y_hat == train_val_y)

## fit xgboost on entire available data
set.seed(1)
start_time <- Sys.time()
mnist_model <-
  xgboost(
    data = train_x,
    label = train_y,
    nrounds = 2000,
    objective = "multi:softmax",
    num_class = 10,
    early_stopping_rounds = 5,
    max_depth = 2,
    eta = 1)
end_time <- Sys.time()
time_diff <- end_time - start_time
y_hat <- predict(mnist_model, test_x)
submission <- data.frame(ImageId = seq_len(nrow(test_x)), Label = y_hat)
write.csv(submission, "./submission.csv", row.names = FALSE)
