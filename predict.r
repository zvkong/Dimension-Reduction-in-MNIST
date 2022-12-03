## require(devtools)
## install_version("lle", version = "1.1", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(Matrix)
library(xgboost)

## load labels
train_data <- read.csv("./mnist/train.csv")

## load lower dimension representations
## embedding_le <- readRDS("./embedding_le.rds")
embedding_autoencoder <- read.csv("./output/mnist_autoencoder.csv")

## resplit train/test
train_x_low_data <- embedding_autoencoder[1:42000, ]
test_x_low_data <- embedding_autoencoder[42001:70000, ]
train_x <- as.matrix(train_x_low_data)
train_y <- train_data$label
test_x <- as.matrix(test_x_low_data)

## fit xgboost
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
