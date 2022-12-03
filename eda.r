train <- read.csv("./mnist/train.csv", header = TRUE)
test <- read.csv("./mnist/test.csv", header = TRUE)

# combine features
x_train <- as.matrix(train[, -1])
x_test <- as.matrix(test)
y_train <- train$label
x <- rbind(x_train, x_test)

counts_0 <- c()
for (i in 0:9) {
  x_train_i <- x_train[y_train == i, ]
  counts_0 <- rbind(counts_0, mean(x_train_i == 0))
}
counts_0 <- t(counts_0)
colnames(counts_0) <- 0:9

png(filename="./img/eda1.png")
barplot(counts_0, xlab="Digit")
dev.off()


digit_mean <- c()
for (i in 0:9) {
  x_train_i <- x_train[y_train == i, ]
  digit_mean <- rbind(digit_mean, mean(x_train_i))
}
digit_mean <- t(digit_mean)
colnames(digit_mean) <- 0:9
