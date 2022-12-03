library(dimRed)

train <- read.csv("./mnist/train.csv", header = TRUE)
test <- read.csv("./mnist/test.csv", header = TRUE)

# combine features
x_train <- as.matrix(train[, -1])
x_test <- as.matrix(test)
y_train <- train$label
x <- rbind(x_train, x_test)

## fit Laplacian Eigenmaps
start = Sys.time()
embedding_le <- dimRed::embed(X, "LaplacianEigenmaps")
end = Sys.time()
time_diff = end - start
write.csv(time_diff, "le_time.csv")
saveRDS(embedding_le, "embedding_le.rds")

# plot low-dim train data with labels
## embedding_le <- readRDS("./embedding_le.rds")
embedding_le_train <- embedding_le@data@data[1:42000, ]
png(filename="./img/laplacian_eigenmaps.png")
plot(embedding_le_train, col = y_train)
dev.off()
