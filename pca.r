library(dimRed)

train <- read.csv("./mnist/train.csv", header = TRUE)
test <- read.csv("./mnist/test.csv", header = TRUE)

# combine features
x_train <- as.matrix(train[, -1])
x_test <- as.matrix(test)
y_train <- train$label
X <- rbind(x_train, x_test)

set.seed(1)
subsamp <- X[c(sample(1:35000, 700),
               sample(35001: 70000, 700)), ]

Xcentered <- scale(subsamp, center = T, scale = F)
svd <- svd(Xcentered)
scores <- Xcentered %*% svd$v[,1:2]
trueD <- svd$d

set.seed(1)
scramble <- function(x) x[sample(length(x), length(x))]
xscramble <- apply(Xcentered, 2, scramble)
svdscramble <- svd(xscramble)
scrambleD <- svdscramble$d
plot(log(trueD[1:100]), pch = 20, col = 'red',
     xlab = 'Dimension', ylab = 'Singular Value', sub = 'Figure 2.1')
points(log(scrambleD[1:100]), pch = 18, col = 'blue')

# Do the pca
pcomp <- prcomp(Xcentered)
saveRDS(pcomp, './pcomp.rds')
comps <- pcomp$rotation[,1:37]
compx <- X%*%comps
