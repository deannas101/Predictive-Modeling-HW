library(AppliedPredictiveModeling)
library(caret)
library(glmnet)
library(kernlab)
library(klaR)
library(MASS)
library(tidyverse)

# 13.1 ------------------------------------------------------------------------

data(hepatic)

# creating dataset
full_dataset <- bio
full_dataset[, 185] <- injury
# injury is named V185


# splitting data
set.seed(80)
training_rows <- createDataPartition(full_dataset[, 185], p = 0.80, list = FALSE)

training_set <- full_dataset[training_rows, ]
testing_set <- full_dataset[-training_rows, ]

# will determine the best model using kappa since there are 3 classes

ctrl <- trainControl(
  summaryFunction = defaultSummary,
  classProbs = TRUE,
  savePredictions = TRUE
)

# Quadratic Discriminant Analysis - n>p, so this model is not applicable

# Regularized Discriminant Analysis
set.seed(80)
rda_model <- train(
  x = training_set[, 1:184],
  y = training_set$V185,
  method = "rda",
  metric = "Kappa",
  tuneGrid = expand.grid(.lambda = 1:5, .gamma = 1:5),
  trControl = ctrl
)

rda_model
plot(rda_model, main = "Regularized Discriminant Analysis")
rda_pred <- predict(rda_model, newdata = testing_set[, 1:184])
confusionMatrix(data = rda_pred, reference = testing_set$V185)

# Mixture Discriminant Analysis
set.seed(80)
mda_model <- train(
  x = training_set[, 1:184],
  y = training_set$V185,
  method = "mda",
  metric = "Kappa",
  preProc = c("knnImpute", "nzv"),
  tuneGrid = expand.grid(.subclasses = 1:5),
  trControl = ctrl
)

mda_model
plot(mda_model, main = "Mixture Discriminant Analysis")
mda_pred <- predict(mda_model, newdata = testing_set[, 1:184])
confusionMatrix(data = mda_pred, reference = testing_set$V185)

# Neural Networks
nnet_grid <- expand.grid(.size = 1:20, .decay = c(0, .1, 1, 2))
max_size <- max(nnet_grid$.size)
num_weights <- (max_size * (184 + 1) + (max_size + 1) * 3) # 184 predictors, 3 classes

set.seed(80)
nnet_model <- train(
  x = training_set[, 1:184],
  y = training_set$V185,
  method = "nnet",
  metric = "Kappa",
  preProc = c("center", "scale", "spatialSign", "nzv"),
  tuneGrid = nnet_grid,
  trace = FALSE,
  maxit = 2000,
  MaxNWts = num_weights,
  trControl = ctrl
)

nnet_model
plot(nnet_model, main = "Neural Networks")
nnet_pred <- predict(nnet_model, newdata = testing_set[, 1:184])
confusionMatrix(data = nnet_pred, reference = testing_set$V185)

# Flexible Discriminant Analysis - n>p, so this model is not applicable

# Support Vector Machine (Radial)
sigma_reduced_range <- sigest(as.matrix(training_set[, 1:184]))
svm_reduced_grid <- expand.grid(
  .sigma = sigma_reduced_range[1],
  .C = 2^(seq(-4, 6))
)

set.seed(80)
svm_model <- train(
  x = training_set[, 1:184],
  y = training_set$V185,
  method = "svmRadial",
  metric = "Kappa",
  tuneGrid = svm_reduced_grid,
  fir = FALSE,
  trControl = ctrl
)

svm_model
plot(svm_model, main = "Support Vector Machines (Radial)")
svm_pred <- predict(svm_model, newdata = testing_set[, 1:184])
confusionMatrix(data = svm_pred, reference = testing_set$V185)

# K-Nearest Neighbors
set.seed(80)
knn_model <- train(
  x = training_set[, 1:184],
  y = training_set$V185,
  method = "knn",
  metric = "Kappa",
  preProc = c("center", "scale", "nzv"),
  tuneGrid = data.frame(.k = 1:50),
  trControl = ctrl
)

knn_model
plot(knn_model, main = "K-Nearest Neighbors")
knn_pred <- predict(knn_model, newdata = testing_set[, 1:184])
confusionMatrix(data = knn_pred, reference = testing_set$V185)

# Naive Bayes
set.seed(80)
nb_model <- train(
  x = training_set[, 1:184],
  y = training_set$V185,
  method = "nb",
  metric = "Kappa",
  preProc = c("center", "scale", "nzv"),
  tuneGrid = data.frame(.fL = c(0, .1, 1, 2), .usekernel = TRUE, .adjust = TRUE),
  trControl = ctrl
)

nb_model
plot(nb_model, main = "Naive Bayes")
nb_pred <- predict(nb_model, newdata = testing_set[, 1:184])
confusionMatrix(data = nb_pred, reference = testing_set$V185)

# variable importance
imp_vars <- varImp(knn_model, scale = FALSE)
plot(imp_vars, top = 5, main = "K-Nearest Neighbors")
