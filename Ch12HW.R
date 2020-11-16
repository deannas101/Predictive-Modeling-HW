library(AppliedPredictiveModeling)
library(caret)
library(glmnet)
library(MASS)
library(modeldata)
library(pROC)
library(tidyverse)

####12.1####

set.seed(80)

data(hepatic)

#a) I would create a training and testing set by splitting the data using
#stratified random sampling and using injury as the factor

#b) I would optimize using kappa since we have more than two levels of classes,
#which makes using ROC ineffective.

####12.1c####

fullSet <- bio
fullSet[, 185] <- injury
#injury is named V378

trainingRows <- createDataPartition(fullSet[,185], p=0.80, list=FALSE)

fullTraining <- fullSet[trainingRows,]
fullTesting <- fullSet[-trainingRows,]

#Logistic Regression

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = defaultSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

lrFit <- train(fullTraining[,1:184],
                y = fullTraining$V378,
                method = "multinom",
                metric = "Accuracy",
                preProc = c("knnImpute", "nzv"),
                trControl = ctrl)

lrFit
confusionMatrix(data = lrFit$pred$pred,
                reference = lrFit$pred$obs)
plot(lrFit)

#Linear Discriminant Analysis

ldaFit <- train(fullTraining[,1:184],
                y = fullTraining$V378,
                 method = "lda",
                 metric = "Kappa",
                 preProc = c("center", "scale", "nzv", "corr"),
                 trControl = ctrl)

ldaFit
confusionMatrix(data = ldaFit$pred$pred,
                reference = ldaFit$pred$obs)
plot(ldaFit)

#Partial Least Squares Discriminant Analysis

ctrl <- trainControl(summaryFunction = defaultSummary,
                     classProbs = TRUE)

plsFit <- train(fullTraining[,1:184],
                y = fullTraining$V378,
                 method = "pls",
                 tuneGrid = expand.grid(.ncomp = 1:10),
                 preProc = c("center","scale", "nzv", "corr"),
                 metric = "Kappa",
                 maxit = 2000,
                 trControl = ctrl)

plsFit

#not working
confusionMatrix(data = plsFit$pred$pred,
                reference = plsFit$pred$obs)
plot(plsFit)

#Penalized Models

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = defaultSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))

glmnFit <- train(fullTraining[,1:184],
                   y = fullTraining$V378,
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   preProc = c("center", "scale", "nzv"),
                   metric = "Kappa",
                   trControl = ctrl)

glmnFit
confusionMatrix(data = glmnFit$pred$pred,
                reference = glmnFit$pred$obs)
plot(glmnFit)

#Nearest Shrunken Centroids

ctrl <- trainControl(summaryFunction = defaultSummary,
                     classProbs = TRUE)

nscGrid <- data.frame(.threshold = seq(0,4, by=0.1))

nscFit <- train(fullTraining[,1:184],
                y = fullTraining$V378,
                  method = "pam",
                  preProc = c("center", "scale", "nzv"),
                  tuneGrid = nscGrid,
                  metric = "Kappa",
                  trControl = ctrl)

nscFit
#not working
confusionMatrix(data = nscFit$pred$pred,
                reference = nscFit$pred$obs)
plot(nscFit)

####12.1d####
#edit me! -> highest kappa value
nnImp <- varImp(nnFit, scale = FALSE)
nnImp
plot(nnImp, top = 5, main = "Neural Network")

####12.3####

set.seed(80)

data(mlc_churn)

mlc_churn <- as.data.frame(mlc_churn)

####12.3a####

#outcome: churn, yes/no

pairs(mlc_churn)

#12.3b: ROC should be used since the outcome has two classes.

####12.3c####

trainingRows <- createDataPartition(mlc_churn$churn, p=0.80, list=FALSE)

fullTraining <- mlc_churn[trainingRows,]
fullTesting <- mlc_churn[-trainingRows,]

#Logistic Regression

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

lrFit <- train(fullTraining[,1:19],
                y = fullTraining$churn,
                method = "glm",
                metric = "ROC",
                trControl = ctrl)

confusionMatrix(data = lrFit$pred$pred,
                reference = lrFit$pred$obs)

lrRoc <- roc(response = lrFit$pred$obs,
               predictor = lrFit$pred$yes,
               levels = rev(levels(lrFit$pred$obs)))
plot(lrRoc, legacy.axes = TRUE)
auc(lrRoc)

#Linear Discriminant Analysis

#not working
ldaFit <- train(x = fullTraining[,1:19],
                y = fullTraining$churn,
                 method = "lda",
                 metric = "ROC",
                 trControl = ctrl)

confusionMatrix(data = ldaFit$pred$pred,
                reference = ldaFit$pred$obs)

ldaRoc <- roc(response = ldaFit$pred$obs,
             predictor = ldaFit$pred$yes,
             levels = rev(levels(ldaFit$pred$obs)))
plot(ldaRoc, legacy.axes = TRUE)
auc(ldaRoc)

#Partial Least Squares

ctrl <- trainControl(summaryFunction = twoClassSummary,
                     classProbs = TRUE)

#not working
plsFit <- train(fullTraining[,1:19],
                y = fullTraining$churn,
                 method = "pls",
                 tuneGrid = expand.grid(.ncomp = 1:10),
                 preProc = c("center","scale"),
                 metric = "ROC",
                 trControl = ctrl)

confusionMatrix(data = plsFit$pred$pred,
                reference = plsFit$pred$obs)

plsRoc <- roc(response = plsFit$pred$obs,
              predictor = plsFit$pred$yes,
              levels = rev(levels(plsFit$pred$obs)))
plot(plsRoc, legacy.axes = TRUE)
auc(plsRoc)

#Penalized Models

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))

#not working
glmnFit <- train(fullTraining[,1:19],
                 y = fullTraining$churn,
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   preProc = c("center", "scale"),
                   metric = "ROC",
                   trControl = ctrl)

confusionMatrix(data = glmnFit$pred$pred,
                reference = glmnFit$pred$obs)

glmnRoc <- roc(response = glmnFit$pred$obs,
              predictor = glmnFit$pred$yes,
              levels = rev(levels(glmnFit$pred$obs)))
plot(glmnRoc, legacy.axes = TRUE)
auc(glmnRoc)

#Nearest Shrunken Centroids

ctrl <- trainControl(summaryFunction = twoClassSummary,
                     classProbs = TRUE)

nscGrid <- data.frame(.threshold = seq(0,4, by=0.1))

nscFit <- train(x = fullTraining[,1:19],
                y = fullTraining$churn,
                  method = "pam",
                  preProc = c("center", "scale"),
                  tuneGrid = nscGrid,
                  metric = "ROC",
                  trControl = ctrl)

confusionMatrix(data = nscFit$pred$pred,
                reference = nscFit$pred$obs)

nscRoc <- roc(response = nscFit$pred$obs,
               predictor = nscFit$pred$yes,
               levels = rev(levels(nscFit$pred$obs)))
plot(nscRoc, legacy.axes = TRUE)
auc(nscRoc)