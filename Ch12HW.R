library(AppliedPredictiveModeling)
library(caret)
library(glmnet)
library(MASS)
library(pROC)
library(tidyverse)

set.seed(80)

####12.1####

data(hepatic)

#a) I would create a training and testing set by splitting the data using
#stratified random sampling and using injury as the factor

#b) I would optimize using kappa since we have more than two levels of classes,
#which makes using ROC ineffective.

####12.1c####

fullSet <- bio
fullSet[,185:377] <- chem
fullSet[, 378] <- injury
#injury is named V378

trainingRows <- createDataPartition(fullSet[,378], p=0.80, list=FALSE)

fullTraining <- fullSet[trainingRows,]
fullTesting <- fullSet[-trainingRows,]

#Logistic Regression

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = defaultSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

lrFit <- train(fullTraining[,1:377],
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

ldaFit <- train(fullTraining[,1:377],
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

plsFit <- train(fullTraining[,1:377],
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

glmnFit <- train(fullTraining[,1:377],
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

nscFit <- train(fullTraining[,1:377],
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
#edit me!
nnImp <- varImp(nnFit, scale = FALSE)
nnImp
plot(nnImp, top = 20, main = "Neural Network")

####12.3####

data(mlc_churn)

####12.3a####

#outcome: churn, yes/no

#pairwise plot
#correlation plot

pairs(cleanChurn)

####12.3c####

trainingRows <- createDataPartition(mlc_churn$churn, p=0.80, list=FALSE)

fullTraining <- mlc_churn[trainingRows,]
fulTtesting <- mlc_churn[-trainingRows,]