library(AppliedPredictiveModeling)
library(caret)
library(glmnet)
library(MASS)
library(pROC)
library(tidyverse)

####12.1####

data(hepatic)

#a) I would create a training and testing set by splitting the data using
#stratified random sampling and using injury as the factor

#b) I would optimize using a ROC curve since it balances sensitivity and specificity
#to find the position where both values are maximized.

####12.1c####

fullSet <- bio
fullSet[,185:377] <- chem
fullSet[, 378] <- injury
#injury is named V378

redSet <- predict(preProcess(fullSet, method=c("nzv", "corr")), fullSet)

trainingRows <- createDataPartition(fullSet[,378], p=0.80, list=FALSE)

fullTraining <- fullSet[trainingRows,]
fullTesting <- fullSet[-trainingRows,]

trainingRows <- createDataPartition(redSet[,238], p=0.80, list=FALSE)

redTraining <- redSet[trainingRows,]
redTesting <- redSet[-trainingRows,]

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
summary(lrFit)

confusionMatrix(data = lrFit$pred$pred,
                reference = lrFit$pred$obs)

lrRoc <- roc(response = lrFit$pred$obs,
               predictor = lrFit$pred$None,
               levels = rev(levels(lrFit$pred$obs)))
plot(lrRoc, legacy.axes = TRUE)
auc(lrRoc)

#Linear Discriminant Analysis

ldaFit <- train(redTraining[,1:237],
                 y = redTraining$V378,
                 method = "lda",
                 metric = "ROC",
                 preProc = c("center", "scale"),
                 trControl = ctrl)

ldaFit
summary(ldaFit)

confusionMatrix(data = ldaFit$pred$pred,
                reference = ldaFit$pred$obs)

#also don't work
ldaRoc <- roc(response = ldaFit$pred$obs,
               predictor = ldaFit$pred$None,
               levels = rev(levels(ldaFit$pred$obs)))
plot(ldaRoc, legacy.axes = TRUE)
auc(ldaRoc)

#Partial Least Squares Discriminant Analysis



#Penalized Models

#Nearest Shrunken Centroids

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