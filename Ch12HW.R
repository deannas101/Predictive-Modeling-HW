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
                y = fullTraining$V185,
                method = "multinom",
                metric = "Accuracy",
                preProc = c("knnImpute", "nzv"),
                trControl = ctrl)

lrFit
confusionMatrix(data = lrFit$pred$pred,
                reference = lrFit$pred$obs)
plot(lrFit)

predicted <- predict(lrFit, fullTesting[,1:184])
lrValues <- data.frame(obs = fullTesting[,185], pred = predicted)

defaultSummary(lrValues)

#Linear Discriminant Analysis

ldaFit <- train(fullTraining[,1:184],
                y = fullTraining$V185,
                 method = "lda",
                 metric = "Kappa",
                 preProc = c("center", "scale", "nzv", "corr"),
                 trControl = ctrl)

ldaFit
confusionMatrix(data = ldaFit$pred$pred,
                reference = ldaFit$pred$obs)
plot(ldaFit) #no tuning parameters

predicted <- predict(ldaFit, fullTesting[,1:184])
ldaValues <- data.frame(obs = fullTesting[,185], pred = predicted)

defaultSummary(ldaValues)

#Partial Least Squares Discriminant Analysis

ctrl <- trainControl(summaryFunction = defaultSummary,
                     classProbs = TRUE)

plsFit <- train(fullTraining[,1:184],
                y = fullTraining$V185,
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

predicted <- predict(plsFit, fullTesting[,1:184])
plsValues <- data.frame(obs = fullTesting[,185], pred = predicted)

defaultSummary(plsValues)

#Penalized Models

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = defaultSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))

glmnFit <- train(fullTraining[,1:184],
                   y = fullTraining$V185,
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   preProc = c("center", "scale", "nzv"),
                   metric = "Kappa",
                   trControl = ctrl)

glmnFit
confusionMatrix(data = glmnFit$pred$pred,
                reference = glmnFit$pred$obs)
plot(glmnFit)

predicted <- predict(glmnFit, fullTesting[,1:184])
glmnValues <- data.frame(obs = fullTesting[,185], pred = predicted)

defaultSummary(glmnValues)

#Nearest Shrunken Centroids

ctrl <- trainControl(summaryFunction = defaultSummary,
                     classProbs = TRUE)

nscGrid <- data.frame(.threshold = seq(0,4, by=0.1))

nscFit <- train(fullTraining[,1:184],
                y = fullTraining$V185,
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

predicted <- predict(nscFit, fullTesting[,1:184])
nscValues <- data.frame(obs = fullTesting[,185], pred = predicted)

defaultSummary(nscValues)

####12.1d####
#highest kappa value
plsImp <- varImp(plsFit, scale = FALSE)
plsImp
plot(plsImp, top = 5, main = "Partial Least Squares")

####12.3####

set.seed(80)

data(mlc_churn)

mlc_churn <- as.data.frame(mlc_churn)

####12.3a####

#outcome: churn, yes/no

pairs(mlc_churn)

#12.3b: ROC should be used since the outcome has two classes.

####12.3c####

dummRes <-dummyVars("~state+area_code+international_plan+voice_mail_plan",  data=mlc_churn, fullRank=TRUE)
Add_dumm <- data.frame(predict(dummRes, newdata=mlc_churn))

full_churn <- Add_dumm
full_churn[,54:68] <- mlc_churn[,6:20]

trainingRows <- createDataPartition(full_churn$churn, p=0.80, list=FALSE)

fullTraining <- full_churn[trainingRows,]
fullTesting <- full_churn[-trainingRows,]

#Logistic Regression

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

lrFit <- train(fullTraining[,1:67],
                y = fullTraining$churn,
                method = "glm",
                metric = "ROC",
                trControl = ctrl)

lrFit
confusionMatrix(data = lrFit$pred$pred,
                reference = lrFit$pred$obs)

lrRoc <- roc(response = lrFit$pred$obs,
               predictor = lrFit$pred$yes,
               levels = rev(levels(lrFit$pred$obs)))
plot(lrRoc, legacy.axes = TRUE)
auc(lrRoc)

predicted <- predict(lrFit, fullTesting[,1:67])
lrValues <- data.frame(obs = fullTesting[,68], pred = predicted)

defaultSummary(lrValues)

#Linear Discriminant Analysis

ldaFit <- train(x = fullTraining[,1:67],
                y = fullTraining$churn,
                 method = "lda",
                 metric = "ROC",
                 trControl = ctrl)

ldaFit
confusionMatrix(data = ldaFit$pred$pred,
                reference = ldaFit$pred$obs)

ldaRoc <- roc(response = ldaFit$pred$obs,
             predictor = ldaFit$pred$yes,
             levels = rev(levels(ldaFit$pred$obs)))
plot(ldaRoc, legacy.axes = TRUE)
auc(ldaRoc)

predicted <- predict(ldaFit, fullTesting[,1:67])
ldaValues <- data.frame(obs = fullTesting[,68], pred = predicted)

defaultSummary(ldaValues)

#Partial Least Squares

ctrl <- trainControl(summaryFunction = twoClassSummary,
                     classProbs = TRUE)

plsFit <- train(fullTraining[,1:67],
                y = fullTraining$churn,
                 method = "pls",
                 tuneGrid = expand.grid(.ncomp = 1:10),
                 preProc = c("center","scale"),
                 metric = "ROC",
                 trControl = ctrl)

plsFit #not working
confusionMatrix(data = plsFit$pred$pred,
                reference = plsFit$pred$obs)

plsRoc <- roc(response = plsFit$pred$obs,
              predictor = plsFit$pred$yes,
              levels = rev(levels(plsFit$pred$obs)))
plot(plsRoc, legacy.axes = TRUE)
auc(plsRoc)

predicted <- predict(plsFit, fullTesting[,1:67])
plsValues <- data.frame(obs = fullTesting[,68], pred = predicted)

defaultSummary(plsValues)

#Penalized Models

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))

glmnFit <- train(fullTraining[,1:67],
                 y = fullTraining$churn,
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   preProc = c("center", "scale"),
                   metric = "ROC",
                   trControl = ctrl)

glmnFit
confusionMatrix(data = glmnFit$pred$pred,
                reference = glmnFit$pred$obs)

glmnRoc <- roc(response = glmnFit$pred$obs,
              predictor = glmnFit$pred$yes,
              levels = rev(levels(glmnFit$pred$obs)))
plot(glmnRoc, legacy.axes = TRUE)
auc(glmnRoc)

predicted <- predict(glmnFit, fullTesting[,1:67])
glmnValues <- data.frame(obs = fullTesting[,68], pred = predicted)

defaultSummary(glmnValues)

#Nearest Shrunken Centroids

ctrl <- trainControl(summaryFunction = twoClassSummary,
                     classProbs = TRUE)

nscGrid <- data.frame(.threshold = seq(0,4, by=0.1))

nscFit <- train(x = fullTraining[,1:67],
                y = fullTraining$churn,
                  method = "pam",
                  preProc = c("center", "scale"),
                  tuneGrid = nscGrid,
                  metric = "ROC",
                  trControl = ctrl)

nscFit #not working
confusionMatrix(data = nscFit$pred$pred,
                reference = nscFit$pred$obs)

nscRoc <- roc(response = nscFit$pred$obs,
               predictor = nscFit$pred$yes,
               levels = rev(levels(nscFit$pred$obs)))
plot(nscRoc, legacy.axes = TRUE)
auc(nscRoc)

predicted <- predict(nscFit, fullTesting[,1:67])
nscValues <- data.frame(obs = fullTesting[,68], pred = predicted)

defaultSummary(nscValues)
