library(AppliedPredictiveModeling)
library(caret)
library(corrplot)
library(glmnet)
library(MASS)
library(modeldata)
library(pamr)
library(pls)
library(pROC)
library(RColorBrewer)
library(rms)
library(sparseLDA)
library(subselect)
library(tidyverse)

####12.1####

data(hepatic)

#a) I would create a training and testing set by splitting the data using
#stratified random sampling and using injury as the factor

#b) I would optimize using.... specificity or sensitivity etc

####12.1c####

fullSet <- bio
fullSet[,185:377] <- chem
fullSet[, 378] <- injury
#injury is named V378

redSet <- predict(preProcess(fullSet, method=c("nzv", "corr")), fullSet)

trainingRows <- createDataPartition(fullSet[,378], p=0.80, list=FALSE)

fullTraining <- fullSet[trainingRows,]
fulTtesting <- fullSet[-trainingRows,]

trainingRows <- createDataPartition(redSet[,238], p=0.80, list=FALSE)

redTraining <- redSet[trainingRows,]
redTesting <- redSet[-trainingRows,]

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