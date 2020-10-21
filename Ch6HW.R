library(AppliedPredictiveModeling)
library(caret)
library(e1071)
library(MASS)
library(pls)
library(tidyverse)

####6.1####

data(tecator)
?tecator

absorp <- as.data.frame(absorp)
endpoints <- as.data.frame(endpoints)
#absorp - absorbance data
#endpoints - percentages of water, fat, protein (cols 1-3)

pca_absorp <- predict(preProcess(absorp, method = c("center", "scale", "pca")), absorp) #default cutoff is 95%

pca_prePlot <- prcomp(absorp, scale = TRUE, center = TRUE)
screeplot(pca_prePlot, type = c("lines"))

#dimensions after pca: 215 x 2

preSplit <- data.frame(PCA = pca_absorp[,1], Fat = endpoints[,2])
preSplit <- prcomp(preSplit, scale = TRUE, center = TRUE)
trainingRows <- createDataPartition(preSplit[,1], p=0.80, list=FALSE)

training <- preSplit[trainingRows,]
testing <- preSplit[-trainingRows,]

ctrl <- trainControl(method = "cv", number = 3)
lmFit1 <- train(Fat ~ PCA, data = training, method = "lm", trControl = ctrl)
lmFit1

#an attempt to make errors go AWAY
PCA <- testing[,1]

lmPred1 <- predict(lmFit1, testing[,1])
lmValues1 <- data.frame(obs = testing$Fat, pred = lmPred1)
defaultSummary(lmValues1)

####6.2####

data(permeability)
?permeability

#fingerprints - matrix of binary fingerprint indicator variables
#permeability - permeability values for each compound

remove <- nearZeroVar(fingerprints)
fingers <- fingerprints[,-remove]
#388 predictors left for modeling

