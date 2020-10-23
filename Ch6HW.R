library(AppliedPredictiveModeling)
library(caret)
library(e1071)
library(elasticnet)
library(glmnet)
library(MASS)
library(pls)
library(RANN)
library(tidyverse)

####6.1####

set.seed(80)

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

endpoints <- predict(preProcess(endpoints, method = c("center", "scale")), endpoints)

preSplit <- data.frame(PCA = pca_absorp[,1], Fat = endpoints[,2])
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

set.seed(80)

data(permeability)
?permeability

fingerprints <- as.data.frame(fingerprints)
#fingerprints - matrix of binary fingerprint indicator variables
#permeability - permeability values for each compound

remove <- nearZeroVar(fingerprints)
fingers <- fingerprints[,-remove]
#388 predictors left for modeling

fingers$permeability <- permeability
trainingRows <- createDataPartition(fingers[,1], p=0.80, list=FALSE)

training <- fingers[trainingRows,]
testing <- fingers[-trainingRows,]

ctrl <- trainControl(method = "cv", number = 10)
plsTune <- train(permeability ~ ., data = training, method = "pls", tuneLength = 20, 
                 trControl = ctrl, preProc = c("center", "scale"))
plsTune
plot(plsTune)

xTest <- testing[,1:388]

predicted <- predict(plsTune, xTest)
lmValues2 <- data.frame(obs = testing[,389], pred = predicted)
colnames(lmValues2) <- c("obs", "pred")

defaultSummary(lmValues2)

####6.3####

set.seed(80)

data(ChemicalManufacturingProcess)
?ChemicalManufacturingProcess

#yield is outcome

#seeing missing values
image(is.na(ChemicalManufacturingProcess), main = "Missing Values", xlab = "Observation", ylab = "Variable", xaxt = "n", yaxt = "n", bty = "n")
axis(1, seq(0, 1, length.out = nrow(ChemicalManufacturingProcess)), 1:nrow(ChemicalManufacturingProcess), col = "white")

#imputing missing values
imputedManufacturing <- predict(preProcess(ChemicalManufacturingProcess, "knnImpute"), ChemicalManufacturingProcess)

image(is.na(imputedManufacturing), main = "Missing Values", xlab = "Observation", ylab = "Variable", xaxt = "n", yaxt = "n", bty = "n")
axis(1, seq(0, 1, length.out = nrow(imputedManufacturing)), 1:nrow(imputedManufacturing), col = "white")

#splitting data
trainingRows <- createDataPartition(imputedManufacturing[,1], p=0.80, list=FALSE)

training <- imputedManufacturing[trainingRows,]
testing <- imputedManufacturing[-trainingRows,]

ctrl <- trainControl(method = "cv", number = 3)
xTest <- imputedManufacturing[,2:58]

##lm##

lmFit2 <- train(Yield ~ ., data = imputedManufacturing, method = "lm", preProc = c("center", "scale"), trControl = ctrl)
lmFit2
#plot(lmFit2)

predicted <- predict(lmFit2, xTest)
lmValues2 <- data.frame(obs = imputedManufacturing[,1], pred = predicted)

defaultSummary(lmValues2)

##Ridge##

ridgeGrid <- data.frame(.lambda = seq(0, 1, length = 15))
ridgeRegFit <- train(Yield ~ ., data = imputedManufacturing, method = "ridge", preProc = c("center", "scale"), trControl = ctrl, tuneGrid = ridgeGrid)
ridgeRegFit
plot(ridgeRegFit)

predicted <- predict(ridgeRegFit, xTest)
ridgeValues <- data.frame(obs = imputedManufacturing[,1], pred = predicted)

defaultSummary(ridgeValues)

##lasso##

lassoGrid <- expand.grid(alpha = 1, lambda = c(seq(0.1, 2, by =0.1) ,  seq(2, 5, 0.5) , seq(5, 25, 1)))
lassoFit <- train(Yield ~ ., data = imputedManufacturing, method = "glmnet", preProc = c("center", "scale"), trControl = ctrl, tuneGrid = lassoGrid)
lassoFit
plot(lassoFit)

predicted <- predict(lassoFit, xTest)
lassoValues <- data.frame(obs = imputedManufacturing[,1], pred = predicted)

defaultSummary(lassoValues)

##elastic net##

enetGrid <- expand.grid(.lambda = c(0, 0.01, .1), .fraction = seq(.05, 1, length = 20))
enetTune <- train(Yield ~ ., data = imputedManufacturing, method = "enet", preProc = c("center", "scale"), trControl = ctrl, tuneGrid = enetGrid)
enetTune
plot(enetTune)

predicted <- predict(enetTune, xTest)
enetValues <- data.frame(obs = imputedManufacturing[,1], pred = predicted)

defaultSummary(enetValues)

##Variable Importance Plot##

#lm has the lowest RMSE
lmImp <- varImp(lmFit2, scale = FALSE)
lmImp
plot(lmImp, top = 20)
