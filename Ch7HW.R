library(AppliedPredictiveModeling)
library(caret)
library(earth)
library(kernlab)
library(mlbench)
library(nnet)
library(tidyverse)

####7.1####

set.seed(100)
x <- runif(100, min = 2, max = 10)
y <- sin(x) + rnorm(length(x)) * .25
sinData <- data.frame(x = x, y = y)
plot(x, y)
## Create a grid of x values to use for prediction
dataGrid <- data.frame(x = seq(2, 10, length = 100))

####7.1a####

#example
rbfSVM <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = "automatic", C = 1, epsilon = 0.1)
modelPrediction <- predict(rbfSVM, newdata = dataGrid)
plot(x, y)
points(x = dataGrid$x, y = modelPrediction[,1], type = "l", col = "blue")

#constant C, change epsilon
rbfSVM <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = "automatic", C = 1, epsilon = 1)
modelPrediction <- predict(rbfSVM, newdata = dataGrid)
plot(x, y)
points(x = dataGrid$x, y = modelPrediction[,1], type = "l", col = "blue")

rbfSVM <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = "automatic", C = 1, epsilon = 10)
modelPrediction <- predict(rbfSVM, newdata = dataGrid)
plot(x, y)
points(x = dataGrid$x, y = modelPrediction[,1], type = "l", col = "blue")

#change C, constant epsilon
rbfSVM <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = "automatic", C = 10, epsilon = 0.1)
modelPrediction <- predict(rbfSVM, newdata = dataGrid)
plot(x, y)
points(x = dataGrid$x, y = modelPrediction[,1], type = "l", col = "blue")

rbfSVM <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = "automatic", C = 0.1, epsilon = 0.1)
modelPrediction <- predict(rbfSVM, newdata = dataGrid)
plot(x, y)
points(x = dataGrid$x, y = modelPrediction[,1], type = "l", col = "blue")

####7.1b####

rbfSVM <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = list(sigma = 1), C = 1, epsilon = 0.1)
modelPrediction <- predict(rbfSVM, newdata = dataGrid)
plot(x, y)
points(x = dataGrid$x, y = modelPrediction[,1], type = "l", col = "blue")

rbfSVM <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = list(sigma = 0.1), C = 1, epsilon = 0.1)
modelPrediction <- predict(rbfSVM, newdata = dataGrid)
plot(x, y)
points(x = dataGrid$x, y = modelPrediction[,1], type = "l", col = "blue")

rbfSVM <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = list(sigma = 10), C = 1, epsilon = 0.1)
modelPrediction <- predict(rbfSVM, newdata = dataGrid)
plot(x, y)
points(x = dataGrid$x, y = modelPrediction[,1], type = "l", col = "blue")

####7.2####

set.seed(200)
trainingData <- mlbench.friedman1(200, sd = 1)
trainingData$x <- data.frame(trainingData$x)
featurePlot(trainingData$x, trainingData$y)
testData <- mlbench.friedman1(5000, sd = 1)
testData$x <- data.frame(testData$x)

#example
knnModel <- train(x = trainingData$x, y = trainingData$y, method = "knn", preProc = c("center", "scale"), tuneLength = 10)
knnModel
knnPred <- predict(knnModel, newdata = testData$x)
postResample(pred = knnPred, obs = testData$y)

marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:50)
marsModel <- train(x = trainingData$x, y = trainingData$y, method = "earth", tuneGrid = marsGrid, trControl = trainControl(method = "cv"))
marsModel
marsPred <- predict(marsModel, newdata = testData$x)
postResample(pred = marsPred, obs = testData$y)

varImp(marsModel)

####7.5####

set.seed(80)

data(ChemicalManufacturingProcess)
?ChemicalManufacturingProcess

#yield is outcome

#imputing missing values
imputedManufacturing <- predict(preProcess(ChemicalManufacturingProcess, "knnImpute"), ChemicalManufacturingProcess)

#splitting data
trainingRows <- createDataPartition(imputedManufacturing[,1], p=0.80, list=FALSE)

training <- imputedManufacturing[trainingRows,]
testing <- imputedManufacturing[-trainingRows,]

ctrl <- trainControl(method = "cv", number = 3)
xTest <- imputedManufacturing[,2:58]

#Neural Network

#MARS

#SVM

#KNN