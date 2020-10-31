library(AppliedPredictiveModeling)
library(caret)
library(earth)
library(kernlab)
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
points(x = dataGrid$x, y = modelPrediction[,1], type = "l", col = "blue")
