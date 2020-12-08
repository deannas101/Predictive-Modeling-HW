library(AppliedPredictiveModeling)
library(caret)
library(glmnet)
library(MASS)
library(tidyverse)

# 13.1 ------------------------------------------------------------------------

data(hepatic)

# creating dataset
full_dataset <- bio
full_dataset[, 185] <- injury
# injury is named V378


# splitting data
training_rows <- createDataPartition(full_dataset[, 185], p = 0.80, list = FALSE)

training_set <- full_dataset[trainingRows, ]
testing_set <- full_dataset[-trainingRows, ]

# will determine the best model using kappa since there are 3 classes

# Quadratic Discriminant Analysis

# Regularized Discriminant Analysis

# Mixture Discriminant Analysis

# Neural Networks

# Flexible Discriminant Analysis

# Support Vector Machine

# K-Nearest Neighbors

# Naive Bayes