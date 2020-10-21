library(AppliedPredictiveModeling)
library(caret)
libray(e1071)
library(tidyverse)

####6.1####

data(tecator)
?tecator

#absorp - absorbance data
#endpoints - percentages of water, fat, protein (cols 1-3)

colnames(absorp) <- c(1:100)
pca_absorp <- predict(preProcess(absorp, method = c("center", "scale", "pca")), absorp) #default cutoff is 95%

#make: screeplot

#dimensions after pca: 215 x 2

trainingRows <- createDataPartition(absorp[,1], p=0.80, list=FALSE)
testingRows <- createDataPartition(absorp[,1], p=0.20, list=FALSE)

training <- absorp[trainingRows,]
testing <- absorp[testingRows,]


#?RMSE

####6.2####

data(permeability)
?permeability

#fingerprints - matrix of binary fingerprint indicator variables
#permeability - permeability values for each compound

remove <- nearZeroVar(fingerprints)
fingers <- fingerprints[,-remove]
#388 predictors left for modeling

