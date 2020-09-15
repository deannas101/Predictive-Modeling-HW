library(AppliedPredictiveModeling)
library(caret)
library(corrplot)
library(e1071)
library(ggpubr)
library(mlbench)
library(RColorBrewer)
library(tidyverse)

####Exercise 3.1####
#first the data is loaded
data(Glass)

#next the 'Type' column needs to be removed from the dataset,
#there is no need to remove more columns as the predictors are all continuous
type <- Glass[,10]
clean_glass <- Glass[,1:9]
dirty_glass <- Glass[,1:9]

#centering and scaling the data
clean_glass <- clean_glass %>%
    scale(center = TRUE, scale = TRUE) %>%
    as.data.frame()

#to visualize the data, boxplots of each predictor will be made into one plot
ri_plot <- ggplot(clean_glass, aes(x="",y=RI)) +
    geom_violin() +
    labs(x = "RI", y = "Values")

na_plot <- ggplot(clean_glass, aes(x="",y=Na)) +
    geom_violin() +
    labs(x = "Na", y = "Values")

mg_plot <- ggplot(clean_glass, aes(x="",y=Mg)) +
    geom_violin() +
    labs(x = "Mg", y = "Values")

al_plot <- ggplot(clean_glass, aes(x="",y=Al)) +
    geom_violin() +
    labs(x = "Al", y = "Values")

si_plot <- ggplot(clean_glass, aes(x="",y=Si)) +
    geom_violin() +
    labs(x = "Si", y = "Values")

k_plot <- ggplot(clean_glass, aes(x="",y=K)) +
    geom_violin() +
    labs(x = "K", y = "Values")

ca_plot <- ggplot(clean_glass, aes(x="",y=Ca)) +
    geom_violin() +
    labs(x = "Ca", y = "Values")

ba_plot <- ggplot(clean_glass, aes(x="",y=Ba)) +
    geom_violin() +
    labs(x = "Ba", y = "Values")

fe_plot <- ggplot(clean_glass, aes(x="",y=Fe)) +
    geom_violin() +
    labs(x = "Fe", y = "Values")

ggarrange(ri_plot, na_plot, mg_plot, al_plot, si_plot, k_plot, ca_plot, ba_plot, fe_plot)

#checking the skewness of the predictors
apply(clean_glass, 2, skewness)

#next is a correlation matrix to show the relationships between predictors
corrplot(cor(clean_glass), method = "number", type = "upper", col = brewer.pal(n=8, name="RdYlBu"))

#as for transformations, all predictors except for Na would benefit from a transformation as they are moderately or highly skewed
#do I need to know which transformation for each predictor?
####Exercise 3.2####
data("Soybean")

#remove ordered factors and compensated for everything increased by 1 during conversion
Soybean <- Soybean %>%
    mutate_at(vars(`date`:`roots`), as.numeric)
    
Soybean[,2:36] <- Soybean[,2:36] - 1

#make the data long for plotting
long_soy <- Soybean %>%
    pivot_longer(cols = c(date:roots), names_to = "predictor", values_to = "values")

#plotting!
ggplot(long_soy, aes(x = values)) +
    geom_histogram() +
    facet_wrap(~predictor) +
    theme_minimal() +
    labs(title = "Frequency of Predictors", x = "Predictor", y = "Count")

#delinquent predictors don't have much variation and one value is way higher than the others
nearZeroVar(Soybean)

#seeing missing values
image(is.na(Soybean), main = "Missing Values", xlab = "Observation", ylab = "Variable", xaxt = "n", yaxt = "n", bty = "n")
axis(1, seq(0, 1, length.out = nrow(Soybean)), 1:nrow(Soybean), col = "white")

#for missing data I would use imputation over removing predictors due to the pattern of the missing data.
#Sections of rows are missing versus a column of data from a particular predictor.
####Exercise 3.3####
data(BloodBrain)

#preprocessing data for correlations
transformations <- bbbDescr %>%
    preProcess(method = c("center", "scale"))

clean_brain <- predict(transformations, bbbDescr)

#plotting a correlation matrix, which is an absolute nightmare
correlation <- cor(clean_brain)
corrplot(correlation, method = "shade", order = "hclust", type = "upper", col = brewer.pal(n=8, name="RdYlBu"), tl.pos = 'n')

#here's what correlated predrictors should be removed
high_cor <- findCorrelation(correlation, cutoff = .85)
remaining_predictors <- ncol(clean_brain) - length(high_cor)
remaining_predictors