#################################################
# load packages, data, and register cores for parallel
library(dplyr)
library(FUNctions)
library(caret)
library(AppliedPredictiveModeling)
library(Hmisc)
library(C50)
library(subselect)
library(corrplot)
library(ggbiplot)
library(kernlab)
library(doMC)

registerDoMC(4)
data(churn)

#################################################
#################################################
# have a quick look at data
# we see a very strong class imbalance, so we may want
# to take this into account for the metric to tune models to
# View(churnTrain)
describe(churnTrain)
summary(churnTrain)

# accuracy not the best metric- as I can get 85% accuracy by
# just guess no churn. More interested in sensitivity
# ie out of those we say churn, who actually churns?
table(churnTrain$churn)

# pull out predictive factors and the outcome
trainOutcome <- churnTrain$churn
testOutcome <- churnTest$churn
trainPred <- churnTrain[!names(churnTrain) == "churn"]
testPred <- churnTest[!names(churnTest) == "churn"]

#################################################
#################################################
# strategy: explore covariates, highlight any potential issues
# recode categorical as dummies
# calculate interactions between numerical predictors
# extract a full and reduced set of predictors
# build variety of models
# evaluate models with lift curves, calibration plots etc

##################################################
#################################################
# step 1: some visualisations

# firstly see what class the predictors are
vapply(churnTrain, class, character(1))

# seperate out factor, double (continuous) and integer (count) predictors
# train
facCols <- trainPred[, vapply(trainPred, is.factor, logical(1))]
numCols <- trainPred[, vapply(trainPred, is.double, logical(1))]
countCols <- trainPred[, vapply(trainPred, is.integer, logical(1))]

# test 
facColsT <- testPred[, vapply(testPred, is.factor, logical(1))]
numColsT <- testPred[, vapply(testPred, is.double, logical(1))]
countColsT <- testPred[, vapply(testPred, is.integer, logical(1))]

# CONTINUOUS NUMERIC PREDICTORS
# custom phil plots- if you dont have my FUNctions library you 
# cant make these. Also, you may want to build your own
# color theme- philTheme() probably isn't available to you!
plotListDouble <- ggplotListDens(numCols, trainOutcome)
ggMultiplot(plotListDouble, cols = 2)

# investigate correlated covariates- costs and minutes
ggplot(numCols, aes(x = total_day_minutes,
                    y = total_day_charge,
                    color = factor(trainOutcome))) +
  geom_point(alpha = 0.4, size = 4) +
  theme_bw() +
  scale_color_manual(values = philTheme()[c(4, 1)], name = "Churn") +
  theme(legend.position = c(0.1, 0.8),
        legend.text.align = 0) 

# investigate pairwise relation with caret::featurePlot
transparentTheme(trans = 0.1)
featurePlot(x = numCols,
            y = trainOutcome,
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 2))

# COUNT DATA
plotListCount <- ggplotListHist(countCols, trainOutcome)
ggMultiplot(plotListCount, cols = 3)

# pairs
transparentTheme(trans = 0.1)
featurePlot(x = countCols,
            y = trainOutcome,
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 2))

# do we see any obvious class seperation for numeric?
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)

featurePlot(x = countCols,
            y = ifelse(trainOutcome== "yes", 1, 0),
            plot = "scatter",
            layout = c(4, 2))
featurePlot(x = numCols,
            y = ifelse(trainOutcome== "yes", 1, 0),
            plot = "scatter",
            layout = c(4, 2))

# FACTORS
plotListFac <- ggplotListBar(facCols, trainOutcome)
ggMultiplot(plotListFac, cols = 2)

##################################################
##################################################
# Lets act on what we discovered with visualistions.
# Remember everything done for train must be done for test!

# loose the charges, as correlated with minutes (totally uninformative
# to keep both)
# Some models will be very sensitive to non-informative predictors,
# so will further investigate correlations shortly
numCols <- numCols[, !names(numCols) %in% c("total_day_charge",
                                            "total_eve_charge",
                                            "total_night_charge",
                                            "total_intl_charge")]
numColsT <- numColsT[, !names(numCols) %in% c("total_day_charge",
                                              "total_eve_charge",
                                              "total_night_charge",
                                              "total_intl_charge")]

# not too worried about skewness as will apply
# other preprocessing to sensitive models (e.g nnet)
# during training process, such as spatial sign transformation

##################################################
##################################################
# process categorical data

# can look at chisq or fisher tests to ivestigate odds
# or obtain p-values for association where appropriate
table(churnTrain$international_plan, churnTrain$churn)
fisher.test(table(churnTrain$international_plan, churnTrain$churn))

# can we diagnose how informative the state is?
# chisq test fails- my guess is too many levels
stateTab <- table(churnTrain$state, churnTrain$churn)
chisq.test(stateTab)

# for now, lets keep all factors. dummy up.
catDummies <- dummyVars(~. ,
                        data = facCols)

facTrans <- data.frame(predict(catDummies, facCols))
facTransT <- data.frame(predict(catDummies, facColsT))

# important to remember that for nonlinear models, some will be negatively impacted
# by colliniear predictors, eg those where a binary yes/no we only need one to deduce
# the other. To be on the safe side, lets remove one for each category.
# the unfortunate side effect is that models that are uneffected by this become less interpretable. 
# however, we are more concerned with predictive accuracy than interpretability
# (many nonlinear models will be uninterpretable anyway)

# use subselect::trim.matrix() to find collinear
reducedCovMat <- cov(facTrans[, ])
trimmingResults <- trim.matrix(reducedCovMat)
trimmingResults$names.discarded

# remove offending columns
facTrans <- facTrans[, !names(facTrans) %in% trimmingResults$names.discarded]
facTransT <- facTransT[, !names(facTransT) %in% trimmingResults$names.discarded]

# rename
facTrans <- facTrans %>%
  dplyr::rename(voice_mail_plan = voice_mail_plan.yes,
                international_plan = international_plan.yes,
                area_code_510 = area_code.area_code_510,
                area_code_408 = area_code.area_code_408)

facTransT <- facTransT %>%
  dplyr::rename(voice_mail_plan = voice_mail_plan.yes,
                international_plan = international_plan.yes,
                area_code_510 = area_code.area_code_510,
                area_code_408 = area_code.area_code_408)

# check for zero variance and near zero variance
# we see that states have near zero variance...
# could imply not highly informative. For sensitive models
# we definitley don't want to include. Not such an issue for those 
# which can peform feature selection.
# we will come back to this when building a full and reduced set 
# of predictors
nzvFac <- nearZeroVar(facTrans, saveMetric = TRUE)

#################################################
#################################################
# next step - lets combine all of our numerical predictors
numInput <- cbind(numCols, countCols)
numInputT <- cbind(numColsT, countColsT)

# combine with categorical
trainInput <- cbind(numInput, facTrans)
testInput <- cbind(numInputT, facTransT)

#################################################
#################################################
# start filtering. make a full set and a reduced set- full set for
# models that can do feature selection, reduce set for those that cannot

# remove near zero variance for reduced set
isNZV <- nearZeroVar(trainInput, saveMetrics = TRUE)
fullSet <- names(trainInput[, !isNZV$zeroVar])
reducedSet <- names(trainInput[, !isNZV$nzv])

# investigate correlation- set a threshold of 0.9 for 
# reduced set, 0.99 for full set
trainCorr <- cor(trainInput)
highCorr <- findCorrelation(trainCorr, cutoff = 0.9)
fullCorr <- findCorrelation(trainCorr, cutoff = 0.99)
highCorrNames <- names(trainInput)[highCorr]
fullCorrNames <- names(trainInput)[fullCorr]

fullSet <- fullSet[!fullSet %in% fullCorrNames]
reducedSet <- reducedSet[!reducedSet %in% highCorrNames]

# do a pretty correlation plot
corrplot(cor(trainInput[, reducedSet]), order = "hclust", tl.cex = .6)

# do a pca plot
trainPCA <- prcomp(trainInput[, reducedSet], scale = TRUE)
type <- trainOutcome
reduced_pca <- ggbiplot(trainPCA, obs.scale = 1, 
                        var.scale = 1,
                        groups = type,
                        ellipse = TRUE,
                        circle = TRUE,
                        var.axes = FALSE,
                        varname.size = 3,
                        alpha = 0.3)
reduced_pca +
  theme_bw() +
  scale_color_manual(values = philTheme()[c(6, 1)], name = "Churn") 

#################################################
#################################################
# fit models
# always set seeds before training for reproduceability

# set up train control
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 5,
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     summaryFunction = twoClassSummary)

#################################################
# qda
# regularized DA (mix of lda and qda)
# tune over lambda and gamma
set.seed(476)
qdaTune <- train(x = trainInput[, reducedSet],
                 y = trainOutcome,
                 method = "rda",
                 preProc = c("center", "scale"),
                 metric = "Sens",
                 trControl = ctrl)
# we see best model has lambda = 0, gamma = 1
# this corresponds to
qdaTune

# can save and load models like this:
#save(qdaTune, file = "qdaChurn.Rdata")
#load("qdaChurn.Rdata")

# predictions for test set: can do for class and probabilities
qdaPred <- predict(qdaTune, newdata = testInput[, reducedSet])
qdaPredProb <- predict(qdaTune, newdata = testInput[, reducedSet], type = "prob")

# sensitivity of 0.446 on test set
confusionMatrix(data = qdaPred, reference = testOutcome)

#################################################
# nnet
# set up grid and params
nnetGrid <- expand.grid(size = 1:10,
                        decay = c(0.01, 0.03, 0.1, 0.3, 1))
maxSize <- max(nnetGrid$size)
numWts <- (maxSize * (length(reducedSet) + 1) + maxSize + 1)

set.seed(476)
# do a spatialSign transformation on data-
# can really boost model peformance
nnetTune <- train(x = trainInput[, reducedSet],
                  y = trainOutcome,
                  method = "nnet",
                  metric = "Sens",
                  preProc = c("center", "scale", "spatialSign"),
                  tuneGrid = nnetGrid,
                  trace = FALSE,
                  maxit = 1000,
                  MaxNWts = numWts,
                  trControl = ctrl)
nnetTune
#save(nnetTune, file = "nnetChurn.Rdata")
#load("nnetChurn.Rdata")

nnetPred <- predict(nnetTune, newdata = testInput[, reducedSet])
nnetPredProb <- predict(nnetTune, newdata = testInput[, reducedSet], type = "prob")

# sensitivity of 0.705 on test set
confusionMatrix(data = nnetPred, reference = testOutcome)

#################################################
# fda with MARS
marsGrid <- expand.grid(degree = 1:2,
                        nprune = seq(2, 40, 2))
set.seed(476)
fdaTune <- train(x = trainInput[, fullSet],
                 y = trainOutcome,
                 method = "fda",
                 metric = "Sens",
                 preProc = c("center", "scale"),
                 tuneGrid = marsGrid,
                 trControl = ctrl)

fdaTune
#save(fdaTune, file = "fdaChurn.Rdata")
#load("fdaChurn.Rdata")

fdaPred <- predict(fdaTune, newdata = testInput[, fullSet])
fdaPredProb <- predict(fdaTune, newdata = testInput[, fullSet], type = "prob")

# sensitivity of 0.746 on test set
confusionMatrix(data = fdaPred, reference = testOutcome)

#################################################
# svm Radial basis
# estimate sigma
set.seed(123)
sigmaRangeReduced <- sigest(as.matrix(trainInput[, reducedSet]))
svmRGridReduced <- expand.grid(sigma = sigmaRangeReduced[1],
                               C = 2 ^ (seq(-4, 4)))
set.seed(476)
svmRTune <- train(x = trainInput[, reducedSet],
                  y = trainOutcome,
                  method = "svmRadial",
                  metric = "Sens",
                  preProc = c("center", "scale"),
                  tuneGrid = svmRGridReduced,
                  fit = FALSE,
                  trControl = ctrl)
svmRTune
#save(svmRTune, file = "svmRChurn.Rdata")
#load("svmRChurn.Rdata")

svmRPred <- predict(svmRTune, newdata = testInput[, reducedSet])
svmRPredProb <- predict(svmRTune, newdata = testInput[, reducedSet], type = "prob")

# sensitivity of 0.549 on test set
confusionMatrix(data = svmRPred, reference = testOutcome)

################################################
# svm poly
svmPGrid <-  expand.grid(degree = 1:2,
                         scale = c(0.01, .005),
                         C = 2 ^ (seq(-6, -2, length = 10)))
set.seed(476)
svmPTune <- train(x = trainInput[, reducedSet],
                     y = trainOutcome,
                     method = "svmPoly",
                     metric = "Sens",
                     preProc = c("center", "scale"),
                     tuneGrid = svmPGrid,
                     trControl = ctrl)
svmPTune
#save(svmPTune, file = "svmPChurn.Rdata")
#load("svmPChurn.Rdata")

svmPPred <- predict(svmPTune, newdata = testInput[, reducedSet])
svmPPredProb <- predict(svmPTune, newdata = testInput[, reducedSet], type = "prob")

# sensitivity of 0.469 on test set
confusionMatrix(data = svmPPred, reference = testOutcome)
#################################################
# k-nn
set.seed(476)
knnTune <- train(x = trainInput[, reducedSet],
                y = trainOutcome,
                method = "knn",
                metric = "Sens",
                preProc = c("center", "scale"),
                tuneGrid = data.frame(k = seq(1, 51, 2)),
                trControl = ctrl)
knnTune
#save(knnTune, file = "knnChurn.Rdata")
#load("knnChurn.Rdata")

knnPred <- predict(knnTune, newdata = testInput[, reducedSet])
knnPredProb <- predict(knnTune, newdata = testInput[, reducedSet], type = "prob")

# sensitivity of 0.366 on test set... remarkable considering
# how simple the method is
confusionMatrix(data = knnPred, reference = testOutcome)

#################################################
#################################################
# gather results for comparison

# look at training resamples for model metrics
models <- list(qda = qdaTune,
               nnet = nnetTune,
               fda = fdaTune,
               svmR = svmRTune,
               svmP = svmPTune,
               knn = knnTune)

resamp <- resamples(models)
bwplot(resamp)
splom(resamp)

# two top models 
# reject the null - fda is better than nnet 
# for sensitivity
t.test(resamp$values$`fda~Sens`,
       resamp$values$`nnet~Sens`,
       paired = TRUE)

# pull out results so can look at lift curves and calibration
# need predicted probabilities for this
results <- data.frame(qda = qdaPredProb$yes,
                      nnet = nnetPredProb$yes,
                      fda = fdaPredProb$yes,
                      svmR = svmRPredProb$yes,
                      svmP = svmPPredProb$yes,
                      knn = knnPredProb$yes,
                      class = testOutcome)

# calibration curves
# to make max's pretty plots use bookTheme() from APM package
trellis.par.set(bookTheme())
calCurve <- calibration(class ~ svmR + nnet + fda,
                        data = results)
calCurve
xyplot(calCurve,
       auto.key = list(columns = 3))

# lift curves
liftCurve <- lift(class ~ svmR + nnet + fda, data = results)
liftCurve
xyplot(liftCurve,
       auto.key = list(columns = 2,
                       lines = TRUE,
                       points = FALSE))

# lift plot plots CumTestedPct (x-axis) vs CumEventPct(y-axis)
# for fda,
# for cumEventPct = 80, we need 15.4 % of cumTestedPct
# 15.4 of total is 0.154 * 1667 = 257 samples
# of this 257, 0.8 * 224 = 179 are churns
