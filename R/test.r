library(arrow)
library(caret)

clean <- function() {
        rm(list = ls())
        invisible(gc())
}

#! -------------- CALCULATE METRICS WITH DATAFRAME ------------------------

getConfusionMatrix <- function(results) {
        confusion <- data.frame(TruePositive = 0,
                                FalseNegative = 0,
                                FalsePositive = 0,
                                TrueNegative = 0)
        results$predicted <- as.logical(results$predicted)
        results$labels <- as.logical(results$labels)

        confusion$TruePositive <-
                sum(results$predicted & results$labels)        # TP
        confusion$FalseNegative <-
                sum(!results$predicted & results$labels)       # FN
        confusion$FalsePositive <-
                sum(results$predicted & !results$labels)       # FP
        confusion$TrueNegative <-
                sum(!results$predicted & !results$labels)      # TN
        return(confusion)
}

getAccuracy <- function(confusion) {
        return((confusion$TruePositive + confusion$TrueNegative)
                / sum(confusion))
}

getPrecision <- function(confusion) {
        return(confusion$TruePositive /
                (confusion$TruePositive + confusion$FalsePositive))
}

getRecall <- function(confusion) {
        return(confusion$TruePositive /
                (confusion$TruePositive + confusion$FalseNegative))
}

getF1 <- function(confusion) {
        precision <- getPrecision(confusion)
        recall <- getRecall(confusion)
        return(2 * precision * recall / (precision + recall))
}

getMetrics <- function(results) {
        confusion <- getConfusionMatrix(results)
        accuracy <- getAccuracy(confusion)
        precision <- getPrecision(confusion)
        recall <- getRecall(confusion)
        f1 <- getF1(confusion)
        return(c(accuracy, precision, recall, f1))
}
#! ------------------------------------------------------------------------

#! -------------- CALCULATE METRICS WITH MATRIX ------------------------
# getConfusionMatrix <- function(results) {
#         confusionMatrix <- matrix(0, nrow = 2, ncol = 2)
#         results$predicted <- as.logical(results$predicted)
#         results$labels <- as.logical(results$labels)
#         confusionMatrix[1, 1] <-
#                 sum(results$predicted & results$labels)        # TP
#         confusionMatrix[1, 2] <-
#                 sum(!results$predicted & results$labels)       # FN
#         confusionMatrix[2, 1] <-
#                 sum(results$predicted & !results$labels)       # FP
#         confusionMatrix[2, 2] <-
#                 sum(!results$predicted & !results$labels)      # TN
#         return(confusionMatrix)
# }
# getAccuracy <- function(confusion) {
#         return((confusion[1, 1] + confusion[2, 2]) / sum(confusion))
# }
# getPrecision <- function(confusion) {
#         return(confusion[1, 1] / (confusion[1, 1] + confusion[2, 1]))
# }
# getRecall <- function(confusion) {
#         return(confusion[1, 1] / (confusion[1, 1] + confusion[1, 2]))
# }
# getF1 <- function(confusion) {
#         precision <- getPrecision(confusion)
#         recall <- getRecall(confusion)
#         return(2 * precision * recall / (precision + recall))
# }
# getMetrics <- function(results) {
#         confusion <- getConfusionMatrix(results)
#         accuracy <- getAccuracy(confusion)
#         precision <- getPrecision(confusion)
#         recall <- getRecall(confusion)
#         f1 <- getF1(confusion)
#         return(c(accuracy, precision, recall, f1))
# }
#! ---------------------------------------------------------------------

covid <- data.frame(read_parquet("covidClean.parquet"))

covid[, c("PATIENT_ID", "USMER", "MEDICAL_UNIT", "PATIENT_TYPE",
        "ADMISSION_DATE", "SYMPTOMS_DATE",
        "DEATH_DATE", "ORIGIN_COUNTRY")] <- NULL
covid[, c("DIED", "ICU", "INTUBED")] <- NULL


for (col in colnames(covid)){
        if (is.logical(covid[, col])) {
                covid[, col] <- as.factor(covid[, col])
        }
}


#! -------------- TRAINING AND TEST SET ------------------------

indexis <- createDataPartition(covid$AT_RISK, p = 0.75, list = FALSE)

trainingSet <- covid[indexis, ]
testSet <- covid[-indexis, ]

testSet_True <- testSet[, "AT_RISK"]
testSet_Data <- testSet[, - which(colnames(trainingSet) == "AT_RISK")]

#! ----------------------------------------------------------


if (0) {
#! -------------- DECISION TREE CART ------------------------

cart <- train(trainingSet[, 1:ncol(trainingSet)-1],
        trainingSet$AT_RISK, method = "rpart", trControl = cntrl)

predicted <- predict(cart, testSet_Data)
results <- data.frame(labels = testSet_True, predicted = predicted)

metrics <- getMetrics(results)

print(paste("Metrics CART", metrics))

}
#! ----------------------------------------------------------

if (0) {
#! -------------- DECISION TREE C4.5 ------------------------

c4.5 <- train(trainingSet[, 1:ncol(trainingSet)-1],
        trainingSet$AT_RISK, method = "J48", trControl = cntrl)

predicted <- predict(c4.5, testSet_Data)
results <- data.frame(labels = testSet_True, predicted = predicted)

metrics <- getMetrics(results)

print(paste("Metrics C4.5", metrics))

}
#! ----------------------------------------------------------


if (0) {
#! -------------- RANDOM FOREST ------------------------

rf <- train(trainingSet[, 1:ncol(trainingSet)-1],
        trainingSet$AT_RISK, method = "rf", trControl = cntrl)

predicted <- predict(rf, testSet_Data)
results <- data.frame(labels = testSet_True, predicted = predicted)

metrics <- getMetrics(results)

print(paste("Metrics Random Forest", metrics))

}
#! ----------------------------------------------------------

if (0) {
#! -------------- NAIVE BAYES CONTINUE ------------------------

nb <- train(trainingSet[, 1:ncol(trainingSet)-1],
        trainingSet$AT_RISK, method = "naive_bayes", trControl = cntrl)

predicted <- predict(nb, testSet_Data)
results <- data.frame(labels = testSet_True, predicted = predicted)

metrics <- getMetrics(results)

print(paste("Metrics Naive Bayes Continue: ", metrics))

}
#! ----------------------------------------------------------

if (1) {
#! -------------- NAIVE BAYES DISCRETE ------------------------

trainingSet[, "AGE"] <- as.factor(trainingSet[, "AGE"])
testSet_Data[, "AGE"] <- factor(testSet_Data[, "AGE"],
        levels = levels(trainingSet[, "AGE"]))

nb <- train(trainingSet[, 1:ncol(trainingSet)-1],
        trainingSet$AT_RISK, method = "nbDiscrete", trControl = cntrl)

predicted <- predict(nb, testSet_Data)
results <- data.frame(labels = testSet_True, predicted = predicted)

metrics <- getMetrics(results)

print("Metrics Naive Bayes Discrete: ")
print(metrics)

}
#! ----------------------------------------------------------

clean()
