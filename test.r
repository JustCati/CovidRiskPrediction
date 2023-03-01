library(arrow)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)

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
# #! -------------- CALCULATE METRICS WITH MATRIX ------------------------
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
# #! ---------------------------------------------------------------------

covid <- data.frame(read_parquet("covidClean.parquet"))

covid[, c("PATIENT_ID", "USMER", "MEDICAL_UNIT", "PATIENT_TYPE",
        "ADMISSION_DATE", "SYMPTOMS_DATE",
        "DEATH_DATE", "ORIGIN_COUNTRY")] <- NULL
covid[, c("DIED", "ICU", "INTUBED")] <- NULL


#! -------------- TRAINING AND TEST SET ------------------------
indexis <- createDataPartition(covid$AT_RISK, p = 0.75, list = FALSE)

trainingSet <- covid[indexis, ]
testSet <- covid[-indexis, ]

testSet_True <- testSet[, "AT_RISK"]
testSet_Data <- testSet[, - which(colnames(trainingSet) == "AT_RISK")]
#! ----------------------------------------------------------


if (FALSE) {
#! -------------- DECISION TREE CART ------------------------

cart <- rpart(AT_RISK ~ ., data = trainingSet, method = "class")
cp <- cart$cptable[which.min(cart$cptable[, "xerror"]), "CP"]
cartPruned <- prune(cart, cp = cp)

predicted <- predict(cartPruned, testSet_Data, type = "class")
results <- data.frame(labels = testSet_True, predicted = predictedCart)

metrics <- getMetrics(results)

print("Metrics CART")
print(metrics)

}
#! ----------------------------------------------------------

if (TRUE) {
#! -------------- RANDOM FOREST ------------------------

rf <- randomForest(AT_RISK ~ ., data = trainingSet)
plot(rf)

}
#! ----------------------------------------------------------

clean()
