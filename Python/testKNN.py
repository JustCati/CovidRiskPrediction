import math

import cudf as cu
from cuml.neighbors import KNeighborsClassifier
from cuml.model_selection import train_test_split as cuTrainTestSplit

#! ------ CALCULATE METRICS -------

def getConfusionMatrix(labels, predictions):
    confusion = dict()
    confusion["TP"] = sum(labels & predictions)
    confusion["TN"] = sum(~labels & ~predictions)
    confusion["FP"] = sum(~labels & predictions)
    confusion["FN"] = sum(labels & ~predictions)
    return confusion

def getMetrics(confusion):
    metrics = dict()
    metrics["accuracy"] = (confusion["TP"] + confusion["TN"]) / (confusion["TP"] + confusion["TN"] + confusion["FP"] + confusion["FN"])
    metrics["precision"] = confusion["TP"] / (confusion["TP"] + confusion["FP"])
    metrics["recall"] = confusion["TP"] / (confusion["TP"] + confusion["FN"])
    metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
    return metrics

def printMetrics(confusion):
    print("Accuracy: \t", confusion["accuracy"])
    print("Precision: \t", confusion["precision"])
    print("Recall: \t", confusion["recall"])
    print("F1: \t\t", confusion["f1"])
#! ---------------------------------

cuCovid = cu.read_parquet('covidClean.parquet')

toRemove = ["PATIENT_ID", "USMER", "SYMPTOMS_DATE",
        "MEDICAL_UNIT", "ADMISSION_DATE", "PATIENT_TYPE",
        "DEATH_DATE", "ORIGIN_COUNTRY"]
cuCovid = cuCovid.drop(columns = toRemove)
cuCovid = cuCovid.drop(columns= ["DIED", "INTUBED", "ICU"])
cuCovid = cuCovid.astype("float32")

labels = cuCovid["AT_RISK"]
covidX = cuCovid.drop(columns = ["AT_RISK"])
cuTrainingX, cuTestX, cuTrainingY, cuTestY = cuTrainTestSplit(covidX, labels, test_size=0.25)

cuTestY = cuTestY.to_pandas().to_numpy().astype("bool")




#! ------------- KNN ------------------

def findBestK(lower, upper, metric="accuracy"):
    median = (lower + upper) // 2
    if lower == median:
        return lower
    
    results = dict()
    for k in [lower, median, upper]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(cuTrainingX, cuTrainingY)
        predicted = knn.predict(cuTestX)
        
        predicted = predicted.to_pandas().to_numpy().astype("bool")

        results[k] = getMetrics(getConfusionMatrix(cuTestY, predicted))[metric]
        print("K: ", k, metric, ": ", results[k])

    if results[median] > results[lower]:
        return findBestK(median, upper, metric)
    else:
        return findBestK(lower, median, metric)

bestK = findBestK(1, 1000, "accuracy")
print("Best K: ", bestK)

knn = KNeighborsClassifier(n_neighbors=827)
knn.fit(cuTrainingX, cuTrainingY)
predicted = knn.predict(cuTestX)
predicted = predicted.to_pandas().to_numpy().astype("bool")

print("\nKNN Metrics:")
printMetrics(getMetrics(getConfusionMatrix(cuTestY, predicted)))



#! KNN CON K = SQRT(N) (N = Numero di sample)
knnSQRT = KNeighborsClassifier(n_neighbors=int(math.sqrt(len(cuTrainingX))))
knnSQRT.fit(cuTrainingX, cuTrainingY)

predicted = knnSQRT.predict(cuTestX)

predicted = predicted.to_pandas().to_numpy().astype("bool")

print("\nKNN SQRT Metrics:")
printMetrics(getMetrics(getConfusionMatrix(cuTestY, predicted)))

#! -----------------------------------
