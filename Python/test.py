import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB


import cudf as cu
from cuml.svm import SVC as cuSVC, LinearSVC as cuLinearSVC
from cuml.model_selection import train_test_split as cuTrainTestSplit
from cuml import RandomForestClassifier as cuRandomForestClassifier


#! ------ CALCULATE METRICS -------
def getConfusionMatrix(labels: np.ndarray, predictions: np.ndarray):
    confusion = dict()
    confusion["TP"] = np.sum(labels & predictions)
    confusion["TN"] = np.sum(~labels & ~predictions)
    confusion["FP"] = np.sum(~labels & predictions)
    confusion["FN"] = np.sum(labels & ~predictions)
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

#! ----------- CPU ---------------------

covid = pd.read_parquet('covidClean.parquet')

toRemove = ["PATIENT_ID", "USMER", "SYMPTOMS_DATE",
            "MEDICAL_UNIT", "ADMISSION_DATE", "PATIENT_TYPE",
            "DEATH_DATE", "ORIGIN_COUNTRY"]
covid = covid.drop(columns = toRemove)
covid = covid.drop(columns= ["DIED", "INTUBED", "ICU"])

trainingSet, testSet = train_test_split(covid, test_size = 0.25)

trainingY = np.array(trainingSet["AT_RISK"])
trainingX = trainingSet.drop(columns = ["AT_RISK"])

testY = np.array(testSet["AT_RISK"])
testX = testSet.drop(columns = ["AT_RISK"])


#! ------------- GPU ------------------

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

#! -----------------------------------


if 0:
#! ------------- CART ---------------- 

#? ------------- CART FILE -----------
    pass
    
#! -----------------------------------

if 0:
#! ------------- RF GPU -----------------
    rf = cuRandomForestClassifier(n_estimators=1000)

    rf.fit(cuTrainingX, cuTrainingY)
    predictions = rf.predict(cuTestX)

    cuTestY = cuTestY.to_pandas().to_numpy().astype("bool")
    predictions = predictions.to_pandas().to_numpy().astype("bool")

    metrics = getMetrics(getConfusionMatrix(cuTestY, predictions))
    print("\nRF metrics (GPU):")
    printMetrics(metrics)
#! -----------------------------------


if 0:
#! ------------- NAIVE BAYES GAUSSIAN ------------------

    nb = GaussianNB()
    
    nb.fit(trainingX, trainingY)
    predictions = nb.predict(testX)
    
    metrics = getMetrics(getConfusionMatrix(testY, predictions))

    print("\nNB Gaussian metrics:")
    printMetrics(metrics)

#! -----------------------------------

if 0:
#! ------------- NAIVE BAYES MULTINOMIAL ------------------

    nb = MultinomialNB()
    
    nb.fit(trainingX, trainingY)
    predictions = nb.predict(testX)

    metrics = getMetrics(getConfusionMatrix(testY, predictions))

    print("\nNB Multinomial metrics:")
    printMetrics(metrics)

#! -----------------------------------


if 0:
#! ------------- KNN ------------------

#? ------------- KNN FILE -----------
    pass

#! -----------------------------------


if 0:
#! ------------- LINEAR SVM ------------------

    svm = cuLinearSVC(max_iter=100000)

    svm.fit(cuTrainingX, cuTrainingY)
    predictions = svm.predict(cuTestX)
    
    predictions = predictions.to_pandas().to_numpy().astype("bool")

    metrics = getMetrics(getConfusionMatrix(cuTestY, predictions))

    print("\nLinear SVM metrics (GPU):")
    printMetrics(metrics)

#! -----------------------------------

if 1:
#! ------------- SVM ------------------

    results = dict()
    kernels = ["poly", "rbf", "sigmoid"]

    for kernel in kernels:
        svm = cuSVC(kernel=kernel, max_iter=10)
        svm.fit(cuTrainingX, cuTrainingY)
        predictions = svm.predict(cuTestX)
        
        predictions = predictions.to_pandas().to_numpy().astype("bool")

        metrics = getMetrics(getConfusionMatrix(cuTestY, predictions))
        print("\n SVM con kernel '", kernel, "' metrics: ")
        printMetrics(metrics)

#! -----------------------------------
