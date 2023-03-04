import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score

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
#! ---------------------------------

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

if 0:
#! ------------- CART ---------------- 

#? ------------- CART FILE -----------
    pass
    
#! -----------------------------------


if 0:
#! ------------- RF ------------------

    rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

    rf.fit(trainingX, trainingY)
    predictions = rf.predict(testX)

    metric = getMetrics(getConfusionMatrix(testY, predictions))
    print("\nRF metrics:")
    print("Accuracy: \t", metric["accuracy"])
    print("Precision: \t", metric["precision"])
    print("Recall: \t", metric["recall"])
    print("F1: \t\t", metric["f1"])

#! -----------------------------------


if 0:
#! ------------- NAIVE BAYES GAUSSIAN ------------------

    nb = GaussianNB()
    
    nb.fit(trainingX, trainingY)
    predictions = nb.predict(testX)
    
    metric = getMetrics(getConfusionMatrix(testY, predictions))

    print("\nNB Gaussian metrics:")
    print("Accuracy: \t", metric["accuracy"])
    print("Precision: \t", metric["precision"])
    print("Recall: \t", metric["recall"])
    print("F1: \t\t", metric["f1"])

#! -----------------------------------

if 0:
#! ------------- NAIVE BAYES MULTINOMIAL ------------------

    nb = MultinomialNB()
    
    nb.fit(trainingX, trainingY)
    predictions = nb.predict(testX)

    metric = getMetrics(getConfusionMatrix(testY, predictions))

    print("\nNB Multinomial metrics:")
    print("Accuracy: \t", metric["accuracy"])
    print("Precision: \t", metric["precision"])
    print("Recall: \t", metric["recall"])
    print("F1: \t\t", metric["f1"])

#! -----------------------------------


if 0:
#! ------------- KNN ------------------

#? ------------- KNN FILE -----------
    pass

#! -----------------------------------

