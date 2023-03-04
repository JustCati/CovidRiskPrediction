import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score

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
#! ---------------------------------

covid = pd.read_parquet('covidClean.parquet')

toRemove = ["PATIENT_ID", "USMER", "SYMPTOMS_DATE",
            "MEDICAL_UNIT", "ADMISSION_DATE", "PATIENT_TYPE",
            "DEATH_DATE", "ORIGIN_COUNTRY"]
covid = covid.drop(columns = toRemove)
covid = covid.drop(columns= ["DIED", "INTUBED", "ICU"])

trainingSet, testSet = train_test_split(covid, test_size = 0.25)

trainingY = trainingSet["AT_RISK"]
trainingX = trainingSet.drop(columns = ["AT_RISK"])

testY = testSet["AT_RISK"]
testX = testSet.drop(columns = ["AT_RISK"])


#! ------------- KNN ------------------

k_values = range(150, 1000, 5)
results = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(trainingX, trainingY)
    predicted = knn.predict(testX)
    results.append(getMetrics(getConfusionMatrix(testY, predicted))["accuracy"])

plt.figure(figsize=(10, 5))
plt.plot(k_values, results)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show()
exit()

knn.fit(trainingX, trainingY)
predictions = knn.predict(testX)

metric = getMetrics(getConfusionMatrix(testY, predictions))

print("\nKNN metrics:")
print("Accuracy: \t", metric["accuracy"])
print("Precision: \t", metric["precision"])
print("Recall: \t", metric["recall"])
print("F1: \t\t", metric["f1"])

#! -----------------------------------

