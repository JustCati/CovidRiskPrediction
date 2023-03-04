import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

#! ------------- CART ---------------- 

cart = DecisionTreeClassifier()

path = cart.cost_complexity_pruning_path(trainingX, trainingY)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[ccp_alphas >= 0]

dimChunks = 1000
chunks = np.split(ccp_alphas, [i for i in range(dimChunks, len(ccp_alphas), dimChunks)])

accuracyTest, accuracyTrain = [], []
def getTreePerf(ccp_alpha):
    clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
    clf.fit(trainingX, trainingY)
    predictionTrain = clf.predict(trainingX)
    predictionTest = clf.predict(testX)
    accuracyTrain.append(getMetrics(getConfusionMatrix(trainingY, predictionTrain))["accuracy"])
    accuracyTest.append(getMetrics(getConfusionMatrix(testY, predictionTest))["accuracy"])
    return

for chunk in chunks:
    joblib.Parallel(n_jobs=-1, verbose=10)(joblib.delayed(getTreePerf)(ccp_alpha) for ccp_alpha in chunk)

plt.figure(figsize=(10, 5))
plt.plot(ccp_alphas, accuracyTrain, marker='o', label="train", drawstyle="steps-post")
plt.plot(ccp_alphas, accuracyTest, marker='o', label="test", drawstyle="steps-post")
plt.xlabel("alpha")
plt.ylabel("accuracy")
plt.legend()
plt.show()
exit(1)

cart.fit(trainingX, trainingY)
predictions = cart.predict(testX)

metric = getMetrics(getConfusionMatrix(testY, predictions))
print("CART metrics:")
print("Accuracy: \t", metric["accuracy"])
print("Precision: \t", metric["precision"])
print("Recall: \t", metric["recall"])
print("F1: \t\t", metric["f1"])
#! -----------------------------------