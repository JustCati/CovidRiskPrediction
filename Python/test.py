import pandas as pd
from sklearn.tree._tree import TREE_LEAF
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


def prune_index(inner_tree, index, threshold):
    if inner_tree.value[index].min() < threshold:
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)

cart = DecisionTreeClassifier()

prune_index(cart.tree_, 0, 0.1)

cart.fit(trainingX, trainingY)
predictions = cart.predict(testX)

metric = getMetrics(getConfusionMatrix(testY, predictions))
print("CART metrics:")
print("Accuracy: \t", metric["accuracy"])
print("Precision: \t", metric["precision"])
print("Recall: \t", metric["recall"])
print("F1: \t\t", metric["f1"])

#! -----------------------------------


#! ------------- RF ------------------

# rf = RandomForestClassifier(n_estimators=2000, n_jobs=-1)

# rf.fit(trainingX, trainingY)
# predictions = rf.predict(testX)

# metric = getMetrics(getConfusionMatrix(testY, predictions))
# print("\nRF metrics:")
# print("Accuracy: \t", metric["accuracy"])
# print("Precision: \t", metric["precision"])
# print("Recall: \t", metric["recall"])
# print("F1: \t\t", metric["f1"])

#! -----------------------------------
