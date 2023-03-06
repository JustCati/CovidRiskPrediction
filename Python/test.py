import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
cuTrainY = cuTrainingY.to_pandas().to_numpy().astype("bool")

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
    
    predictionsTrain = rf.predict(cuTrainingX)
    predictionsTest = rf.predict(cuTestX)

    predictionsTrain = predictionsTrain.to_pandas().to_numpy().astype("bool")
    predictionsTest = predictionsTest.to_pandas().to_numpy().astype("bool")

    metricsTest = getMetrics(getConfusionMatrix(cuTestY, predictionsTest))
    metricsTrain = getMetrics(getConfusionMatrix(cuTrainY, predictionsTrain))
    
    print("\nRF metrics (GPU) on Training Set:")
    printMetrics(metricsTrain)
    print("\nRF metrics (GPU) on Test Set:")
    printMetrics(metricsTest)
    
#! -----------------------------------


if 0:
#! ------------- NAIVE BAYES GAUSSIAN ------------------

    nb = GaussianNB()
    nb.fit(trainingX, trainingY)
    
    predictionsTrain = nb.predict(trainingX)
    predictionTest = nb.predict(testX)
    
    metricsTest = getMetrics(getConfusionMatrix(testY, predictionTest))
    metricsTraing = getMetrics(getConfusionMatrix(trainingY, predictionsTrain))

    print("\nNB Gaussian metrics on Training Set:")
    printMetrics(metricsTraing)
    print("\nNB Gaussian metrics on Test Set:")
    printMetrics(metricsTest)

#! -----------------------------------

if 0:
#! ------------- NAIVE BAYES MULTINOMIAL ------------------

    nb = MultinomialNB()
    nb.fit(trainingX, trainingY)
    
    predictionsTrain = nb.predict(trainingX)
    predictionTest = nb.predict(testX)
    
    metricsTest = getMetrics(getConfusionMatrix(testY, predictionTest))
    metricsTraing = getMetrics(getConfusionMatrix(trainingY, predictionsTrain))

    print("\nNB Multinomial metrics on Training Set:")
    printMetrics(metricsTraing)
    print("\nNB Multinomial metrics on Test Set:")
    printMetrics(metricsTest)

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
    
    predictionsTrain = svm.predict(cuTrainingX)
    predictionsTest = svm.predict(cuTestX)
    
    predictionsTrain = predictionsTrain.to_pandas().to_numpy().astype("bool")
    predictionsTest = predictionsTest.to_pandas().to_numpy().astype("bool")

    metricsTrain = getMetrics(getConfusionMatrix(cuTrainY, predictionsTrain))
    metricsTest = getMetrics(getConfusionMatrix(cuTestY, predictionsTest))

    print("\nLinear SVM metrics (GPU) on Training Set:")
    printMetrics(metricsTrain)
    print("\nLinear SVM metrics (GPU) on Test Set:")
    printMetrics(metricsTest)

#! -----------------------------------

if 0:
#! ------------- SVM ------------------

    results = dict()
    kernels = ["poly", "rbf", "sigmoid"]

    for kernel in kernels:
        svm = cuSVC(kernel=kernel, max_iter=100000)
        svm.fit(cuTrainingX, cuTrainingY)
    
        predictionsTrain = svm.predict(cuTrainingX)
        predictionsTest = svm.predict(cuTestX)
        
        predictionsTrain = predictionsTrain.to_pandas().to_numpy().astype("bool")
        predictionsTest = predictionsTest.to_pandas().to_numpy().astype("bool")

        metricsTrain = getMetrics(getConfusionMatrix(cuTrainY, predictionsTrain))
        metricsTest = getMetrics(getConfusionMatrix(cuTestY, predictionsTest))
        
        print("\n SVM con kernel '" + kernel + "' metrics on Training Set: ")
        printMetrics(metricsTrain)
        print("\n SVM con kernel '" + kernel + "' metrics on Test Set: ")
        printMetrics(metricsTest)

#! -----------------------------------

if 1:
#! ------------- LOGISTIC REGRESSION ------------------
    
    totalMetrics = dict()
    solvers = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
    
    for solver in solvers:
        lr = LogisticRegression(max_iter=100000, solver=solver)
        lr.fit(trainingX, trainingY)
        
        predictionsTrain = lr.predict(trainingX)
        predictionsTest = lr.predict(testX)
    
        metricsTest = getMetrics(getConfusionMatrix(testY, predictionsTest))
        metricsTrain = getMetrics(getConfusionMatrix(trainingY, predictionsTrain))
        
        totalMetrics[solver] = (metricsTrain, metricsTest)
    
        print("\nLogistic Regression metrics (solver = \"" + solver + "\") on Training Set:")
        printMetrics(metricsTrain)
        print("\nLogistic Regression metrics (solver = \"" + solver + "\") on Test Set:")
        printMetrics(metricsTest)
        
    matplotlib.use("GTK3Agg")
    plt.plot(solvers, [totalMetrics[solver][0]["accuracy"] for solver in solvers], label="Training Set", marker="o")
    plt.show()

    
#! ------------------------------------------------

