import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def zeroRClassifier(trainData, targetAttribute):
    mostFrequentClass = trainData[targetAttribute].mode()[0]
    return mostFrequentClass

def oneRClassifier(trainData, targetAttribute):
    bestFeature = None
    bestRule = {}
    bestAccuracy = 0

    for feature in trainData.drop(columns=[targetAttribute]):
        featureRules = {}
        for value in trainData[feature].unique():
            mostFrequentClass = trainData[trainData[feature] == value][targetAttribute].mode()[0]
            featureRules[value] = mostFrequentClass

        predictions = trainData[feature].map(featureRules)
        accuracy = np.mean(predictions == trainData[targetAttribute])

        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestFeature = feature
            bestRule = featureRules

    return bestFeature, bestRule

def evaluateModels(datasetPath, targetAttribute):
    data = pd.read_csv(datasetPath)

    trainData, testData = train_test_split(data, test_size=0.3, random_state=42)

    zeroRPrediction = zeroRClassifier(trainData, targetAttribute)
    

    oneRFeature, oneRRules = oneRClassifier(trainData, targetAttribute)

    oneRTrainPredictions = trainData[oneRFeature].map(oneRRules)
    oneRTrainAccuracy = np.mean(oneRTrainPredictions == trainData[targetAttribute])
    oneRTestPredictions = testData[oneRFeature].map(oneRRules)
    oneRTestAccuracy = np.mean(oneRTestPredictions == testData[targetAttribute])

    print("Training Dataset:")
    print(trainData)
    print("\nTesting Dataset:")
    print(testData)

    print("\nZero-R:")
    print(f"  Predición por todas las instancias: {zeroRPrediction}")


    print("One-R:")
    print(f"-->Reglas: {oneRRules}")
    print(f"--Training Accuracy: {100*oneRTrainAccuracy}%")
    print(f"--Testing Accuracy: {100*oneRTestAccuracy}%")

evaluateModels('cars.csv', 'Clase')