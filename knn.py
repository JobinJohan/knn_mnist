import numpy as np
import matplotlib.pyplot as plt
import operator
from scipy.special import expit
from collections import Counter
import sys
import getopt


# Implementation of manhattan distance
def manhattanDistance(vector1, vector2):
    return np.sum(np.absolute(vector1 - vector2))

# Condensing algorithm that reduces the training set in order to keep only relevant data


def condensedTrainingSet(trainingSet, trainLabels, typeOfDistance):
    newSet = []
    newSetLabels = []
    newSet.append(trainingSet[0])
    newSetLabels.append(trainLabels[0])

    trainingSet[1:, :]
    trainLabels[1:]

    for indexLabel1, xi in enumerate(trainingSet):
        shortestDistance = None
        labelShortestDistance = ""
        for indexLabel, yi in enumerate(newSet):
            # Euclidian distance
            if typeOfDistance == 0:
                computedDistance = np.linalg.norm(xi - yi)
            # Manhattan distance
            elif typeOfDistance == 1:
                computedDistance = manhattanDistance(xi - yi)

            if shortestDistance == None or computedDistance < shortestDistance:
                shortestDistance = computedDistance
                labelShortestDistance = newSetLabels[indexLabel]

        if labelShortestDistance != trainLabels[indexLabel1]:
            newSet.append(xi)
            newSetLabels.append(trainLabels[indexLabel1])
    return [newSet, newSetLabels]


def getKClosestNeighbors(trainingSet, testSet, trainingLabels, testLabels, k, typeOfDistance):
    correctlyClassified = 0
    for indexTest, testDigit in enumerate(testSet):
        allDistancesFromOneTest = []
        for indexTraining, trainingDigit in enumerate(trainingSet):
            # Euclidian distance
            if typeOfDistance == 0:
                computedDistance = np.linalg.norm(testDigit - trainingDigit)
            # Manhattan distance
            elif typeOfDistance == 1:
                computedDistance = manhattanDistance(testDigit, trainingDigit)
            allDistancesFromOneTest.append([computedDistance, trainingLabels[indexTraining]])
        kClosestNeighbors = [i[1] for i in sorted(allDistancesFromOneTest)[:k]]
        predictedLabel = Counter(kClosestNeighbors).most_common(1)[0][0]
        if(predictedLabel == testLabels[indexTest]):
            correctlyClassified += 1
    print("Accuracy:", (correctlyClassified / len(testSet)) * 100)
    print("End of program")


def main(argv):
    # Get arguments from command line
    try:
        opts, args = getopt.getopt(argv, "t:T:k:d:", ["trainingSet=", "TestSet=", "kClosestNeigbors=", "distanceFunction"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-t", "--trainingSet"):
            trainingSet = arg
        elif opt in ("-T", "--TestSet"):
            testSetArg = arg
        elif opt in ("-k", "--kClosestNeigbors"):
            k = int(arg)
        elif opt in ("-d", "--distanceFunction"):
            d = int(arg)

    # Load data
    trainSet = np.genfromtxt(trainingSet, delimiter=",")
    trainSet = trainSet[:, 1:]
    testSet = np.genfromtxt(testSetArg, delimiter=",")
    testSet = testSet[:, 1:]
    trainLabels = np.genfromtxt(trainingSet, delimiter=",", usecols=(0), dtype=int)
    testLabels = np.genfromtxt(testSetArg, delimiter=",", usecols=(0), dtype=int)
    condensedTrainSet = condensedTrainingSet(trainSet, trainLabels, d)
    getKClosestNeighbors(condensedTrainSet[0], testSet, condensedTrainSet[1], testLabels, k, d)


if __name__ == "__main__":
    main(sys.argv[1:])
