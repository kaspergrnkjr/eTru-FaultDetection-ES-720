import csv
import os
import numpy as np
import random
from sklearn.datasets import make_blobs
import warnings

def load_train_test_labels(train_fault_types, test_fault_types, concatenation, output_type, useTestData=False, fraction=1, binary=False):
    """
    :param train_fault_types: desired faults, (-1) for all
    :param test_fault_types: desired faults, (-1) for all
    :param concatenation: True if initial conditions should not be separated.
    :param output_type: 'array' for numpy array or 'list'
    :param equalSamplesInClass: True for equal samples in each class.
    :param useTestData: True if test data should be used
    :param balancing: faulty/non-faulty
    :param binary: binary labeling (true or false)
    :return: [trainData, trainLabels, testData, testLabels]
    """

    if useTestData:
        # How many initial conditions are there
        numberOfInitialConditions = 18
        trainData = []
        trainLabel = []
        testData = []
        testLabel = []
        nonbinlbl = []
        if train_fault_types == -1:
            train_fault_types = list(range(0, 21))
        if test_fault_types == -1:
            test_fault_types = list(range(0, 21))

        if type(test_fault_types) == int:
            test_fault_types = [test_fault_types]

        # download all the training files in the set
        for fault in train_fault_types:
            for j in range(numberOfInitialConditions):
                path = os.path.join('data', 'Data' + str(fault),
                                    'Faulttype' + str(fault) + '_initialcondition' + str(j + 1) + '.csv')
                file = open(path)
                csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
                rows = []
                for row in csvreader:
                    rows.append(row)
                file.close()

                if concatenation:
                    # Concatenate the initial conditions
                    trainData += rows
                    trainLabel += [fault] * len(rows)
                else:
                    # If concatenation is not desired, just append to the list
                    trainData.append(rows)
                    trainLabel.append([fault] * len(rows))
        trainData, trainLabel = unison_shuffled_copies(trainData, trainLabel)
        trainData, trainLabel, _ = balancingData(fraction, trainData, trainLabel, binary=binary)
        for fault in test_fault_types:
            tr, te = loadTestData(250, [fault], binary=False)

            if concatenation:
                # Concatenate the initial conditions
                testData += tr[0]
                testLabel += te[0]
            else:
                # If concatenation is not desired, just append to the list
                testData.append(tr)
                testLabel.append(te)

        testData, testLabel = unison_shuffled_copies(testData, testLabel)
        nonbinlbl = testLabel.copy()
        if binary:
            # Set labels as binary
            idxs = np.argwhere(testLabel > 0)
            testLabel[idxs] = 1
        if output_type == 'array' or output_type == 'Array':
            return np.array(trainData), np.array(trainLabel), np.array(testData), np.array(testLabel), np.array(nonbinlbl)
        elif output_type == 'list' or output_type == 'List':
            return trainData, trainLabel, testData, testLabel, nonbinlbl
        else:
            return 0
    else:
        trainData, trainLabel, validationData, validationLabel = load_PercentageData(test_fault_types, 80, True, True, binary=False)
        trainData, trainLabel = unison_shuffled_copies(trainData, trainLabel)
        validationData, validationLabel = unison_shuffled_copies(validationData, validationLabel)
        trainData, trainLabel, _ = balancingData(fraction, trainData, trainLabel, binary=binary)
        nonbinlbl = validationLabel.copy()
        if binary:
            # Set labels as binary
            idxs = np.argwhere(validationLabel > 0)
            validationLabel[idxs] = 1

        #validationData, validationLabel, nonbinlbl = balancingData(fraction, validationData, validationLabel, binary=binary)
        if output_type == 'array' or output_type == 'Array':
            return np.array(trainData), np.array(trainLabel), np.array(validationData), np.array(validationLabel), np.array(nonbinlbl)
        elif output_type == 'list' or output_type == 'List':
            return trainData, trainLabel, validationData, validationLabel, nonbinlbl
        else:
            return 0


def balancingData(fraction, data, label, binary):
    classes, counts = np.unique(np.array(label), return_counts=True)
    if binary:
        if len(np.where(classes == 0)[0]) == 0:
            raise Exception("No non-faulty data selected")
        n_nonFaulty = counts[np.argwhere(classes == 0)]
        maxFraction = sum(counts[np.argwhere(classes != 0)]) / n_nonFaulty
        if fraction > maxFraction:
            warnings.warn("Desired fraction could not be used. Fraction used: ", DeprecationWarning, stacklevel=2)
            print("Fraction could not be used, the fraction used is: " + str(maxFraction))
            fractionUsed = maxFraction
        else:
            fractionUsed = fraction

        numberOfFaultyInEachClass = (n_nonFaulty*fractionUsed)/(len(classes)-1)
        numberOfFaultyInEachClass = round(numberOfFaultyInEachClass[0][0])
        outputData = []
        outputLabel = []
        occurrences = [0] * (len(classes)-1)
        nonbinlbl = []
        for idx, sample in enumerate(data):
            lbl = label[idx]
            if lbl == 0:
                outputData.append(sample)
                outputLabel.append(lbl)
                nonbinlbl.append(lbl)
                continue
            idxOccurrences = (np.argwhere(classes == lbl) - 1)[0][0]
            if occurrences[idxOccurrences] < numberOfFaultyInEachClass:
                outputData.append(sample)
                outputLabel.append(1)
                nonbinlbl.append(lbl)
                occurrences[idxOccurrences] += 1

    else:
        numberOfFaultyInEachClass = min(counts)
        outputData = []
        outputLabel = []
        occurrences = [0] * len(classes)
        nonbinlbl = []
        for idx, sample in enumerate(data):
            lbl = label[idx]
            idxOccurrences = np.argwhere(classes == lbl)[0][0]
            if occurrences[idxOccurrences] < numberOfFaultyInEachClass:
                outputData.append(sample)
                outputLabel.append(lbl)
                nonbinlbl.append(lbl)
                occurrences[idxOccurrences] += 1
    return outputData, outputLabel, nonbinlbl


def load_file(fault_type, initial_condition, concatenation, output_type):
    """
    :param fault_type: Integer representing the desired fault.
    :param initial_condition: Integer representing the desired initial condition. Zero if all conditions are desired.
    :param concatenation: True if initial conditions should not be separated.
    :param output_type: 'array' for numpy array or 'list'
    :return: Loaded data
    """
    # How many initial conditions are there
    numberOfInitialConditions = 18
    outputFile = []

    # If initial condition is not 0 then return the desired initial condition
    if initial_condition != 0:

        path = os.path.join('data', 'Data' + str(fault_type), 'Faulttype' + str(fault_type) + '_initialcondition' + str(initial_condition) + '.csv')
        file = open(path)
        csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        rows = []
        for row in csvreader:
            rows.append(row)
        file.close()
        outputFile = rows

    else:
        if fault_type == 9:
            numberOfInitialConditions = 15
        # download all the files in the set
        for i in range(numberOfInitialConditions):
            path = os.path.join('data', 'Data' + str(fault_type), 'Faulttype' + str(fault_type) + '_initialcondition' + str(i + 1) + '.csv')
            file = open(path)
            csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
            rows = []
            for row in csvreader:
                rows.append(row)
            file.close()

            if concatenation:
                # Concatenate the initial conditions
                outputFile += rows
            else:
                # If concatenation is not desired, just append to the list
                outputFile.append(rows)
    if output_type == 'array' or output_type == 'Array':
        return np.array(outputFile)
    elif output_type == 'list' or output_type == 'List':
        return outputFile
    else:
        return 0


def loadDataAndLabels(numTrainSamples, labelType, desiredFaults):

    """

    :param numTrainSamples: Number of desired training samples. The rest will be returned as test data.
    :param labelType: 'full' for fault type distinction, 'binary' for faulty/non-faulty distinction
    :param desiredFaults: An array consisting of the desired faults
    :return: [trainData, trainLabels, testData, testLabels]
    """

    trainData = []
    trainLabels = []
    testData = []
    testLabels = []

    for fault in desiredFaults:
        data = load_file(fault, 0, False, 'array')
        totalNumberOfSamples = len(data[0])
        numberOfTestSamples = totalNumberOfSamples - numTrainSamples
        if numberOfTestSamples <= 0:
            raise Exception('Number of training samples exeeds total number of samples in set')
        testIdx = random.sample(range(totalNumberOfSamples), numberOfTestSamples)
        for condition in data:
            testData.append(condition[testIdx])
            testLabels += [fault] * len(testIdx)
            condition = np.delete(condition, testIdx, 0)
            trainData.append(condition)
            trainLabels += [fault] * len(condition)

    if labelType == 'binary' or labelType == 'Binary':
        trainLabels[:] = [x if x == 0 else 1 for x in trainLabels]
        testLabels[:] = [x if x == 0 else 1 for x in testLabels]

    trainData = np.vstack(np.array(trainData))
    testData = np.vstack(np.array(testData))

    return [trainData, trainLabels, testData, testLabels]


def load_PercentageData(faultypes, Percentage, trainData = None, validationData = None, binary=False):
    """
    :param Percentage: Number of desired training samples. The rest will be returned as validation and test data.
    :param trainData: Is training data desired?
    :param validationData: Is validation data desired?
    :param binary: if binary data is desired
    :return: [trainData, trainLabels, validationData,ValidationLabel testData, testLabels]
    """

    safeCheck = [60, 80, 96]
    outputData = [[], []]
    outputLabels = [[], []]
    _numberOfFaulttype = 21
    if Percentage not in safeCheck:
        raise ValueError('Not a correct percentage!')

    for fault in faultypes:
        if trainData:
            path = os.path.join('PercentageWiseData', 'Faulttype' + str(fault), 'Percentage_' + str(Percentage), 'Train_' + str(Percentage) + '.csv')
            with open(path) as file:
                csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
                rows = []
                for row in csvreader:
                    rows.append(row)
                rows = np.array(rows)
                outputData[0].extend(rows[:, 0:-2])
                if binary:
                    if fault == 0:
                        outputLabels[0].extend(rows[:, -1])
                    else:

                        outputLabels[0].extend([1] * len(rows))
                else:
                    outputLabels[0].extend(rows[:, -1])

        if validationData:
            path = os.path.join('PercentageWiseData', 'Faulttype' + str(fault), 'Percentage_' + str(Percentage), 'Val_' + str(Percentage) + '.csv')
            with open(path) as file:
                csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
                rows = []
                for row in csvreader:
                    rows.append(row)
                rows = np.array(rows)
                outputData[1].extend(rows[:, 0:-2])
                if binary:
                    if fault == 0:
                        outputLabels[1].extend(rows[:, -1])
                    else:

                        outputLabels[1].extend([1] * len(rows))
                else:
                    outputLabels[1].extend(rows[:, -1])

    return outputData[0], outputLabels[0], outputData[1], outputLabels[1]

# Create a ToyExample:


def createNewBlobs():
    classes = 20
    samples = 1000*classes
    dimension = 10
    X, y, centers = make_blobs(samples, centers=classes, n_features=dimension, cluster_std=1, center_box=(-6, 6), shuffle=True, return_centers=True)
    data = np.column_stack([X, y])
    np.savetxt('ToyExampleBig.csv', data, delimiter=',')


def loadTestData(dataPoints, test_fault_types,binary=False):
    classes = 21
    outputData = [[],[]]
    outputLabels = [[], []]
    if type(test_fault_types) == int:
        test_fault_types = [test_fault_types]
    if test_fault_types == 'all':
        for i in range(classes):
            if i == 0:
                path = os.path.join('newValid', 'Newvalid_n' + str(i + 1) + '.csv')
            else:
                path = os.path.join('newValid', 'Newvalid_f' + str(i) + '.csv')
            with open(path) as file:
                csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
                rows = []
                heading = next(file)
                row_count = sum(1 for row in csvreader)
            with open(path) as file:
                csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
                heading = next(file)
                rowNumber = 0
                for row in csvreader:
                    if rowNumber > row_count - dataPoints - 1:
                        rows.append(row)
                    rowNumber += 1
            rows = np.array(rows)
            outputData[0].extend(rows)
            if binary==False:
                outputLabels[0] += [i] * len(rows)
            else:
                k = 0 if i == 0 else 1
                outputLabels[0] += [k] * len(rows)

    else:
        for fault in test_fault_types:
            if fault == 0:
                path = os.path.join('newValid', 'Newvalid_n' + str(fault + 1) + '.csv')
            else:
                path = os.path.join('newValid', 'Newvalid_f' + str(fault) + '.csv')
            with open(path) as file:
                csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
                rows = []
                heading = next(file)
                row_count = sum(1 for row in csvreader)
            with open(path) as file:
                csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
                heading = next(file)
                rowNumber = 0
                for row in csvreader:
                    if rowNumber > row_count-dataPoints-1:
                        rows.append(row)
                    rowNumber += 1

            outputData[0] += rows
            if binary == False:
                outputLabels[0] += [float(fault)] * len(rows)
            else:
                k = 0 if fault == 0 else 1
                outputLabels[0] += [float(k)] * len(rows)
    return outputData, outputLabels


def swapData(data, label, valData, valLabel):
    nonfaulty_idx = np.argwhere(np.array(label) == 0)
    data = np.array(data)
    tr12 = data[nonfaulty_idx, 12].copy()
    tr13 = data[nonfaulty_idx, 13].copy()
    data[nonfaulty_idx, 12] = tr13
    data[nonfaulty_idx, 13] = tr12
    # trainData = trainData.tolist()

    nonfaulty_idx1 = np.argwhere(np.array(valLabel) == 0)
    valData = np.array(valData)
    te12 = valData[nonfaulty_idx1, 12].copy()
    te13 = valData[nonfaulty_idx1, 13].copy()
    valData[nonfaulty_idx1, 12] = te13
    valData[nonfaulty_idx1, 13] = te12
    # bitzerValData = bitzerValData.tolist()
    return data, label, valData, valLabel


def removeThe3features(data, valData):
    data = np.delete(data, (1, 10, 11), 1)
    valData = np.delete(valData, (1, 10, 11), 1)
    return data, valData


def michaelsDummeFunction(bitzerData, data, label, valData, valLabel):
    if bitzerData == 1:
        trainDataPlotting = data
        trainLabelPlotting = label
        validationData = valData
        validationLabel = valLabel
    else:
        rows = []
        with open("ToyExampleBig.csv") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                rows.append(row)
            rows = np.array(rows)
            rows = rows.astype(np.float64)
            data = rows[0:-101, 0:-2]
            labels = rows[0:-101, -1]
            testData = rows[-100:-1, 0:-2]
            testLabel = rows[-100:-1, -1]

        data = np.array(data, dtype=float)
        labels = np.array(labels)
        trainDataPlotting = data
        trainLabelPlotting = labels
        validationData = testData
        validationLabel = testLabel

    return trainDataPlotting, trainLabelPlotting, validationData, validationLabel


def unison_shuffled_copies(a, b):
    a = np.array(a)
    b = np.array(b)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
