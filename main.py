import loadfile as lf
import ML
import numpy as np
import matplotlib.pyplot as plt
import loadfile as lf
import winsound
import Plotter
import time

"""
Options for solver selection:
    - LDA       ---> LDA dimensionality reduction & SVM classification.
    - PCA       ---> PCA dimensionality reduction & SVM classification.
    - none      ---> SVM classification.
    - MLP       ---> MLP classification.


ACTIVATE_TWO_STAGE_ROCKET   ---> Determines whether or not two consecutive stages should be used.

Options for each stage:
    - Solver                ---> Chooses which dimensionality reduction method to use.
    - Dimensions            ---> Chooses what number of dimensions to reduce to. When solver is "none" or "MLP" then this parameter is ignorred.
    - UseBitzerData         ---> Chooses whether or not to use the Bitzer data or synthetic data.
    - rm1                   ---> Chooses whether or not to remove 3 features. (ambient temperature, density and power consumption).
    - binary                ---> Chooses whether or not the stage should train on binary data. That is faulty or non-faulty.
    - useTest               ---> Chooses whether or not the test data should be used. (Only for final tests).
    - fraction              ---> This parameter determines the balancing or the training data. (#non-faulty/#faulty).
    - gamma                 ---> This is the parameter for the gaussian kernel for the SVM classification.
    - C                     ---> This is the regularization parameter for the SVM classification.
    
Options for the MLP:
    - loadModel             ---> If this parameter is set to 1, then a saved model of the weights and biases are loaded.
"""



# For MLP load model
loadModel = 0

ACTIVATE_TWO_STAGE_ROCKET = 0
testTime = 0
timeInstances = []
accTest = []

########STAGE ONE################
solver1 = 'LDA'
Dimensions1 = 9
useBitzerData1 = 1
# remove 3 features (1)
rm1 = 0
# For binary classification (1)
binary1 = 0
# Zero for correct
swapData1 = 0
# Use test data (1)
useTest1 = 1
# Balancing fraction (1 for equal F NF)
fraction1 = 1
gamma1 = 0.01
C1 = 1000

########################STAGE TWO#################################
solver2 = 'none'
Dimensions2 = 9
useBitzerData2 = 1
# remove 3 features (1)
rm2 = 0
# For binary classification (1)
binary2 = 0
# Zero for correct
swapData2 = 0
# Use test data (1)
useTest2 = 1
# Balancing fraction (1 for equal F NF)
fraction2 = 1
gamma2 = 0.01
C2 = 1000

info = [swapData1, useTest1, binary1, ACTIVATE_TWO_STAGE_ROCKET, fraction1, rm1, swapData2, useTest2, binary2,
        fraction2, rm2, solver1, Dimensions1, Dimensions2]

predictedLabels = []

createNewBlobs = False
if createNewBlobs:
    lf.createNewBlobs()

if ACTIVATE_TWO_STAGE_ROCKET:
    print("Aaaaalright!")
    print("Engines firing up...")
    print("3")
    print("2")
    print("1")
    print("Take off")
testList = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20]
fullList = list(range(0, 21))
#testList = list(range(0,21))
trainData1, trainLabel1, bitzerValData1, bitzerValLabel1, nonbinlbl = lf.load_train_test_labels(testList, testList,
                                                                                                True, 'array',
                                                                                                useTestData=useTest1,
                                                                                                binary=binary1,
                                                                                                fraction=fraction1)





trainLabel1 = np.array(trainLabel1)
bitzerValLabel1 = np.array(bitzerValLabel1)
before_rm_bitzerValData1 = np.array(bitzerValData1)

if swapData1 == 1:
    trainData1, trainLabel1, bitzerValData1, bitzerValLabel1 = lf.swapData(trainData1, trainLabel1, bitzerValData1,
                                                                           bitzerValLabel1)


if rm1 == 1:
    trainData1, bitzerValData1 = lf.removeThe3features(trainData1, bitzerValData1)

trainDataPlotting1, trainLabelPlotting1, validationData1, validationLabel1 = lf.michaelsDummeFunction(useBitzerData1,
                                                                                                      trainData1,
                                                                                                      trainLabel1,
                                                                                                      bitzerValData1,
                                                                                                      bitzerValLabel1)

trainLabelPlotting1 = np.array(trainLabelPlotting1)

# Creates an instance of the fault detector class
# Load the model
if loadModel == 1 and solver1 == 'MLP':
    print('Loading model')
    w, b = ML.mlp_load_file('MLP')
    w = np.array(w[0])
    b = np.array(b[0])
    detector = ML.FaultDetector(data=trainDataPlotting1, w=w, b=b)
    detector.reducer = ML.MLP(w, b, 1)
    detector.reducer.a = b.copy()
    detector.reducer.z = b.copy()
else:
    detector = ML.FaultDetector(trainDataPlotting1)

# Performs the training with the specified method (PCA or LDA)
if solver1 != 'MLP' or loadModel != 1:
    detector.train(solver1, Dimensions1, trainLabelPlotting1, C=C1, gamma=gamma1, verbose=True)
if ACTIVATE_TWO_STAGE_ROCKET == 0:
    if testTime:
        predLabels = np.zeros(0)
        for idx, sample in enumerate(validationData1):
            startTime = time.time()
            predLabels = np.append(predLabels, detector.test(sample, validationLabel1[idx], info, predictionTable=False,testTime=testTime))
            endTime = time.time()
            timeInstances.append(endTime-startTime)
    else:
        detector.test(np.array(validationData1), np.array(validationLabel1), info, predictionTable=True)
    if solver1 == 'MLP' and loadModel != 1:
        save = input('Should the model be saved? (y/n)')
        if save == 'y':
            # Save the Model
            detector.save_model('MLP')


else:
    # Getting all predicted labels, real labels, testData sorted in uniform.
    if testTime:
        predLabels = np.zeros(0)
        for idx, sample in enumerate(validationData1):
            startTime = time.time()
            predLabels = np.append(predLabels, detector.test(sample, validationLabel1[idx], info, predictionTable=False,testTime=testTime))
            endTime = time.time()
            timeInstances.append(endTime-startTime)
    else:
        predLabels, sortingIdx, firstStatePredMatrix = detector.test(np.array(validationData1), np.array(validationLabel1), info, predictionTable=False,testTime=False)
        validationLabeltemp = validationLabel1[sortingIdx]
        valNonBin = nonbinlbl[sortingIdx]

        logicVec = (validationLabeltemp == 1) & (predLabels == 0)
        originalWrongClassifiedLabelVec = valNonBin[logicVec]
        for i in range(1, 21):
            firstStatePredMatrix[i][0] = len(np.argwhere(originalWrongClassifiedLabelVec == i))
        firstStatePredMatrix[1][1] = 0


if ACTIVATE_TWO_STAGE_ROCKET:
    fullList = list(range(1, 21))
    trainData2, trainLabel2, bitzerValData2, bitzerValLabel2, _ = lf.load_train_test_labels(fullList,
                                                                                            fullList, True,
                                                                                            'array',
                                                                                            useTestData=useTest2,
                                                                                            binary=binary2,
                                                                                            fraction=fraction2)

    if testTime:
        sortingWhenTestingTime = np.argsort(validationLabel1)
        predLabels = predLabels[sortingWhenTestingTime]
        testSet = np.array(before_rm_bitzerValData1)[sortingWhenTestingTime]
        timeInstances = np.array(timeInstances)
        timeInstances = timeInstances[sortingWhenTestingTime]
        validationLabel1 = validationLabel1[sortingWhenTestingTime]

        faults = np.argwhere(np.array(predLabels) == 1)
        faults = np.array([x[0] for x in faults])
        testSet = testSet[faults]
        testLabel = np.array(nonbinlbl)[sortingWhenTestingTime]
        testLabel = testLabel[faults]
    else:
        faults = np.argwhere(np.array(predLabels) == 1)
        faults = np.array([x[0] for x in faults])
        testSet = np.array(before_rm_bitzerValData1)[sortingIdx]
        testSet = testSet[faults]
        testLabel = np.array(nonbinlbl)[sortingIdx]
        testLabel = testLabel[faults]

    if swapData2 == 1:
        trainData2, trainLabel2, bitzerValData2, bitzerValLabel2 = lf.swapData(trainData2, trainLabel2, bitzerValData2,
                                                                               bitzerValLabel2)

    if rm2 == 1:
        trainData2, testSet = lf.removeThe3features(trainData2, testSet)

    trainDataPlotting2, trainLabelPlotting2, validationData2, validationLabel2 = lf.michaelsDummeFunction(
        useBitzerData2, trainData2, trainLabel2, bitzerValData2, bitzerValLabel2)
    detector2 = ML.FaultDetector(trainDataPlotting2)
    # Performs the training with the specified method (PCA or LDA)
    detector2.train(solver2, Dimensions2, trainLabelPlotting2, C=C2, gamma=gamma2)


    print(len(np.argwhere(testLabel == 0)))

    if testTime:
        for idx, sample in enumerate(testSet):
            startTime = time.time()
            accTest.append(detector2.test(sample, testLabel[idx], info, predictionTable=False, testTime=testTime))
            endTime = time.time()
            timeInstances[idx] += endTime-startTime
            timeInstances = np.array(timeInstances)
            sampleNumber = np.argmax(timeInstances)
            print(max(timeInstances), 'seconds for the highest sample')

    else:
        _, _, secondStatePredMatrix = detector2.test(testSet, testLabel, info, predictionTable=False, testTime=False)
        Plotter.confusionMatrixCombiner(firstStatePredMatrix, secondStatePredMatrix, info)

if testTime:
    timeInstances = np.array(timeInstances)
    sampleNumber = np.argmax(timeInstances)
    print(max(timeInstances), 'seconds for the highest sample')
winsound.Beep(500, 500)
