import scipy.linalg as sci
from sklearn import svm
import matplotlib.pyplot as plt
import Plotter as Pot
import random
from decimal import Decimal
import sklearn
import getpass
import numpy as np
from keras.utils import to_categorical
import warnings
from scipy.io import savemat
from scipy.io import loadmat

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class PCA:

    def __init__(self, trainData):
        """
        :param trainData: Training data. Can be list or numpy array
        """
        # Checks before saving the training data
        if type(trainData) is not np.ndarray:
            trainData = np.array(trainData)
        if len(trainData.shape) > 2:
            raise Exception("Training data may not be nested")
        else:
            self.trainData = trainData
        self.components = 0
        self.eigMatrix = []
        self.eigValues = []
        self.components = 0

    def reduce(self, dimensions):
        """
        :param dimensions: Number of desired dimension
        :return: None
        """
        self.components = dimensions
        # Calculate the covariance of the training set
        cov = np.cov(self.trainData.T)
        # Find eigenvalues and eigenvectors for the covariance matrix
        self.eigValues, eigVec = sci.eig(cov)
        # select the 'dimensions' eigenvectors which have the largest corresponding eigenvalue
        self.eigMatrix = np.array(eigVec[:, 0:dimensions]).T
        self.components = dimensions

    def transform(self, data):
        """
        :param data: Data to be transformed
        :return: Transformed data
        """
        # Calculate the mean of the training set
        mean = np.mean(self.trainData, axis=0)
        # Transform each data point
        Z = []
        for x in range(len(data)):
            z = (self.eigMatrix.dot(data[x].T - mean))
            Z.append(z)

        Z = np.array(Z)
        return -Z



class LDA:
    def __init__(self, n_components, solver="svd"):
        self.transformMatrix = None
        self.Y = None
        self.means = None
        self.Sb = None
        self.Sw = None
        self.tolerance = pow(10, -3)
        self.totalMean = None
        self.components = n_components
        self.solver = solver

    def computeMean(self, samples):
        self.totalMean = np.mean(samples, axis=0)  # Calculates the Mean of any given samples.
        return self.totalMean

    def computeClassMeans(self, samples, labels):
        uniLab, y_t = np.unique(labels, return_inverse=True)  # Find number of classes
        bincount = np.bincount(y_t)
        length = 0
        means = []
        for uniqueLabel in range(len(uniLab)):  # Calculates the mean of each ith classes
            if uniqueLabel < uniLab.max():
                sumOfClass = np.sum(samples[length:length + bincount[uniqueLabel], :],
                                    axis=0)  # Calculation of every class but the last one.
            else:
                sumOfClass = np.sum(samples[length:-1, :],
                                    axis=0)  # Calculating the class means of the last class.
            means.append(sumOfClass / bincount[uniqueLabel])
            length = length + bincount[uniqueLabel]
            self.means = means
        return means

    def computeSB(self, samples, lengthOfEachClass, n_classes):
        # Calculates the Scatter Between Classes (SB) matrix. SB is ONLY used to check Hb.T * Hb = SB, to see if HB is correct.
        N, dim = samples.shape
        Sb = 0
        M = np.mean(samples, axis=0)
        idx1 = 0
        for j in range(n_classes):
            idx2 = idx1 + lengthOfEachClass[j]
            Mc = np.mean(samples[idx1:idx2, :], axis=0) - M
            Sb = Sb + lengthOfEachClass[j] / N * np.outer(Mc, Mc)  # Formula for calculating SB
            self.Sb = Sb.copy()
            idx1 = idx2
        return Sb

    def computeHb(self, samples, labels):  # Calculating Hb
        # Here we find the unique number of labels
        uniLab, y_t = np.unique(labels, return_inverse=True)
        # Counts the number of elements in each class
        bincount = np.bincount(y_t)
        # Calculate the class means if it has not been done
        if (self.means == None):
            self.computeClassMeans(samples, labels)
        Hb = []
        for j in range(len(uniLab)):
            # Calculate sqrt(prior for class j)
            temp = np.sqrt(bincount[j] / len(labels))
            # Calculate Hb for each class
            Hb.append(temp * (self.means[j] - self.totalMean))
        Hb = np.array(Hb)
        # return Hb
        return Hb

    def computeSt(self,samples):  # Calculates the Total Scatter matrix (St). SB is ONLY used to check Ht.T * Ht = St, to see if Ht is correct.
        N, _ = samples.shape  # Finds N rows in samples
        if np.linalg.norm(self.totalMean) == None:
            self.computeMean(samples)  # Calculate the total mean and save it in self.totalMean
        samples = samples - self.totalMean  # Subtract the mean from the samples for every sample
        St = (1 / N) * samples.T @ samples  # We dive every number of the samples with the number of rows in samples
        return St

    def computeHt(self, samples):
        # Find the number of rows in samples
        N, _ = samples.shape
        # Formula for computing Ht.
        Ht = 1 / np.sqrt(N) * (samples - self.totalMean)
        return Ht

    def computeReducedSVD(self, samples, tolerance=pow(10, -6)):  # Calculating the reduced SVD with 2 different inputs, firstly the variable which the SVD should be performed upon, then the tolerance for the SVD.
        U, Sigma, Vh = np.linalg.svd(samples, full_matrices=False)  # Performs the SVD on "Variable".
        Temp = Sigma > tolerance  # Sort the singular values
        self.tolerance = tolerance
        Inverse_Sigma = np.diag(1 / Sigma[Temp])  # Calculates the Inverse Sigma while we are at it. NOTE: This is used for later purposes
        Sigma = np.diag(Sigma[Temp])  # Make a diagonal matrix with temp singular values
        V = Vh.T  # Transpose matrix V
        V = V[:, Temp]  # Only use Temp columns of V
        Vh = V.T  # Transpose V
        return U, Sigma, Vh, Inverse_Sigma

    def computeY(self, H_b, V, Inverse_Sigma):
        Y = H_b @ V @ Inverse_Sigma  # Computing the Y Matrix from previous SVD of Ht.
        self.Y = Y
        return Y
    def sortTheData(self,samples,labels):
        n_classes = np.unique(labels)
        Xd = []
        Xl = []
        for i in range(len(n_classes)):
            Xd.append(samples[labels == i])
            Xl.append(labels[labels == i])
        Xd = np.array(Xd)
        Xd = np.concatenate(Xd, axis=0)
        Xl = np.array(Xl)
        Xl = np.concatenate(Xl, axis=0)
        return Xd, Xl

    def fit(self, samples, labels):  # Compute the A matrix.
        if len(samples) == self.components:  # Ensures that N-1 Dimensions is the largest possible.
            raise ValueError("The number of samples is equal to the number of components, that's a nono! :)")
        # Checks if we have chosen the SVD method.
        if self.solver == "svd":
            samples = np.array(samples)
            labels = np.array(labels)
            # Sort the data
            samples, labels = self.sortTheData(samples, labels)
            # Calculate the mean of each class
            self.computeClassMeans(samples, labels)
            j, k = np.unique(labels, return_counts=True)
            # Calculating the total mean of the samples
            self.computeMean(samples)
            # Calculates Hb
            Hb = self.computeHb(samples, labels)
            # Calculate Ht
            Ht = self.computeHt(samples).copy()
            Sb = self.computeSB(samples, k, len(k))
            # Calculate the reduced SVD of Ht
            U, Sigma, Vh, Inverse_Sigma = self.computeReducedSVD(Ht)
            # Calculate the matrix Y
            Y = self.computeY(Hb, Vh.T, Inverse_Sigma)
            # Computed the reduced SVD of Y
            _, _, Vh_tilde, _ = self.computeReducedSVD(Y)
            # Compute the A matrix using the SVD of Y and Ht.
            A = Vh.T @ Inverse_Sigma @ Vh_tilde.T
            # Saves the transformationMatrix in the class
            self.transformMatrix = A
            return A
        elif self.solver == 'eig':
            raise ValueError("Not implemented")
        else:
            raise ValueError("The solver chosen is unknown")

    def transform(self, samples):
        # subtract the mean from the samples and take that product with the transformation matrix
        transformedData = np.dot(samples - self.totalMean, self.transformMatrix)
        # Return the samples projected onto a lower dim
        return transformedData[:, : self.components]


class FaultDetector:

    def __init__(self, data=None, sizeOfTrainData=None, w=[], b=[]):
        self.reducer = None
        self.classifier = None
        self.sizeOfTrainData = sizeOfTrainData  # Only used to filename of predictedTable
        self.dataIsScalled = None
        self.data = data.copy()
        self.featureMean = None
        self._dataScalling()
        self.w = w
        self.b = b

    def save_model(self, modelName):
        if modelName == 'MLP':
            matdic = {"Weight": self.reducer.w, 'Bias': self.reducer.b}
            savemat(modelName + '.mat', matdic)

    def _dataScalling(self, valData=None):
        if self.data is None:
            raise ValueError('Insert data when initializing fault detector')
        if not isinstance(self.data, (list, np.ndarray)):
            raise ValueError("Wrong type of data")
        if self.dataIsScalled is None:
            self.data = np.array(self.data)
            self.featureMean = np.mean(self.data, axis=0)
            self.data = self.data - self.featureMean
            self.std = np.std(self.data, axis=0)
            self.std[self.std == 0] = 1.0
            self.data = self.data / self.std
            self.scalledTrainData = self.data
            self.dataIsScalled = 1
        else:
            valData = valData - self.featureMean
            valData = valData / self.std
            return valData

    def train(self, reductionMethod, components, trainLabels, gamma=0.1, C=1, verbose=False):
        """
        :param reductionMethod: String. Either PCA or LDA or None (for SVM only)
        :param components: Desired dimension of reduced data
        :param trainLabels: Only required for LDA. Labels of trainData
        :param eps: (optional) Tolerance for when we stop optimizing Lagrange multipliers
        :param tol: (optional) Tolerance for idk
        :param C: (optional) Regulation parameter
        :param gamma: (optional) invers of std for the normal distribution
        :param kernel: (optional) Which kernel to use
        :param verbose: Enable or disable logging
        :return: None
        """

        # --------------PCA------------------------
        if reductionMethod == 'PCA' or reductionMethod == 'pca':
            self.reducer = PCA(self.scalledTrainData)
            self.reducer.reduce(components)
            trainData = self.reducer.transform(self.scalledTrainData)

        # --------------LDA------------------------
        elif reductionMethod == 'LDA' or reductionMethod == 'lda':
            self.reducer = LDA(components)
            self.reducer.fit(self.scalledTrainData, trainLabels)
            trainData = self.reducer.transform(self.scalledTrainData)
        elif reductionMethod == 'none' or reductionMethod == 'None':
            print('Using SVM without reduction')
            trainData = self.scalledTrainData

        # --------------MLP------------------------
        elif reductionMethod == 'MLP' or reductionMethod == 'mlp':
            self.reducer = MLP(w=self.w, b=self.b, max_iter=5)
            self.reducer.fit(self.scalledTrainData, trainLabels, MLPDim=[25, 24, 22])
            print('Training MLP')
            return

        else:
            raise Exception("Invalid reduction method or missing training labels (LDA)")

        # --------------SVM------------------------

        # Creating the hyperplanes with SVM. Ideally replaced with our own code
        self.classifier = svm.SVC(gamma=gamma, C=C, decision_function_shape='ovo')
        self.classifier.fit(trainData, trainLabels)
        # For our implementation.  Have patience...
        # self.classifier = SVM_Multiclass(trainData, trainLabels, kernel=kernel, gamma=gamma, C=C, eps=eps, tol=tol, verbose=verbose)
        # self.classifier.fit()

    def test(self, testData, testLabels, info, predictionTable=False, testTime=False):
        """
        :param testData: Data to classify
        :param testLabels: Labels to calculate accuracy
        :return: None
        """

        testData = self._dataScalling(testData)
        if testTime != 1:
            if len(testData) != len(testLabels):
                raise Exception("Number of data points and labels should match")

            # Checking if the format is correct
            if type(testData) is not np.ndarray:
                testData = np.array(testData)
            if len(testData.shape) > 2:
                raise Exception("Test data may not be nested")
            if type(testLabels) is not np.ndarray:
                testLabels = np.array(testLabels)
            if len(testLabels.shape) > 2:
                raise Exception("Test labels may not be nested")

        if testTime:
            testData = np.reshape(testData, (1, -1))

        # Transform the test data passed into the function (either with PCA or LDA)
        if self.reducer != None and self.reducer.__class__.__name__!= "MLP":
            testData = self.reducer.transform(testData)

        # Guessing the labels of the test data with SVM
        if self.reducer.__class__.__name__ == "MLP":
            predictedLabels = self.reducer.predict(testData, testLabels)
        if self.reducer.__class__.__name__ != "MLP":
            predictedLabels = self.classifier.predict(testData)
        if testTime !=1:
            predictMatrix = np.zeros((21, 21))
            for i in range(len(predictedLabels)):
                predictMatrix[int(testLabels[i]), int(predictedLabels[i])] += 1
                predictionMatrixSum = predictMatrix.copy()
            if predictionTable:
                for i in range(21):
                    predictMatrix[i, :] = predictMatrix[i, :]/np.sum(predictMatrix[i, :])
                fig, ax = plt.subplots()
                idx = np.array(range(0, 21))
                im, cbar = Pot.heatmap(predictMatrix, idx, idx, ax=ax,
                                       cmap="Blues", cbarlabel="Percentage")
                ax.set_xlabel('Predicted label', fontsize=15)
                ax.set_ylabel('True label', fontsize=15)
                ax.patch.set_edgecolor('black')
                ax.patch.set_linewidth(2.50)
                solverID = self.reducer.__class__.__name__
                if self.reducer.__class__.__name__ == 'NoneType' or self.reducer.__class__.__name__ == 'none':
                    solverID = "SVM"
                computer = getpass.getuser()
                print(computer)
                if computer == 'User':
                    basepath = 'Define your base path for saving the confusion matrix'

                if self.reducer != None and self.reducer.__class__.__name__ != "MLP":
                    dimensionID = self.reducer.components
                else:
                    dimensionID = len(self.data[0])
                if solverID == "LDA":
                    plt.title("LDA-SVM", fontsize=25)
                elif solverID == "PCA":
                    plt.title("PCA-SVM", fontsize=25)
                elif solverID == "MLP":
                    plt.title("MLP", fontsize=25)
                else:
                    plt.title("SVM", fontsize=25)

            # Sorting all the labels into classes
            sortingIdx = np.argsort(testLabels)
            predictedLabels = np.array(predictedLabels)[sortingIdx]
            testLabels = testLabels[sortingIdx]

            # Making sure the class names are mapped onto 0 - 1 - 2 - etc.
            mapping = []
            mapping.append(np.unique(testLabels))

            # Calculating the accuracies of each class
            accuracies = []
            totalCorrectGuesses = 0
            for index in mapping[0]:
                # Find the indices of the predicted labels that should all correspond to one class
                classPrediction = predictedLabels[np.argwhere(testLabels == index)]

                # Count how many match with the corresponding label in testLabels
                correct = np.count_nonzero(classPrediction == index)
                totalCorrectGuesses += correct

                # Calculate the number of total possible correct guesses in that class
                numInClass = np.count_nonzero(testLabels == index)

                # Calculate the accuracy
                accuracies.append(correct / numInClass)

            # Printing the accuracies in the console
            printClasses = ''
            for i in range(len(mapping[0])):
                if mapping[0][i] == 0:
                    printClasses += '| Non-faulty \t\t --- ' + str(accuracies[i])
                else:
                    printClasses += '\n| Fault ' + str(mapping[0][i]) + ' \t\t\t --- ' + str(accuracies[i])
            print(printClasses)
            print('\u2500' * 25)
            avgAccuracy = totalCorrectGuesses/len(testLabels)
            print('| Average accuracy \t --- ', avgAccuracy)
            if predictionTable:
                plt.savefig(basepath + solverID + '_Dimensions_' + str(dimensionID) + '_RM3_' + str(info[5]) + '_BINARY_' + str(
                info[2]) + '_SWAPDATA_' + str(info[0]) + '_USETEST_' + str(info[1]) + '_TWOSTAGE_' + str(
                info[3]) + '_FRACTION_' + str(info[4]) + '_ACCURACY_' + str(avgAccuracy) + '.pdf', format="pdf", dpi=1000)
                plt.show()
                predictMatrix = np.nan_to_num(predictMatrix, copy=False)
                TrueNegative = predictMatrix[0][0] / np.sum(predictMatrix[0][:])
                FalsePositive = np.sum(predictMatrix[0][1:]) / np.sum(predictMatrix[0][:])
                FalseNegative = np.sum(predictMatrix[1:, 0]) / np.sum(predictMatrix[1:, :])
                TruePositive = np.sum(predictMatrix[1:, 1:]) / np.sum(predictMatrix[1:, :])
                print('TrueNegative' ,TrueNegative)
                print('FalsePositive' , FalsePositive)
                print('FalseNegative' , FalseNegative)
                print('TruePositive' , TruePositive)

            return predictedLabels, sortingIdx, predictionMatrixSum
        else:
            return predictedLabels



class SVM_Multiclass:

    def __init__(self, data, labels, kernel='linear', gamma=0.1, C=0.1, tol=1e-3, eps=1e-3, verbose=False):
        self.data = []
        self.labels = []
        self.mapping = []
        self.svms = []
        self.gamma = gamma
        self.C = C
        self.tol = tol
        self.eps = eps
        self.verbose = verbose
        self.kernel = kernel.lower()
        self.__sorting(np.array(data), np.array(labels))
        self.__initializeSVMbetw2Classes()

    def __sorting(self, x, y):
        # Making sure the class names are mapped onto 0 - 1 - 2 - etc.
        self.mapping.append(np.unique(y))
        self.mapping.append(list(range(len(self.mapping[0]))))

        for label in self.mapping[0]:
            self.data.append(x[np.where(np.array(y) == label)])
            self.labels.append(y[np.where(np.array(y) == label)])

    def __initializeSVMbetw2Classes(self):
        # Create all the necessary binary svm's such that every class meets every class
        for i in range(len(self.mapping[0])):
            for j in range(i + 1, len(self.mapping[0])):
                self.svms.append(SVM_dataBetweenClasses(i, j))

        if self.verbose:
            print('Number of SVM\'s:', len(self.svms))

    def fit(self):
        # Loop over all svm's
        for i in range(len(self.svms)):
            if self.verbose:
                print('Training SVM #', i + 1, 'of', len(self.svms))
            # Collect and combine the relevant training data and labels
            x1 = np.vstack((self.data[self.svms[i].classes[0]], self.data[self.svms[i].classes[1]]))
            y1 = np.hstack((self.labels[self.svms[i].classes[0]], self.labels[self.svms[i].classes[1]]))
            # Initialize an instance of the svm
            q = SVM_2D_fit(x1, y1, gamma=self.gamma, C=self.C, eps=self.eps, tol=self.tol, kernel=self.kernel)
            # train the svm and save all alphas, b and w
            [self.svms[i].alphas, self.svms[i].b, self.svms[i].w] = q()

        if self.verbose:
            print('Done training SVM')

    def __kernel(self, trainingData, sample):
        if self.kernel == 'linear':

            return trainingData @ np.array(sample)[:, None]

        elif self.kernel == 'gaussian':
            # Calculate the sample norm
            sampleNorm = np.sum(np.array(sample) ** 2, -1)
            # Calculate the outer sum of the norms
            M = np.sum(np.array(trainingData) ** 2, axis=-1) + sampleNorm
            C = np.array(sample)[:, None]
            B = 2 * trainingData @ C
            K = np.exp(-self.gamma * (M[:, None] - B))

            return K

    def predict_2D(self, sample, svm):
        # Collect and combine the relevant training data and labels
        trainingData = np.vstack((self.data[self.svms[svm].classes[0]], self.data[self.svms[svm].classes[1]]))
        targets = np.hstack((self.labels[self.svms[svm].classes[0]], self.labels[self.svms[svm].classes[1]]))
        # Map the labels to -1 and 1
        targets = np.where(targets == targets.min(), -1, 1)
        # Calculate the kernel
        a = self.__kernel(trainingData, sample)
        # Return the 2D prediction
        return np.sum(self.svms[svm].alphas[:, None] * targets[:, None] * a) - self.svms[svm].b

    def predict(self, samples):

        if self.verbose:
            print('Starting prediction')

        # An output corresponding to each given sample
        output = []

        # Loop over all samples and get each svm's prediction
        for idx, sample in enumerate(samples):
            if self.verbose and Decimal(str(idx / len(samples))) % Decimal('0.1') == 0:
                print(idx / len(samples) * 100, '%')
            # Array that contains all the outputs of the svm's
            interOutput = [[], []]

            # Loop over all svm's
            for j in range(len(self.svms)):
                # Get the prediction from the j'th svm
                pred = self.predict_2D(sample, j)
                # Map the svm output to match the actual classes
                predClass = self.svms[j].classes[0] if pred < 0 else self.svms[j].classes[1]
                # Append the predicted class
                interOutput[0].append(predClass)
                # Append the numerical output value e.g. 2.7
                interOutput[1].append(pred)

            # Gather the unique classes from the svm's predictions and the number of time they occur
            predictions, counts = np.unique(interOutput[0], return_counts=True)
            # Save the maximum number of occurrence
            maxOccurrence = max(counts)
            # If there is no single class that occur more than any other
            if np.count_nonzero(counts == maxOccurrence) > 1:
                # Find the svm's numerical output and sum the absolute values
                idx_where_max = np.argwhere(counts == maxOccurrence)
                max_preds = predictions[idx_where_max]
                certainties = []
                for v in range(len(max_preds)):
                    idx = np.argwhere(np.array(interOutput)[0] == max_preds[v])
                    class_certainty = np.sum(np.abs(np.array(interOutput)[1][idx]))
                    certainties.append(class_certainty)
                # Append the class with the highest certainty to the output
                output.append(max_preds[np.argmax(np.array(certainties))][0])

            else:
                # Append the class that occurs the most times to the output
                output.append(predictions[np.argmax(counts)])

        # Return the output vector
        return output


class SVM_dataBetweenClasses:
    def __init__(self, a, b):
        self.classes = [a, b]
        self.alphas = []
        self.b = 0
        self.w = None


class SVM_2D_fit:
    def __init__(self, data, labels, C=0.01, tol=1e-3, eps=1e-3, kernel='linear', gamma=0.1):
        self.data = data
        self.labels = labels
        self.C = C
        self.tol = tol
        self.eps = eps
        self.alphas = np.zeros(len(data))
        self.b = 0.0
        self.E = np.zeros((len(data)))
        self.examineAll = 1
        self.numChanged = 0
        self.gamma = gamma
        self.kernel = kernel.lower()
        self.K = None
        self.w = None
        self.__formatData(self.labels)
        self.fit()

    def __call__(self):
        return self.alphas, self.b, self.w

    def __kernel(self):
        if self.kernel == 'linear':
            self.K = self.data @ self.data.T
        elif self.kernel == 'gaussian':
            # Calculate the norm squared of all samples
            normSquaredOfsamples = np.sum(self.data ** 2, axis=-1)
            # Calculate the outer sum
            M = np.array([normSquaredOfsamples] * len(normSquaredOfsamples))
            M = M + M.T
            B = 2 * self.data @ self.data.T
            self.K = np.exp(-self.gamma * (M - B))

    def fit(self):
        self.__kernel()
        self.__updateErrorCache()
        C = self.C
        eps = self.eps
        while self.numChanged > 0 or self.examineAll:
            self.numChanged = 0

            # For the first pass and when no Lagrange multipliers are changed
            if self.examineAll:
                # Examine all samples
                for i in range(len(self.data)):
                    self.numChanged += self.__examineExample(i, C, eps)
            else:
                # Only examine the samples where alpha is not on the bounds
                for i in range(len(self.alphas)):
                    if 0 < self.alphas[i] < self.C:
                        self.numChanged += self.__examineExample(i, C, eps)
            if self.examineAll == 1:
                self.examineAll = 0
            elif self.numChanged == 0:
                self.examineAll = 1

        scalars = self.alphas * self.labels
        self.w = np.empty(np.array(self.data).shape)
        for i in range(len(self.alphas)):
            self.w[i] = (scalars[i] * self.data[i, :])

        self.w = np.sum(self.w, axis=0)
        return self.alphas, self.b

    def __examineExample(self, i2, C, eps):

        # Finds the target value for this sample
        y2 = self.labels[i2]
        alpha2 = self.alphas[i2]
        E2 = self.E[i2] = np.sum(self.alphas * self.labels * self.K[:, i2]) - self.b - y2
        r2 = E2 * y2
        if (r2 < -self.tol and alpha2 < C) or (r2 > self.tol and alpha2 > 0):
            i1 = 0
            # check if there are at least 2 un-bounded alphas
            if ((np.array(self.alphas) > 0) & (np.array(self.alphas) < C)).sum() > 1:
                # Choose 2nd alpha based on largest error |E1 - E2|
                if E2 >= 0:
                    i1 = np.argmin(np.array(self.E))
                if E2 < 0:
                    i1 = np.argmax(np.array(self.E))
                if self.__takeStep(i1, i2, E2, y2, alpha2, C, eps):
                    return 1

            startingPoint = random.randint(0, len(self.alphas))

            # check if there are at least 2 un-bounded alphas
            if ((np.array(self.alphas) > 0) & (np.array(self.alphas) < C)).sum() > 1:
                # Chose a 2nd alpha which is not bounded
                for j in range(len(self.alphas)):
                    i1 = (j + startingPoint) % len(self.alphas)

                    if i1 == i2:
                        continue

                    if 0 < self.alphas[i1] < C:
                        if self.__takeStep(i1, i2, E2, y2, alpha2, C, eps):
                            return 1

            # Choose a 2nd alpha at random and ensure that it is not the same as i2
            for k in range(len(self.alphas)):
                i1 = (k + startingPoint) % len(self.alphas)

                if i1 == i2:
                    continue

                if self.__takeStep(i1, i2, E2, y2, alpha2, C, eps):
                    return 1
        return 0

    def __takeStep(self, i1, i2, E2, y2, alpha2, C, eps):
        if i1 == i2:
            return 0
        alpha1 = self.alphas[i1]
        y1 = self.labels[i1]
        b = self.b
        E1 = np.sum(self.alphas * self.labels * self.K[:, i1]) - b - y1
        s = y1 * y2
        if y1 == y2:
            L = max(0, alpha2 + alpha1 - C)
            H = min(C, alpha2 + alpha1)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(C, C + alpha2 - alpha1)
        if L == H:
            return 0
        # Compute eta
        k11 = self.K[i1, i1]
        k12 = self.K[i1, i2]
        k22 = self.K[i2, i2]
        eta = k11 + k22 - 2 * k12
        # Eta is the second derivative of the objective function along the diagonal line
        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            # The objective function is not positive definite
            f1 = y1 * (E1 + b) - alpha1 * k11 - s * alpha2 * k12
            f2 = y2 * (E2 + b) - s * alpha1 * k12 - alpha2 * k22
            L1 = alpha1 + s * (alpha2 - L)
            H1 = alpha1 + s * (alpha2 - H)
            psi_L = L1 * f1 + L * f2 + 1 / 2 * L1 * L1 * k11 + 1 / 2 * L * L * k22 + s * L * L1 * k12
            psi_H = H1 * f1 + H * f2 + 1 / 2 * H1 * H1 * k11 + 1 / 2 * H * H * k22 + s * H * H1 * k12

            if psi_L < psi_H - eps:
                a2 = L
            elif psi_L > psi_H + eps:
                a2 = H
            else:
                a2 = alpha2

        if abs(a2 - alpha2) < eps * (a2 + alpha2 + eps):
            return 0

        a1 = alpha1 + s * (alpha2 - a2)

        bounded_1 = (a1 == 0) | (a1 == C)
        bounded_2 = (a2 == 0) | (a2 == C)
        b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + b
        b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + b

        if (not bounded_1) and (not bounded_2):
            self.b = b1
        elif not bounded_1:
            self.b = b1
        elif not bounded_2:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        self.alphas[i1] = a1
        self.alphas[i2] = a2

        return 1

    def __formatData(self, labels):
        self.labels = np.where(labels == labels.min(), -1, 1)

    def __updateErrorCache(self):
        for q in range(len(self.alphas)):
            self.E[q] = np.sum(self.alphas * self.labels * self.K[:, q]) - self.b - self.labels[q]


class MLP:
    def __init__(self, w=[], b=[], max_iter=50):

        self.d = None
        self.n = None
        self.delta = []
        self.w = w
        self.b = b
        self.b_grad = []
        self.a = []
        self.z = []
        self.y_trn_lbl = None
        self.y_tst_lbl = None
        self.w_grad = []
        self.x_trn = None
        self.x_tst = None
        self.iter = max_iter

    def sigmoid(self, a):
        z = 1 / (1 + np.exp(-a))
        return z

    def sigmoid_Derivative(self, a):
        z_der = self.sigmoid(a).copy() * (1 - self.sigmoid(a).copy())
        return z_der

    def softmax(self, a):
        y_pred = np.exp(a) / sum(np.exp(a))
        return y_pred

    def fit(self, X, y, MLPDim=[], lr=0.01):
        X, y = sklearn.utils.shuffle(np.array(X), np.array(y))

        self.n, self.d = np.shape(X)
        self.x_trn = np.array(X).T

        mapping = []
        mapping.append(np.unique(y))
        mapping.append(list(range(len(mapping[0]))))
        mapping = np.array(mapping)

        y_mapped = np.empty(y.shape)
        for idx, lbl in enumerate(y):
            loc = np.argwhere(mapping[0] == lbl)
            y_mapped[idx] = mapping[1][loc]

        self.y_trn_lbl = np.array(to_categorical(y_mapped))


        self.y_trn_lbl = (self.y_trn_lbl.copy()).T
        unq_y = len(np.unique(y))

        # First list
        self.w.append(np.random.uniform(-1, 1, size=(MLPDim[0], self.d)))
        self.w_grad.append(np.random.uniform(-1, 1, size=(MLPDim[0], self.d)))
        self.b.append(np.zeros((MLPDim[0], 1)))
        self.b_grad.append(np.zeros((MLPDim[0], 1)))
        self.a.append(np.random.rand(MLPDim[0], 1))
        self.z.append(np.random.rand(MLPDim[0], 1))
        self.delta.append(np.random.rand(MLPDim[0], 1))
        for b in range(len(MLPDim) - 1):
            self.w.append(np.random.uniform(-1, 1, size=(MLPDim[b + 1], MLPDim[b])))
            self.w_grad.append(np.random.uniform(-1, 1, size=(MLPDim[b + 1], MLPDim[b])))
            self.b.append(np.zeros((MLPDim[b + 1], 1)))
            self.b_grad.append(np.zeros((MLPDim[b + 1], 1)))
            self.a.append(np.random.rand(MLPDim[b + 1], 1))
            self.z.append(np.random.rand(MLPDim[b + 1], 1))
            self.delta.append(np.random.rand(MLPDim[b + 1], 1))
        # Last List
        self.w.append(np.random.uniform(-1, 1, size=(unq_y, MLPDim[-1])))
        self.w_grad.append(np.random.uniform(-1, 1, size=(unq_y, MLPDim[-1])))
        self.b.append(np.zeros((unq_y, 1)))
        self.b_grad.append(np.zeros((unq_y, 1)))
        self.a.append(np.random.rand(unq_y, 1))
        self.z.append(np.random.rand(unq_y, 1))
        self.delta.append(np.random.rand(unq_y, 1))
        # Convert each list to array
        self.w = np.array([np.array(wi) for wi in self.w])
        self.b = np.array([np.array(bi) for bi in self.b])
        self.a = np.array([np.array(ai) for ai in self.a])
        self.z = np.array([np.array(zi) for zi in self.z])
        self.delta = np.array([np.array(delta_i) for delta_i in self.delta])

        for loop in range(self.iter):
            print(loop)
            for i in range(self.n):


                # FeedForward
                input = self.x_trn[:, i].reshape(self.d, -1)

                self.a[0] = np.dot(self.w[0].copy(), input) + self.b[0].copy()  # For 1 sample
                self.z[0] = self.sigmoid(self.a[0])

                for j in range(len(self.w) - 2):
                    self.a[j + 1] = self.w[j + 1].copy().dot(self.z[j].copy()) + self.b[j + 1].copy()
                    self.z[j + 1] = self.sigmoid(self.a[j + 1].copy())

                self.a[-1] = self.w[-1].dot(self.z[-2].copy()) + self.b[-1].copy()
                self.z[-1] = self.softmax(self.a[-1].copy())

                # Error Back-Propagation
                self.delta[-1] = (self.z[-1].copy() - np.array(self.y_trn_lbl[:, i].copy()).reshape(unq_y, -1))

                for l in range(len(self.delta) - 1):
                    self.delta[-2 - l] = self.sigmoid_Derivative(self.a[-2 - l].copy()) * np.matmul(self.w[-1 - l].copy().T, self.delta[-1 - l].copy())

                for q in range(len(self.delta) - 1):
                    self.w_grad[-1 - q] = np.outer(self.delta[-1 - q].copy().flatten(), self.z[-2 - q].copy().flatten())
                    # self.b_grad[-1 - q] = np.sum(self.w_grad[-1 - q].copy(), axis=1).reshape(self.w_grad[-1 - q].copy().shape[0], -1)
                    self.b_grad[-1 - q] = self.delta[-1 - q]

                self.w_grad[0] = np.outer(self.delta[0].copy().flatten(), input.flatten())
                # self.b_grad[0] = np.sum(self.w_grad[0].copy(), axis=1).reshape(self.w_grad[0].copy().shape[0], -1)
                self.b_grad[0] = self.delta[0]

                for o in range(len(self.w)):
                    self.w[o] = self.w[o].copy() - lr * self.w_grad[o].copy()
                    self.b[o] = self.b[o].copy() - lr * self.b_grad[o].copy()

    def predict(self, tst_data, tst_label):

        self.n, self.d = np.shape(tst_data)
        # FeedForward
        self.x_tst = np.array(tst_data).T
        self.y_tst_lbl = np.array(to_categorical(tst_label))
        self.y_tst_lbl = (self.y_tst_lbl).T
        unq_y = len(np.unique(tst_label))
        self.y_pred = []

        mapping = []
        mapping.append(np.unique(tst_label))
        mapping.append(list(range(len(mapping[0]))))
        mapping = np.array(mapping)

        y_mapped = np.empty(tst_label.shape)
        for idx, lbl in enumerate(tst_label):
            loc = np.argwhere(mapping[0] == lbl)
            y_mapped[idx] = mapping[1][loc]

        for i in range(self.n):
            input = self.x_tst[:, i].reshape(self.d, -1)

            self.a[0] = np.dot(self.w[0].copy(), input) + self.b[0].copy()  # For 1 sample
            self.z[0] = self.sigmoid(self.a[0].copy())

            for j in range(len(self.w) - 2):
                self.a[j + 1] = self.w[j + 1].copy().dot(self.z[j].copy()) + self.b[j + 1].copy()
                self.z[j + 1] = self.sigmoid(self.a[j + 1].copy())

            self.a[-1] = self.w[-1].copy().dot(self.z[-2]) + self.b[-1].copy()
            self.z[-1] = self.softmax(self.a[-1].copy())

            for cls in range(unq_y):
                if np.argmax(self.z[-1].copy()) == cls:
                    clc_mapped = mapping[0][np.argwhere(mapping[1] == cls)]
                    self.y_pred.append(clc_mapped)

        return self.y_pred


def mlp_load_file(MatFile):
    data = loadmat(str(MatFile))
    w = np.array(data['Weight'])
    b = np.array(data['Bias'])
    return w, b
