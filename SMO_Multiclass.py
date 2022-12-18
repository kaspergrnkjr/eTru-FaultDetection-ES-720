import math

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import time
import random
import matplotlib.pyplot as plt


class SVM_Multiclass:

    def __init__(self, data, labels, kernel='linear', gamma=0.1, C=0.1, tol=1e-3, eps=1e-3):
        self.data = []
        self.labels = []
        self.mapping = []
        self.svms = []
        self.gamma = gamma
        self.C = C
        self.tol = tol
        self.eps = eps
        self.kernel = kernel.lower()
        self.__sorting(data, labels)
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
            for j in range(i+1, len(self.mapping[0])):
                self.svms.append(SVM_dataBetweenClasses(i, j))

    def fit(self):
        # Loop over all svm's
        for i in range(len(self.svms)):
            # Collect and combine the relevant training data and labels
            x1 = np.vstack((self.data[self.svms[i].classes[0]], self.data[self.svms[i].classes[1]]))
            y1 = np.hstack((self.labels[self.svms[i].classes[0]], self.labels[self.svms[i].classes[1]]))
            # Initialize an instance of the svm
            q = SVM_2D_fit(x1, y1, gamma=self.gamma, C=self.C, eps=self.eps, tol=self.tol, kernel=self.kernel)
            # train the svm and save all alphas, b and w
            [self.svms[i].alphas, self.svms[i].b, self.svms[i].w] = q()

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
        # An output corresponding to each given sample
        output = []

        # Loop over all samples and get each svm's prediction
        for sample in samples:
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
            B = 2*self.data @ self.data.T
            self.K = np.exp(-self.gamma*(M - B))

    def fit(self):
        self.__kernel()
        #self.__formatData(self.labels)
        self.__updateErrorCache()
        while self.numChanged > 0 or self.examineAll:
            self.numChanged = 0

            # For the first pass and when no Lagrange multipliers are changed
            if self.examineAll:
                # Examine all samples
                for i in range(len(self.data)):
                    self.numChanged += self.__examineExample(i)
            else:
                # Only examine the samples where alpha is not on the bounds
                for i in range(len(self.alphas)):
                    if 0 < self.alphas[i] < self.C:
                        self.numChanged += self.__examineExample(i)
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

    def __examineExample(self, i2):
        # Finds the target value for this sample
        y2 = self.labels[i2]
        alpha2 = self.alphas[i2]
        E2 = self.E[i2] = np.sum(self.alphas * self.labels * self.K[:, i2]) - self.b - self.labels[i2]
        r2 = E2 * y2
        if (r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):
            i1 = 0
            # check if there are at least 2 un-bounded alphas
            if ((np.array(self.alphas) > 0) & (np.array(self.alphas) < self.C)).sum() > 1:
                # Choose 2nd alpha based on largest error |E1 - E2|
                if E2 >= 0:
                    i1 = np.argmin(np.array(self.E))
                if E2 < 0:
                    i1 = np.argmax(np.array(self.E))
                if self.__takeStep(i1, i2):
                    return 1

            startingPoint = random.randint(0, len(self.alphas))

            # check if there are at least 2 un-bounded alphas
            if ((np.array(self.alphas) > 0) & (np.array(self.alphas) < self.C)).sum() > 1:
                # Chose a 2nd alpha which is not bounded
                for j in range(len(self.alphas)):
                    i1 = (j + startingPoint) % len(self.alphas)

                    if i1 == i2:
                        continue

                    if 0 < self.alphas[i1] < self.C:
                        if self.__takeStep(i1, i2):
                            return 1

            # Choose a 2nd alpha at random and ensure that it is not the same as i2
            for k in range(len(self.alphas)):
                i1 = (k + startingPoint) % len(self.alphas)

                if i1 == i2:
                    continue

                if self.__takeStep(i1, i2):
                    return 1
        return 0

    def __takeStep(self, i1, i2):
        if i1 == i2:
            return 0
        alpha1 = self.alphas[i1]
        alpha2 = self.alphas[i2]
        y1 = self.labels[i1]
        y2 = self.labels[i2]
        E1 = np.sum(self.alphas * self.labels * self.K[:, i1]) - self.b - self.labels[i1]
        E2 = np.sum(self.alphas * self.labels * self.K[:, i2]) - self.b - self.labels[i2]
        s = y1 * y2
        if y1 == y2:
            L = max(0, self.alphas[i2] + self.alphas[i1] - self.C)
            H = min(self.C, self.alphas[i2] + self.alphas[i1])
        else:
            L = max(0, self.alphas[i2] - self.alphas[i1])
            H = min(self.C, self.C + self.alphas[i2] - self.alphas[i1])
        if L == H:
            return 0
        # Compute eta
        k11 = self.K[i1, i1]
        k12 = self.K[i1, i2]
        k22 = self.K[i2, i2]
        eta = k11 + k22 - 2 * k12
        # Eta is the second derivative of the objective function along the diagonal line
        if eta > 0:
            a2 = self.alphas[i2] + self.labels[i2] * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            # The objective function is not positive definite
            f1 = y1 * (E1 + self.b) - alpha1 * k11 - s * alpha2 * k12
            f2 = y2 * (E2 + self.b) - s * alpha1 * k12 - alpha2 * k22
            L1 = alpha1 + s * (alpha2 - L)
            H1 = alpha1 + s * (alpha2 - H)
            psi_L = L1 * f1 + L * f2 + 1 / 2 * L1 * L1 * k11 + 1 / 2 * L * L * k22 + s * L * L1 * k12
            psi_H = H1 * f1 + H * f2 + 1 / 2 * H1 * H1 * k11 + 1 / 2 * H * H * k22 + s * H * H1 * k12

            if psi_L < psi_H - self.eps:
                a2 = L
            elif psi_L > psi_H + self.eps:
                a2 = H
            else:
                a2 = alpha2

        if abs(a2 - alpha2) < self.eps * (a2 + alpha2 + self.eps):
            return 0

        a1 = alpha1 + s * (alpha2 - a2)

        bounded_1 = (a1 == 0) | (a1 == self.C)
        bounded_2 = (a2 == 0) | (a2 == self.C)
        b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + self.b
        b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + self.b

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


dimensions = 2
classes = 2
samples1 = 100*classes
plot = 1
centers = []
for q in range(classes):
    centers.append((random.uniform(-5, 5), random.uniform(-5, 5)))

X, y = make_blobs(n_samples=samples1, centers=centers, n_features=dimensions, cluster_std=5)


X_test, y_test = make_blobs(n_samples=samples1, centers=centers, n_features=dimensions, cluster_std=5)

start_time = time.time()

svmMultiModel = SVM_Multiclass(X, y, kernel='linear', C=0.00001)
svmMultiModel.fit()
Y = svmMultiModel.predict(X_test)
comparison = Y == y_test
correctGuesses = sum(comparison)
print('Our: ', correctGuesses / len(Y))
print('Our', "--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()

Q = SVC()
Q.fit(X, y)
Y_1 = Q.predict(X_test)
comparison_1 = Y_1 == y_test

correctGuesses_1 = sum(comparison_1)

print('library: ', correctGuesses_1 / len(Y_1))

print('Lib', "--- %s seconds ---" % (time.time() - start_time))


if plot == 1:
    cmap = ['blue', 'red', 'green', 'pink', 'purple', 'yellow']

    for j in range(classes):
        plt.scatter(svmMultiModel.data[j][:, 0], svmMultiModel.data[j][:, 1], s=5, c=cmap[j])

    for i in range(len(svmMultiModel.svms)):
        _classes = svmMultiModel.svms[i].classes
        endpoints_max = max(max(svmMultiModel.data[_classes[0]][:, 0]), max(svmMultiModel.data[_classes[1]][:, 0]))
        endpoints_min = min(min(svmMultiModel.data[_classes[0]][:, 0]), min(svmMultiModel.data[_classes[1]][:, 0]))
        x_points = np.linspace(endpoints_min, endpoints_max)  # generating x-points from -1 to 1
        w = svmMultiModel.svms[i].w
        b = -svmMultiModel.svms[i].b
        y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points

        plt.plot(x_points, y_points, c=cmap[i])

    plt.grid()
    plt.show()
