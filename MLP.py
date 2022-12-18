import math
import math
import numpy as np
import sklearn.datasets as sk
import sklearn.metrics
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import random as random
import loadfile as lf
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import warnings



with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)


def _dataScalling(data, valData):

    data = np.array(data)
    valData = np.array(valData)
    featureMean = np.mean(data, axis=0)
    data = data - featureMean
    std = np.std(data, axis=0)
    std[std == 0] = 1.0
    data = data / std
    scalledTrainData = data
    valData = valData - featureMean
    valData = valData / std
    return scalledTrainData, valData


class MLP:
    def __init__(self, max_iter=20):
        self.d = None
        self.n = None
        self.delta = []
        self.w = []
        self.b = []
        self.b_grad = []
        self.a = []
        self.z = []
        self.y_trn_lbl = None
        self.y_tst_lbl = None
        self.w_grad = []
        self.x_trn = None
        self.x_tst = None
        self.iter = max_iter
        self.threshold = 0.0000000000001

    def sigmoid(self, a):
        z = 1.0 / (1.0 + np.exp(-a))
        return z

    def sigmoid_Derivative(self, a):
        sig = self.sigmoid(a)
        sig_der = sig * (1- sig)
        return sig_der

    def Relu(self, a):
        z = np.maximum(0, a)
        return z

    def Relu_Derivative(self, a):
        z = np.zeros((len(a), 1))
        for i in range(len(a)):
            if a[i] > 0:
                z[i] = 1
            else:
                z[i] = 0
        return z

    def softmax(self, a):
        y_pred = np.exp(a) / sum(np.exp(a))
        return y_pred

    def fit(self, X, y, dim=[], lr=0.01):

        self.n, self.d = np.shape(X)
        self.x_trn = np.array(X).T

        self.y_trn_lbl = np.array(to_categorical(y))
        self.y_trn_lbl = (self.y_trn_lbl).T
        unq_y = len(np.unique(y))

        # First list
        self.w.append(np.random.uniform(-1, 1, size=(dim[0], self.d)))
        self.w_grad.append(np.random.uniform(-1, 1, size=(dim[0], self.d)))
        self.b.append(np.zeros((dim[0], 1)))
        self.b_grad.append(np.zeros((dim[0], 1)))
        self.a.append(np.random.rand(dim[0], 1))
        self.z.append(np.random.rand(dim[0], 1))
        self.delta.append(np.random.rand(dim[0], 1))
        for b in range(len(dim) - 1):
            self.w.append(np.random.uniform(-1, 1, size=(dim[b + 1], dim[b])))
            self.w_grad.append(np.random.uniform(-1, 1, size=(dim[b + 1], dim[b])))
            self.b.append(np.zeros((dim[b + 1], 1)))
            self.b_grad.append(np.zeros((dim[b + 1], 1)))
            self.a.append(np.random.rand(dim[b + 1], 1))
            self.z.append(np.random.rand(dim[b + 1], 1))
            self.delta.append(np.random.rand(dim[b + 1], 1))
        # Last List
        self.w.append(np.random.uniform(-1, 1, size=(unq_y, dim[-1])))
        self.w_grad.append(np.random.uniform(-1, 1, size=(unq_y, dim[-1])))
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
            counter = 0
            for i in range(self.n):
                counter = counter + 1
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

                # Non-Stochastic version
                for q in range(len(self.delta) - 1):
                    self.w_grad[-1 - q] = (np.outer(self.delta[-1 - q].copy().flatten(), self.z[-2 - q].copy().flatten()))
                    self.b_grad[-1 - q] = self.delta[-1 - q]


                self.w_grad[0] = (np.outer(self.delta[0].copy().flatten(), input.flatten()))
                self.b_grad[0] = self.delta[0]
                for o in range(len(self.w)):
                    self.w[o] = self.w[o].copy() - lr * self.w_grad[o].copy()
                    self.b[o] = self.b[o].copy() - lr * self.b_grad[o].copy()

                #Stochastic version
                """
                for q in range(len(self.delta) - 1):
                   self.w_grad[-1 - q] = self.w_grad[-1 - q].copy() + (np.outer(self.delta[-1 - q].copy().flatten(), self.z[-2 - q].copy().flatten()))
                   self.b_grad[-1 - q] = self.b_grad[-1 - q].copy() + (np.sum(self.w_grad[-1 - q].copy(), axis=1).reshape(self.w_grad[-1 - q].copy().shape[0], -1))
                self.w_grad[0] = self.w_grad[0].copy() + (np.outer(self.delta[0].copy().flatten(), input.flatten()))
                self.b_grad[0] = self.b_grad[0].copy() + (np.sum(self.w_grad[0].copy(), axis=1).reshape(self.w_grad[0].copy().shape[0], -1))
                if counter == 50:
                   for o in range(len(self.w)):
                       self.w[o] = self.w[o].copy() - lr * self.w_grad[o].copy()/50
                       self.b[o] = self.b[o].copy() - lr * self.b_grad[o].copy()/50
                       self.w_grad[o].fill(0)
                       self.b_grad[o].fill(0)
                       counter = 0
                """




    def predict(self, tst_data, tst_label):

        self.n, self.d = np.shape(tst_data)
        # FeedForward
        self.x_tst = tst_data.T
        self.y_tst_lbl = np.array(to_categorical(tst_label))
        self.y_tst_lbl = (self.y_tst_lbl).T
        unq_y = len(np.unique(tst_label))
        y_pred = []
        y_true = []
        count = 0

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
                    y_pred.append(cls)

            for tcls in range(unq_y):
                if np.argmax(np.array(self.y_tst_lbl).T) == tcls:
                    y_true.append(tcls)

        for l in range(len(y_pred)):
            if y_true[l] == y_pred[l]:
                count = count + 1

        print(sklearn.metrics.confusion_matrix(y_pred,y_true))
        accuracy = count / (len(y_pred))
        return print('Accuracy:', accuracy,y_pred)



trainData, trainLabels, valData, valLabels, testData, testLabels = lf.load_PercentageData(80, trainData=True, validationData=True, testData=None)

scaler = MinMaxScaler()
trainData = scaler.fit_transform(trainData)
valData = scaler.fit_transform(valData)

trainData, trainLabels = shuffle(trainData, trainLabels, random_state=1)


X1, y1 = sk.make_blobs(n_samples=2500, centers=21, n_features=14)
X1, y1 = shuffle(X1, y1, random_state=0)

X2 = X1[0:round(len(X1)*0.9)-1]
y2 = y1[0:round(len(X1)*0.9)-1]

X3 = X1[round(len(X1)*0.9):-1]
y3 = y1[round(len(X1)*0.9):-1]

mlp = MLP()
mlp.fit(trainData, trainLabels, dim=[10, 10, 10])
mlp.predict(valData, valLabels)


