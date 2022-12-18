#
# SMO_SVM.py
# eTRU_faultDetection
#
# Created by Kasper Grønkjær on 01/11/2022 at 09.02
# Copyright © 2022 Kasper Grønkjær. All rights reserved.
from typing import Any

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import time
import random
import matplotlib.pyplot as plt

start_time = time.time()


samples = 100
dimensions = 2
classes = 2

X, y = make_blobs(n_samples=samples, centers=((0, 2), (0, -2)), n_features=dimensions, cluster_std=0.1)

X_test, y_test = make_blobs(n_samples=samples, centers=classes, n_features=dimensions, random_state=5)
"""
ldamodel = LDA(n_components=3)
Z = ldamodel.fit_transform(X,y)
Z1 = ldamodel.transform(X_test)
"""
Q = SVC()
Q.fit(X, y)

print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
numChanged = 0
examineAll = 1

# Map 0 to -1
y = np.where(y == 0, -1, 1)


K = X @ X.T
# Regularization parameter
C = 1000
tol = 1e-3
eps = 1e-3
alphas = np.zeros(len(X))
b = 0.0
E = np.zeros((len(X)))
bs =[]


def takeStep(i1, i2):
    global b
    if i1 == i2:
        return 0
    alpha1 = alphas[i1]
    alpha2 = alphas[i2]
    y1 = y[i1]
    y2 = y[i2]
    E1 = np.sum(alphas*y*K[:, i1]) - b - y[i1]
    E2 = np.sum(alphas*y*K[:, i2]) - b - y[i2]
    s = y1*y2
    if y1 == y2:
        L = max(0, alphas[i2] + alphas[i1] - C)
        H = min(C, alphas[i2] + alphas[i1])
    else:
        L = max(0, alphas[i2] - alphas[i1])
        H = min(C, C + alphas[i2] - alphas[i1])
    if L == H:
        return 0
    # Compute eta
    k11 = K[i1, i1]
    k12 = K[i1, i2]
    k22 = K[i2, i2]
    eta = k11 + k22 - 2*k12
    # Eta is the second derivative of the objective function along the diagonal line
    if eta > 0:
        a2 = alphas[i2] + y2*(E1 - E2)/eta
        if a2 < L:
            a2 = L
        elif a2 > H:
            a2 = H
    else:
        # The objective function is not positive definite
        f1 = y1*(E1 + b) - alpha1*k11 - s*alpha2*k12
        f2 = y2*(E2 + b) - s*alpha1*k12 - alpha2*k22
        L1 = alpha1 + s*(alpha2 - L)
        H1 = alpha1 + s*(alpha2 - H)
        psi_L = L1*f1 + L*f2 + 1/2*L1*L1*k11 + 1/2*L*L*k22 + s*L*L1*k12
        psi_H = H1*f1 + H*f2 + 1/2*H1*H1*k11 + 1/2*H*H*k22 + s*H*H1*k12

        if psi_L < psi_H - eps:
            a2 = L
        elif psi_L > psi_H + eps:
            a2 = H
        else:
            a2 = alpha2

    if abs(a2-alpha2) < eps*(a2 + alpha2 + eps):
        return 0

    a1 = alpha1 + s*(alpha2 - a2)

    bounded_1 = (a1 == 0) | (a1 == C)
    bounded_2 = (a2 == 0) | (a2 == C)
    b1 = E1 + y1*(a1 - alpha1)*k11 + y2*(a2 - alpha2)*k12 + b
    b2 = E2 + y1*(a1 - alpha1)*k12 + y2*(a2 - alpha2)*k22 + b

    if (not bounded_1) and (not bounded_2):
        b = b1
    elif not bounded_1:
        b = b1
    elif not bounded_2:
        b = b2
    else:
        b = (b1+b2)/2

    bs.append(b)

    alphas[i1] = a1
    alphas[i2] = a2

    return 1


def examineExample(i2):
    # Finds the target value for this sample
    y2 = y[i2]
    alpha2 = alphas[i2]
    E2 = np.sum(alphas*y*K[:, i2]) - b - y[i2]
    r2 = E2*y2
    if (r2 < -tol and alpha2 < C) or (r2 > tol and alpha2 > 0):
        i1 = 0
        # check if there are at least 2 un-bounded alphas
        if ((np.array(alphas) > 0) & (np.array(alphas) < C)).sum() > 1:
            # Choose 2nd alpha based on largest error |E1 - E2|
            if E2 >= 0:
                i1 = np.argmin(np.array(E))
            if E2 < 0:
                i1 = np.argmax(np.array(E))
            if takeStep(i1, i2):
                return 1

        startingPoint = random.randint(0, len(alphas))

        # check if there are at least 2 un-bounded alphas
        if ((np.array(alphas) > 0) & (np.array(alphas) < C)).sum() > 1:
            # Chose a 2nd alpha which is not bounded
            for j in range(len(alphas)):
                i1 = (j + startingPoint) % len(alphas)

                if i1 == i2:
                    continue

                if 0 < alphas[i1] < C:
                    if takeStep(i1, i2):
                        return 1

        # Choose a 2nd alpha at random and ensure that it is not the same as i2
        for k in range(len(alphas)):
            i1 = (k + startingPoint) % len(alphas)

            if i1 == i2:
                continue

            if takeStep(i1, i2):
                return 1
    return 0


def updateErrorCache():
    for q in range(len(alphas)):
        E[q] = np.sum(alphas*y*K[:, q]) - b - y[q]


updateErrorCache()

while numChanged > 0 or examineAll:
    numChanged = 0

    # For the first pass and when no Lagrange multipliers are changed
    if examineAll:
        # Examine all samples
        for i in range(len(X)):
            numChanged += examineExample(i)
    else:
        # Only examine the samples where alpha is not on the bounds
        for i in range(len(alphas)):
            if 0 < alphas[i] < C:
                numChanged += examineExample(i)


    if examineAll == 1:
        examineAll = 0
    elif numChanged == 0:
        examineAll = 1

scalars = alphas*y
w = np.empty(np.array(X).shape)
for i in range(len(alphas)):
    w[i] = (scalars[i]*X[i, :])

w = np.sum(w, axis=0)

print("--- %s seconds ---" % (time.time() - start_time))
print(w)
print(b)

red = []
blue = []

for i in range(len(X)):
    gtr = alphas*y*K[:, i]
    vt = np.sum(gtr)
    if (vt - b) >= 0:
        red += [i]
    else:
        blue += [i]


plt.scatter(X[red, 0], X[red, 1], s=5, c='red')
plt.scatter(X[blue, 0], X[blue, 1], s=5, c='blue')

b = -b

x_points = np.linspace(-2, 2)    # generating x-points from -1 to 1
y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
# Plotting a red hyperplane
plt.plot(x_points, y_points, c='purple')
plt.grid()
plt.show()



