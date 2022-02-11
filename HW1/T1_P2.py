#####################
# CS 181, Spring 2022
# Homework 1, Problem 2
# Start Code
##################

from dis import dis
import math
from this import d
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# set up data
data = [(0., 0.),
        (1., 0.5),
        (2., 1),
        (3., 2),
        (4., 1),
        (6., 1.5),
        (8., 0.5)]

x_train = np.array([d[0] for d in data])
y_train = np.array([d[1] for d in data])

x_test = np.arange(0, 12, .1)

print("y is:")
print(y_train)

def predict_knn(k=1, tau=1):
    """Returns predictions for the values in x_test, using KNN predictor with the specified k."""
    y_test = []

    for x in x_test:
        pred = []
        for pt in data:
            pred.append([np.exp(-(x - pt[0]) * (x - pt[0]) / tau), pt[0], pt[1]])

        sorted_pred = sorted(pred, key=lambda i: i[0], reverse=True)

        prediction = 0
        for i in range(k):
            prediction += sorted_pred[i][2]

        y_test.append(prediction / k)

    return y_test


def plot_knn_preds(k):
    plt.xlim([0, 12])
    plt.ylim([0,3])
    
    y_test = predict_knn(k=k)
    
    plt.scatter(x_train, y_train, label = "training data", color = 'black')
    plt.xlabel(r'$x^*$')
    plt.ylabel(r'$f(x^*)$')
    plt.plot(x_test, y_test, label = "predictions using k = " + str(k))

    plt.legend()
    plt.title("KNN Predictions with k = " + str(k))
    plt.savefig('k' + str(k) + '.png')
    plt.show()

for k in (1, 3, len(x_train)-1):
    plot_knn_preds(k)