from os import pread
import numpy as np
from random import random
import matplotlib.pyplot as plt

# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = 0.01
        self.lam = 0.001
        self.W = np.ones(shape=(3, 3))
        self.loss_hist = []
        self.iters = []

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __gradient(self, X, y):
        y_pred = self.__softmax(X @ self.W.T)
        grad = (y_pred - y).T.dot(X)

        regularize = 2 * self.lam * self.W

        return grad + regularize

    def __softmax(self, z):
        new_z = []
        for i in z:
            new_z.append(np.exp(i) / np.sum(np.array(list(map(lambda x : np.exp(x), i)))))

        return np.array(new_z)

    # TODO: Implement this method!
    def fit(self, X, y):
        self.X = X

        self.W = np.ones(shape=(3, 3))
        X = np.append(np.ones(shape=(27, 1)), X, axis=1)

        one_hot = []
        for i in range(len(y)):
            if y[i] == 0:
                one_hot.append([1, 0, 0])
            elif y[i] == 1:
                one_hot.append([0, 1, 0])
            else:
                one_hot.append([0, 0, 1])

        one_hot = np.array(one_hot)

        epoch = 0
        while epoch < 200000:
            gradient = self.__gradient(X, one_hot)
            self.W -= self.eta * gradient

            self.loss_hist.append(np.sum(one_hot * -1 *np.log(self.__softmax(X @ self.W.T))))
            self.iters.append(epoch)

            epoch += 1
        return

    # TODO: Implement this method!
    def predict(self, X_pred):
        X_pred = np.append(np.ones(shape=(np.shape(X_pred)[0], 1)), X_pred, axis=1)

        return np.argmax(X_pred @ self.W.T, axis=1)

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        plt.plot(self.iters, self.loss_hist)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Negative Log-Likelihood Loss")
        plt.title(r'Negative Log-Likelihood Loss for $\lambda = $' +  str(self.lam) +  r'and $\eta = $' + str(self.eta))
        plt.legend()
        pass
