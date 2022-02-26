from errno import ENOSPC
from statistics import mean
import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance
        self.k = 3
        self.n = 0
        self.mu = []
        self.bias = 0
        self.pi = None
        self.cov = None
        self.shared_cov = None

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):
        n = y.shape[0]
        self.n = n
        mu = np.zeros(shape=(self.k, X.shape[1]))
        cov = []
        cov_shared = np.zeros(shape=(X.shape[1], X.shape[1]))

        self.pi = np.log(np.bincount(y) / np.sum(np.bincount(y)))

        for i in range(self.k):
            numerator = []
            for j in range(n):
                if y[j] == i:
                    numerator.append(X[j])

            mu[i] = np.mean(np.array(numerator), axis=0)
        
        self.mu = mu

        for i in range(self.k):
            x_k = X[y == i]
            if self.is_shared_covariance:
                cov_shared += np.cov(x_k.T) * x_k.shape[0]
            else:
                cov.append(np.cov(x_k.T))

        if self.is_shared_covariance:
            self.shared_cov = cov_shared / X.shape[0]
        else:
            self.cov = cov
            
        return

    # TODO: Implement this method!
    def predict(self, X_pred):
        probs = np.zeros(shape=(self.k, X_pred.shape[0]))

        for i in range(self.k):
            if self.is_shared_covariance:
                probs[i] = mvn.logpdf(X_pred, self.mu[i], self.shared_cov)
            else:
                probs[i] = mvn.logpdf(X_pred, self.mu[i], self.cov[i])

        # return np.argmax(probs.T + self.bias, axis=1)
        return np.argmax(probs.T + self.pi, axis=1)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        probs = np.ones(shape=(self.k, X.shape[0]))

        for i in range(self.k):
            if self.is_shared_covariance:
                probs[i] = mvn.logpdf(X, self.mu[i], self.shared_cov)
            else:
                probs[i] = mvn.logpdf(X, self.mu[i], self.cov[i])

        neg_log = probs.T[np.arange(X.shape[0]), y] + self.pi[y]
        return -np.sum(neg_log)
