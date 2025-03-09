import numpy as np
from base import Model, RegressionModel

class LinearRegression(RegressionModel):
    def __init__(self, lr=0.01, n_iters=500):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (2/n_samples) * np.dot(X.T, y_pred-y)
            db = (2/n_samples) * np.sum(y_pred-y)

            self.weights -= self.lr*dw
            self.bias -= self.lr*db
 
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
class RidgeLinearregression(RegressionModel):
    def __init__(self, lr=0.01, n_iters=1000, alpha=1.0):
        self.lr = lr
        self.n_iters = n_iters
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape 

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias 

            dw = (2/n_samples) * np.dot(X.T, (y_pred - y)) + 2*self.alpha*self.weights
            db = (2/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias 
    
class LassoLinearregression(RegressionModel):
    def __init__(self, lr=0.01, n_iters=1000, alpha=1.0):
        self.lr = lr
        self.n_iters = n_iters
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape 

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias 

            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
            db = (2/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            self.weights = np.sign(self.weights) * np.maximum(self.weights - self.lr*self.alpha, 0)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias 
    
class ElasticNet(RegressionModel):
    def __init__(self, lr=0.01, n_iters=1000, alpha=1.0, l2_ratio=0.5):
        self.lr = lr
        self.n_iters = n_iters
        self.alpha = alpha
        self.l2_ratio = l2_ratio
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape 

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias 

            dw = (2/n_samples) * np.dot(X.T, (y_pred - y)) + 2*self.l2_ratio*self.weights
            db = (2/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            self.weights = np.sign(self.weights) * np.maximum(np.abs(self.weights) - self.lr*self.alpha, 0)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias 