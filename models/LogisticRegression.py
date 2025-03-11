import numpy as np
from base import ClassificationModel

def sigmoide(x):
    return 1/(1+np.exp(-x))

class LogisticRegression(ClassificationModel):
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
            y_pred = sigmoide(np.dot(X, self.weights)+ self.bias)

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights -= self.lr*dw
            self.bias -= self.lr*db


    def predict(self, X):
        y_pred = sigmoide(np.dot(X, self.weights)+ self.bias)
        return (y_pred >= 0.5).astype(int)