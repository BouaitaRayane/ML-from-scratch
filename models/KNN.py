import numpy as np
from base import ClassificationModel

def distance_euclidienne (x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN(ClassificationModel):
    """ K-Nearest Neighboors classifier.

    Parameters:
    -----------
    k: int
       The number of closest neighboors that we will take into account to determine the class of the sample we want to predict 
    
    """
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self._predict(x))
        return np.array(y_pred)

    def _predict(self, x):
        distances = []
        for xi in self.X_train:
            distances.append(distance_euclidienne(x, xi))
        k_closest = np.argsort(distances)[:self.k]
        closest_class = np.bincount(self.y_train[k_closest].astype(int)).argmax()
        return closest_class