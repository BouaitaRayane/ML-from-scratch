import numpy as np 

class Model:
    def __init__(self):
        pass

    def fit(self, X, y):
        raise NotImplementedError ("La méthode 'fit' n'est pas implémentée")
    
    def predict(self, X):
        raise NotImplementedError("La méthode 'predict' n'est pas implémentée")
    
    def score (self, X, y):
        raise NotImplementedError("La méthode 'score' n'est pas implémentée")
    
class RegressionModel(Model):
    def __init__(self):
        pass

    def fit(self, X, y):
        raise NotImplementedError ("La méthode 'fit' n'est pas implémentée")
    
    def predict(self, X):
        raise NotImplementedError("La méthode 'predict' n'est pas implémentée")
    
    def score (self, X, y):
        y_pred = self.predict(X)

        y = np.array(y).ravel()
        y_pred = np.array(y_pred).ravel()

        mse = np.mean((y-y_pred)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(abs(y-y_pred))
        r2 = 1 - ((np.sum((y-y_pred)**2))/(np.sum((y - np.mean(y))**2)))
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

class ClassificationModel(Model):
    def __init__(self):
        pass

    def fit(self, X, y):
        raise NotImplementedError ("La méthode 'fit' n'est pas implémentée")
    
    def predict(self, X):
        raise NotImplementedError("La méthode 'predict' n'est pas implémentée")

    def score(self, X, y):
        y_pred = self.predict(X)

        y = np.array(y).ravel()
        y_pred = np.array(y_pred).ravel()

        TP = np.sum((y==1)&(y_pred==1)) 
        TN = np.sum((y==0)&(y_pred==0)) 
        FP = np.sum((y==0)&(y_pred==1))
        FN = np.sum((y==1)&(y_pred==0))

        precision = TP / (TP + FP) if TP+FP != 0 else 0
        recall = TP / (TP + FN) if TP+FN != 0 else 0
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        F1 = (2*precision*recall)/(precision+recall) if precision+recall != 0 else 0

        return {"Accuracy": accuracy, "Recall": recall, "F1-score": F1, "Precision": precision}