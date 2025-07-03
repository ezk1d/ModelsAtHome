# Homemade Linear Regression class
import numpy as np
import math

class MyLinearRegression:
    def __init__(self):
        self.theta = np.array([])
    
    def fit(self, X, y, fit_intercept=False, ridge_lambda=0.0, regularize_bias=False, 
            lasso_lambda=0.0, max_iter=1000):
        X = np.array(X)
        y = np.array(y)
        if fit_intercept:
           X = np.hstack((np.ones((X.shape[0], 1)), X))
        I = np.eye(X.shape[1])
        if not regularize_bias:
            I[0,0] = 0
        if lasso_lambda > 0:
            self.theta = np.zeros(X.shape[1])
            for _ in range(max_iter):
                theta_old = self.theta.copy()
                for j in range(X.shape[1]):
                    pred = X @ self.theta
                    residual = y - pred + X[:, j] * self.theta[j]
                    rho = np.dot(X[:, j], residual)
                    if j == 0:
                        self.theta[j] = rho / np.sum(X[:, j] ** 2)
                    else:
                        z = np.sum(X[:, j] ** 2)
                        if rho < -lasso_lambda:
                            self.theta[j] = (rho + lasso_lambda) / z
                        elif rho > lasso_lambda:
                            self.theta[j] = (rho - lasso_lambda) / z
                        else:
                            self.theta[j] = 0.0
                if np.linalg.norm(self.theta - theta_old, ord=1) < 0.0001:
                    break
        else:
            self.theta = np.linalg.inv(X.T @ X + ridge_lambda * I) @ X.T @ y
        self.X = X
        self.y = y
        
    def predict(self, X):
        X = np.array(X)
        if self.X.shape[1] == X.shape[1] + 1:
            X = np.column_stack([np.ones(X.shape[0]), X])
        return X @ self.theta

    def mse(self, X, y):
           return sum((self.predict(X[i]) - y[i]) ** 2 for i in range(len(y))) / len(y)
    
    def rmse(self, X, y):
        return self.mse(self, X, y) ** (1/2)

    def r_squared(self, X, y):
        y_mean = sum(y) / len(y)
        rss = sum((self.predict(X[i]) - y[i]) ** 2 for i in range(len(y)))
        tss = sum((y[i]-y_mean) ** 2 for i in range(len(y))) + 1e-8
        if tss == 0:
            return 1
        return 1 - (rss / tss) 


# Homemade Logistic Regression Class
class MyLogisticRegression:
    def __init__(self):
        self.theta = np.array([])

    def _initialize_arrays(self, X, y, fit_intercept):
        X = np.array(X)
        y = np.array(y)
        if fit_intercept:
             X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.theta = np.zeros(X.shape[1])
        self.X = X
        self.y = y

    def fit(self, X, y, fit_intercept=False, alpha=0.1, epochs=1000, early_stoppage=True, limit=10,
            min_delta=1e-4, batch_size=None, decay_rate=0.001):
        self._initialize_arrays(X, y, fit_intercept)
        X = self.X
        y = self.y
        best = float('inf')
        wait = 0
        self.loss = []
        if batch_size == None:
            batch_size = len(X)
        for epoch in range(epochs):
            indices = np.random.permutation(len(y))
            X, y = X[indices], y[indices]
            epoch_loss = 0
            for batch_start in range(0, len(X), batch_size):
                batch_end = batch_start + batch_size
                if batch_end > len(X):
                    batch_end = len(X)
                batch_X = X[batch_start:batch_end]
                batch_y = y[batch_start:batch_end]
                loss = self._get_gradients(batch_X,batch_y,alpha)
                epoch_loss += loss
                if early_stoppage:
                    if (best - loss > min_delta):
                        best = loss
                        wait = 0
                    else:
                        wait += 1
                        if wait == limit:
                            print("Early Stoppage reached")
                            return
            self.loss.append(epoch_loss / math.ceil(len(X) / batch_size))
            alpha = alpha / (1 + decay_rate * epoch)

    def _get_gradients(self, X, y, alpha):
        preds = self._get_prob(X)
        gradient = (1 / len(y)) * X.T @ (preds - y)
        self.theta -= alpha * gradient
        loss = -np.mean(y * np.log(preds + 1e-8) + (1-y) * np.log(1 - preds + 1e-8))
        return loss

    def _get_prob(self, X):
        z = X @ self.theta
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def plot_loss(self):
        import matplotlib.pyplot as plt
        epochs = [x for x in range(len(self.loss))]
        plt.scatter(epochs, self.loss, label="Loss")
        plt.legend()
        plt.xlabel("# of Epochs")
        plt.ylabel("Amount of loss")
        plt.title("Loss across Epochs")
        plt.show()
    
    def predict_prob(self, X):
        X = np.array(X)
        if self.X.shape[1] == X.shape[1] + 1:
            X = np.column_stack([np.ones(X.shape[0]), X])
        return self._get_prob(X)

    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)
    
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
    
    # Evaluation Metrics
    def precision(self, X, y):
        # TP / (TP + FP)
        preds = self.predict(X)
        TP = 0
        FP = 0
        for pred, yi in zip(preds, y):
            if pred == 1:
                if yi == 1:
                    TP += 1
                else:
                    FP += 1
        return TP / (TP + FP) if (TP + FP) > 0 else 0
    
    def recall(self, X, y):
        # TP / (TP + FN)
        preds = self.predict(X)
        TP = 0
        FN = 0
        for pred, yi in zip(preds, y):
            if yi == 1:
                if pred == 1:
                    TP += 1
                else:
                    FN += 1
        return TP / (TP + FN) if (TP + FN) > 0 else 0
    
    def f1_score(self, X, y):
        precision = self.precision(X,y)
        recall = self.recall(X,y)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    def evaluate(self, X, y, verbose=True):
        accuracy = self.score(X, y)
        precision = self.precision(X, y)
        recall = self.recall(X, y)
        f1 = self.f1_score(X, y)

        if verbose:
            print(f"Evaluation Results:")
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    
    # For Model Serialization
    def save(self, path="model.npy"):
        np.save(path, self.theta)

    def load(self, path="model.npy"):
        self.theta = np.load(path)

