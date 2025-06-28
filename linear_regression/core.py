# Homemade Linear Regression class
import numpy as np

class MyLinearRegression:
    def __init__(self):
        self.theta = np.array([])
    
    def fit(self, X, y, fit_intercept=False, ridge_lambda=0.0, regularize_bias=False, 
            lasso_lambda=0.0, max_iter=1000):
        X = np.array(X)
        if fit_intercept:
            X = np.array([[1.0] + list(row) for row in X])
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


