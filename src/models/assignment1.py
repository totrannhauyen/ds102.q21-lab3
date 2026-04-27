import numpy as np

class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=3):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for i in range(n_samples):
                condition = y[i] * (np.dot(X[i], self.w) - self.b) >= 1

                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y[i] * X[i])
                    self.b -= self.lr * y[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)
