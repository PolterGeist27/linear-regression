import numpy as np

class LinearRegression:

    def __init__(self, learning_rate = 0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    # Training
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        # Predict result
        y_pred = np.dot(X, self.weights) + self.bias


    # Prediction
    def predict():
        pass
