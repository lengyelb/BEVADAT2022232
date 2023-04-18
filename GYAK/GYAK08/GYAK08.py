import numpy as np


class LinearRegression:
    def __init__(self, epochs: int = 1000, lr: float = 1e-3):
        self.epochs = epochs
        self.lr = lr

        self.m = 0
        self.c = 0

    def fit(self, x: np.array, y: np.array):
        n = float(len(x))  # Number of elements in X

        # Performing Gradient Descent
        losses = []
        for i in range(self.epochs):
            y_pred = self.m * x + self.c

            residuals = y - y_pred
            loss = np.sum(residuals ** 2)
            losses.append(loss)
            D_m = (-2 / n) * sum(x * residuals)
            D_c = (-2 / n) * sum(residuals)
            self.m = self.m - self.lr * D_m
            self.c = self.c - self.lr * D_c

    def predict(self, x):
        return self.m * x + self.c

    def evaluate(self, x_test, y_test):
        return np.mean((self.predict(x_test) - y_test) ** 2)
