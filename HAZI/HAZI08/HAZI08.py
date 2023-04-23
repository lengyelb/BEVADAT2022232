import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pandas.core.frame
import sklearn.utils

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression


def load_iris_data() -> sklearn.utils.Bunch:
    return load_iris()


def check_data(iris: sklearn.utils.Bunch) -> pandas.core.frame.DataFrame:
    return pd.DataFrame(data=iris.data, columns=iris.feature_names).head()


def linear_train_data(iris: sklearn.utils.Bunch) -> (np.ndarray, np.ndarray):
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    X = iris_df.drop(['sepal length (cm)'], axis=1).values
    y = iris_df['sepal length (cm)'].values
    return X, y


def logistic_train_data(iris: sklearn.utils.Bunch) -> (np.ndarray, np.ndarray):
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target

    X = iris_df.drop(['target'], axis=1).values
    y = iris_df['target'].values

    return X[y != 2], y[y != 2]


def split_data(X, y) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_linear_regression(X_train, y_train):  # -> sklearn.linear_model._base.LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train):  # -> sklearn.linear_model._base.LogisticRegression:
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def predict(model, X_test) -> np.ndarray:
    return model.predict(X_test)



def plot_actual_vs_predicted(y_test, y_pred):  # -> plt.figure.Figure:
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted Target Values')
    return fig


def evaluate_model(y_test, y_pred):  # -> float:
    return np.mean((y_pred - y_test) ** 2)
