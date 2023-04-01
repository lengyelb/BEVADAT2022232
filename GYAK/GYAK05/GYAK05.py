from typing import Optional, Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot
from scipy.stats import mode
from sklearn.metrics import confusion_matrix


class KNNClassifier:
    def __init__(self, k: int, test_split_ratio: float):
        self.k = k
        self.test_split_ratio = test_split_ratio

        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

        self.x_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        self.y_preds: Optional[np.ndarray] = None

    @staticmethod
    def load_csv(path: str, clean_dataset: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(42)
        dataset = np.genfromtxt(path, delimiter=',')
        np.random.shuffle(dataset)
        x, y = dataset[:, :4], dataset[:, -1]

        if clean_dataset:
            x[np.isnan(x)] = 3.5
            y[np.isnan(y)] = 3.5

            y = np.delete(y, np.where(x < 0)[0], axis=0)
            y = np.delete(y, np.where(x > 10)[0], axis=0)

            x = np.delete(x, np.where(x < 0)[0], axis=0)
            x = np.delete(x, np.where(x > 10)[0], axis=0)

        return x, y

    def train_test_split(self, features: np.ndarray, labels: np.ndarray):
        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size
        assert (len(features) == test_size + train_size)

        self.x_train, self.y_train = features[:train_size, :], labels[:train_size]
        self.x_test, self.y_test = features[train_size:train_size + test_size, :], \
            labels[train_size:train_size + test_size]

    def euclidean(self, element_of_x: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((self.x_train - element_of_x) ** 2, axis=1))

    def predict(self, x_test: np.ndarray) -> None:
        labels_pred = []
        for x_test_element in x_test:
            distances = self.euclidean(x_test_element)
            distances = np.array(sorted(zip(distances, self.y_train)))

            label_pred = mode(distances[:self.k, 1], keepdims=False).mode
            labels_pred.append(label_pred)

        self.y_preds = np.array(labels_pred, dtype=np.int32)

    def accuracy(self) -> float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test) * 100

    def confusion_matrix(self):
        conf_matrix = confusion_matrix(self.y_test, self.y_preds)
        sns.heatmap(conf_matrix, annot=True)

    @property
    def k_neighbors(self):
        return self.k


# # region testing
# csv_path = "datasets/iris.csv"
# x_test, y_test = KNNClassifier.load_csv(csv_path, True)
#
# knn = KNNClassifier(3, 0.2)
# knn.train_test_split(x_test, y_test)
# knn.predict(knn.x_test)
# print(knn.accuracy())
# knn.confusion_matrix()
# pyplot.show()
# # endregion
