import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from LinearRegressionSkeleton import LinearRegression

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

x_example = df['petal width (cm)'].values
y_example = df['sepal length (cm)'].values

x_train, x_test, y_train, y_test = train_test_split(x_example, y_example, test_size=0.2, random_state=42)

ln_model = LinearRegression()
ln_model.fit(x_example, y_example)

y_pred = ln_model.predict(x_example)
eva = ln_model.evaluate(x_test, y_test)

print(eva)

plt.scatter(x_test, y_test)
plt.plot([min(x_test), max(x_test)], [min(y_pred), max(y_pred)], color='red')
plt.show()
