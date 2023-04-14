import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from NJCleaner import NJCleaner
from GYAK.GYAK06.GYAK06 import DecisionTreeClassifier


base_csv_path = "datasets/NJ_Transit+Amtrak.csv"
clean_csv_path = "datasets/NJ.csv"

nj_cleaner = NJCleaner(base_csv_path)
nj_cleaner.prep_df(clean_csv_path)

data = pd.read_csv(clean_csv_path)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

results = []

for min_sample_split in range(1, 6):
    for max_depth in range(1, 5):
        classifier = DecisionTreeClassifier(min_samples_split=min_sample_split, max_depth=max_depth)
        classifier.fit(X_train, Y_train)

        Y_pred = classifier.predict(X_test)
        intermediate_result = (min_sample_split, max_depth, accuracy_score(Y_test, Y_pred))
        print(intermediate_result)
        results.append(intermediate_result)

print(results)
print(max(results, key=lambda x: x[2]))
with open("datasets/results.txt", 'w') as results_file:
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    for result_line in sorted_results:
        results_file.write(', '.join(map(str, result_line)) + "\n")
