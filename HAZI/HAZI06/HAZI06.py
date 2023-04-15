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

classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=4)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test, Y_pred))

"""
A tanításom elég könyen ment. Nehézségekbe nem nagyon ütköztem. Az egyetlen
ami érdekes volt hogy a may depth előszőr túl nagy értékeket engedtem meg neki,
így errort kaptam a modellben. Miután ezt javítottam már minden jó volt.
A legjobb értékek kereséséhez grid search-öt használtam, min sample split 1-5,
max depth pedig 1-4 range-el. Érdekes mód azt találtam hogy a min sample split
nem befolyásolta az eredményeket ebben a rangeben. A max depth azonban igen, itt 4-nél
kaptam a legjobb eredményeket.

min_sample_split, max_depth, accuracy:
1, 4, 0.7849166666666667
2, 4, 0.7849166666666667
3, 4, 0.7849166666666667
4, 4, 0.7849166666666667
5, 4, 0.7849166666666667
1, 3, 0.7839166666666667
2, 3, 0.7839166666666667
3, 3, 0.7839166666666667
4, 3, 0.7839166666666667
5, 3, 0.7839166666666667
1, 2, 0.7823333333333333
2, 2, 0.7823333333333333
3, 2, 0.7823333333333333
4, 2, 0.7823333333333333
5, 2, 0.7823333333333333
1, 1, 0.7773333333333333
2, 1, 0.7773333333333333
3, 1, 0.7773333333333333
4, 1, 0.7773333333333333
5, 1, 0.7773333333333333
"""
