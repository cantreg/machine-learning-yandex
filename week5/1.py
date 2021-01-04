import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score


def check_n_trees(n):
    kf = KFold(random_state=1, shuffle=True, n_splits=5)
    clf = RandomForestRegressor(n_estimators=n, random_state=1)
    score = cross_val_score(estimator=clf, X=X, y=y.ravel(), cv=kf, scoring=r2_scorer)
    return score


def r2_scorer(clf, X_test, y_test):
    predictions = clf.predict(X_test)
    r2 = r2_score(y_test, predictions)
    return r2


data = pd.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1:].to_numpy()

for n in range(1, 50):
    scores = check_n_trees(n)
    mean_score = sum(scores) / len(scores)
    if mean_score > 0.52:
        print(n)
