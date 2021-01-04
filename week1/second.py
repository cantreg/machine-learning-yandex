import pandas
import numpy as np

from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

data = data[:][['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].dropna().replace('female', 0).replace('male', 1)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(data[:][['Pclass', 'Fare', 'Age', 'Sex']], data['Survived'])

importances = clf.feature_importances_

print(importances)