import pandas
import numpy
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.datasets
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler


def write_answer(num, ans):
    f = open('output'+num+'.txt', 'w')
    f.write(ans)


data = pandas.read_csv('wine.data.txt', header=None, index_col=0)
label = data.index
features = data.values


def get_scores(features, left, right):
    result = []
    for i in range(left, right):
        classifier = sklearn.neighbors.KNeighborsClassifier(i)
        ##classifier.fit(features, label)
        cv = sklearn.model_selection.KFold(5, True, 42)
        cross_val_scores = sklearn.model_selection.cross_val_score(X=features, y=label, estimator=classifier, cv=cv)
        result.append((i, numpy.mean(cross_val_scores)))
    return result


scores = get_scores(features, 1, 51)
print(scores)

features = sklearn.preprocessing.scale(features)

scores = get_scores(features, 1, 51)
print(scores)

boston = sklearn.datasets.load_boston()
boston.data = sklearn.preprocessing.scale(boston.data)

reg_result = []
for p in numpy.linspace(1, 10, 200):
    regressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
    cv = sklearn.model_selection.KFold(5, True, 42)
    cross_val_scores = sklearn.model_selection.cross_val_score(X=boston.data,
                                                               y=boston.target,
                                                               estimator=regressor,
                                                               cv=cv,
                                                               scoring='neg_mean_squared_error')
    reg_result.append((p, numpy.mean(cross_val_scores)))

print(reg_result)

##
data_train = pandas.read_csv('perceptron-train.csv', header=None, index_col=0)
data_test = pandas.read_csv('perceptron-test.csv', header=None, index_col=0)
clf = Perceptron(random_state=241)
clf.fit(data_train.values, data_train.index)
predictions_unscaled = clf.predict(data_test.values)
score_unscaled = sklearn.metrics.accuracy_score(data_test.index, predictions_unscaled)

scaler = StandardScaler()
data_train_values_scaled = scaler.fit_transform(data_train.values)
data_test_values_scaled = scaler.transform(data_test.values)
clf = Perceptron(random_state=241)
clf.fit(data_train_values_scaled, data_train.index)
predictions_scaled = clf.predict(data_test_values_scaled)
score_scaled = sklearn.metrics.accuracy_score(data_test.index, predictions_scaled)


print(score_scaled - score_unscaled)

