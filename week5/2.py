import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import math as m


def sigma_generator(y_generator, X_test):
    for y in y_generator(X_test):
        yield 1 / (1 + m.e ** ((-1) * y))


def draw_plot(x_data, y_data):
    line, axes = plt.subplots()
    axes.set_title(learning_rate)
    axes.plot(x_data, y_data)
    plt.show()


def find_min_loss(gen, y_test):
    x_data = []
    y_data = []
    min_loss = 1
    min_loss_i = 0
    for i, y_pred in enumerate(gen):
        loss = log_loss(y_test, y_pred)
        x_data.append(i + 1)
        y_data.append(loss)
        if loss < min_loss:
            min_loss = loss
            min_loss_i = i + 1
    return min_loss, min_loss_i


data = pd.read_csv('gbm-data.csv')

X = data.values[:, 1:]
y = data.values[:, :1]

for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=learning_rate)
    clf.fit(X=X_train, y=y_train)

    print("LEARNING RATE", learning_rate)
    min_loss, min_loss_i = find_min_loss(sigma_generator(clf.staged_decision_function, X_train), y_train)
    print("TRAIN min loss", min_loss, min_loss_i)
    min_loss, min_loss_i = find_min_loss(sigma_generator(clf.staged_decision_function, X_test), y_test)
    print("TEST min loss", min_loss, min_loss_i)

    clf = GradientBoostingClassifier(n_estimators=min_loss_i, verbose=True, random_state=241, learning_rate=learning_rate)
    clf.fit(X=X_train, y=y_train)
    print("TRAIN proba", log_loss(y_train, clf.predict_proba(X_train)))
    print("TEST proba", log_loss(y_test, clf.predict_proba(X_test)))

clf = RandomForestClassifier(n_estimators=37, random_state=241)
clf.fit(X=X_train, y=y_train)
print("TEST forest proba", log_loss(y_test, clf.predict_proba(X_test)))


