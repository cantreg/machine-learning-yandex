
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas
import numpy as np

df = pandas.read_csv('data-logistic.csv', header=None)

# logreg = LogisticRegression(C=0.1,
#                              solver='saga',
#                              penalty='l1',
#                              tol=1e-5,
#                              class_weight={-1: 0, 1: 0},
#                              max_iter=10000,
#                              random_state=241)
values_X = df.values[:, 1:3]
values_Y = df.values[:, 0]
# logreg.fit(values_X, values_Y)
# probs = logreg.predict_proba(values_X)[:,1]
# score = roc_auc_score(values_Y, probs, average='weighted')
# print(score)


def sigmoid (x1, x2, w1, w2):
    return 1 / (1 + np.e**(-w1*x1 - w2*x2))


def logistic_loss(x1, x2, w1, w2, y, c, reg=False):
    return np.mean(np.log(1+np.e**(-y*(w1*x1+w2*x2)))) + reg*0.5*c*np.sum(np.array([w1, w2])**2)


def gradient_step(x1, x2, w1, w2, y, c, k, reg=False):
    return w1 + k*np.mean(y*x1*(1-(1/(1+np.e**(-y*(w1*x1+w2*x2)))))) - reg*w1*c*k, \
           w2 + k*np.mean(y*x2*(1-(1/(1+np.e**(-y*(w1*x1+w2*x2)))))) - reg*w2*c*k

def fit():

    c = 10;
    k = 0.1;
    y = values_Y;
    m = len(values_Y)
    x1 = values_X[:, 0]
    x2 = values_X[:, 1]
    w1 = np.zeros(m)
    w2 = np.zeros(m)
    n_iter = 1000

    for epoch in range(n_iter):
        loss = logistic_loss(x1, x2, w1, w2, y, c)
        w1, w2 = gradient_step(x1, x2, w1, w2, y, c, k)
        if epoch % 100 == 0:
            print(loss)

    predictions = []
    probs = sigmoid(x1, x2, w1, w2)
    for i in probs:
        if i > 0.5:
            predictions.append(1)
        else:
            predictions.append(-1)

    print(y)
    print(predictions)
    print(probs)
    score = roc_auc_score(y, probs, average='weighted')
    print(score)

fit()
