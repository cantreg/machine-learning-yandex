
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import pandas
import numpy as np

df = pandas.read_csv('scores.csv')

probs_log = df.values[:, 1]
probs_svm = df.values[:, 2]
probs_knn = df.values[:, 3]
probs_tre = df.values[:, 4]
values_Y = df.values[:, 0]

score_log = roc_auc_score(values_Y, probs_log, average='weighted')
score_svm = roc_auc_score(values_Y, probs_svm, average='weighted')
score_knn = roc_auc_score(values_Y, probs_knn, average='weighted')
score_tre = roc_auc_score(values_Y, probs_tre, average='weighted')
print(score_log)
print(score_svm)
print(score_knn)
print(score_tre)

curve_log = precision_recall_curve(values_Y, probs_log)
curve_svm = precision_recall_curve(values_Y, probs_svm)
curve_knn = precision_recall_curve(values_Y, probs_knn)
curve_tre = precision_recall_curve(values_Y, probs_tre)

recall_top_70_log = curve_log[0][np.where(curve_log[1] >= 0.7)]
recall_top_70_svm = curve_svm[0][np.where(curve_svm[1] >= 0.7)]
recall_top_70_knn = curve_knn[0][np.where(curve_knn[1] >= 0.7)]
recall_top_70_tre = curve_tre[0][np.where(curve_tre[1] >= 0.7)]
print(np.max(recall_top_70_log))
print(np.max(recall_top_70_svm))
print(np.max(recall_top_70_knn))
print(np.max(recall_top_70_tre))