# Hello World program in Python

from sklearn.svm import SVC
import pandas as pd

df = pd.read_csv('data.csv', header=None)
svc = SVC(C=100000, kernel='linear', random_state=241)

svc.fit(df[[1, 2]], df[[0]].values.ravel())

print(svc.support_)