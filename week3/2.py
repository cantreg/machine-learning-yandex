# Hello World program in Python

from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import pandas

print("loading..")
newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv, n_jobs=8)

print("fit..")
gs.fit(X, y)
C = gs.best_estimator_.C;
print(f"best: {C}")

clf = SVC(C=C, kernel='linear', random_state=241)
clf.fit(X,y)

n = 10
max_coef_data_indicies = np.argpartition(np.abs(clf.coef_.data), -10)[-10:]
max_coef_feature_indices = clf.coef_.indices[max_coef_data_indicies]
names = np.array(vectorizer.get_feature_names())
top_10_words = names[max_coef_feature_indices]
print(np.sort(top_10_words))

# words = vectorizer.get_feature_names()
# coef = pandas.DataFrame(clf.coef_.data, clf.coef_.indices)
# top_words = np.abs(coef).sort_values(by=0,ascending=False).head(10).index.map(lambda i: words[i])
# print(1, ','.join(np.sort(top_words)))
