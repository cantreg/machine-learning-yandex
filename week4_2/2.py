
import pandas as pd
import numpy as np
import sklearn.decomposition
import scipy

stocks = pd.read_csv('close_prices.csv')
index = pd.read_csv('djia_index.csv')

pca = sklearn.decomposition.PCA(10)
stocks_ = stocks.iloc[:, 1:]
stocks_tr = pca.fit_transform(stocks_)
ratio_sum = pca.explained_variance_ratio_.sum() # 4 компоненты достаточно для 90%
print(ratio_sum)
print(stocks_.columns[pca.components_[0].argmax()])

stack = np.vstack([index.iloc[:, 1], stocks_tr[:, 0]])
corrcoef = np.corrcoef(stack)
print(corrcoef[0, 1])

