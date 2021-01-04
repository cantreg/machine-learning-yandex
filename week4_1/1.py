
import pandas as pd
import numpy as np
import sklearn
import scipy
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

vectorizer = TfidfVectorizer(min_df=5)
enc = DictVectorizer()

def clean_data(filename):
    data = pd.read_csv(filename)
    data['FullDescription'] = data['FullDescription'].str.lower()
    data['FullDescription'] = data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
    data['LocationNormalized'].fillna('nan', inplace=True)
    data['ContractTime'].fillna('nan', inplace=True)
    return data


def get_x_y(filename, train=False, test=False):
    data = clean_data(filename)

    if train :
        description_vector = vectorizer.fit_transform(data['FullDescription'])
        x_train_categ = enc.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
    elif test:
        description_vector = vectorizer.transform(data['FullDescription'])
        x_train_categ = enc.transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))

    x = scipy.sparse.hstack((description_vector, x_train_categ))
    y = data['SalaryNormalized']
    return x, y


# def fill_zeros(original, join):
#     rows_count = original.shape[0]
#     columns_count = join.shape[1]
#     rows_indexes = []
#     for i in range(0, rows_count):
#         rows_indexes += ([i] * columns_count)
#     columns_indexes = join.col[:columns_count * rows_count]
#     none_data = np.full([columns_count * rows_count], 0)
#     zeroMatrix = scipy.sparse.coo_matrix((none_data, (rows_indexes, columns_indexes)), (rows_count, columns_count))
#     union = scipy.sparse.hstack((original, zeroMatrix))
#     return union


X, Y = get_x_y('salary-train.csv', train=True)
testX, testY = get_x_y('salary-test-mini.csv', test=True)

# testX = fill_zeros(testX, X)
# X = fill_zeros(X, testX)

ridge = Ridge(alpha=1, random_state=241)
ridge.fit(X, Y)

predict = ridge.predict(testX)
print(predict)
