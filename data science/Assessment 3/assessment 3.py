import numpy as np
import pandas
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale


def get_k_score(x, cv, y):
    res = []
    for k in range(1, 51):
        neigh = KNeighborsClassifier(n_neighbors=k)
        arr = cross_val_score(neigh, x, y, cv=cv)
        a = round(arr.sum() / len(arr), 2)
        res.append(a)
    print res
    return np.array(res)

if __name__ == '__main__':
    df = pandas.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
                         header=0,
                         index_col=None,
                         names=['CLS', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D', 'F'])

    Y = df['CLS'].as_matrix()
    X = df[['W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D', 'F']].as_matrix()

    kf = KFold(len(Y), n_folds=5, shuffle=True, random_state=42)

    ans = get_k_score(X, kf, Y)
    print ans.max(), ans.argmax()

    ans_scaled = get_k_score(scale(X), kf, Y)
    print ans_scaled.max(), ans_scaled.argmin()


