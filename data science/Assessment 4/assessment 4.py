import numpy as np
from sklearn.preprocessing import scale
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold

data = load_boston()
X = scale(data.data)
y = data.target

kf = KFold(len(y), n_folds=5, shuffle=True, random_state=42)
maxOccuracy = 0
maxP = 1
for p in np.linspace(1, 10, 200):
    neigh = KNeighborsRegressor(p=p, n_neighbors=5, weights='distance')
    arr = cross_val_score(neigh, X, y, cv=kf)
    a = round(max(arr), 2)
    if a >= maxOccuracy:
        maxOccuracy = a
        maxP = round(p, 2)

print maxOccuracy, maxP
