import pandas
from sklearn.svm import SVC

train = pandas.read_csv('svm-data.csv',
                        header=0,
                        index_col=None,
                        names=['Y', 'A', 'B'])

X = train[['A', 'B']]
Y = train['Y']

svc = SVC(C=100000, kernel='linear', random_state=241)

svc.fit(X, Y)

print svc.support_vectors_
