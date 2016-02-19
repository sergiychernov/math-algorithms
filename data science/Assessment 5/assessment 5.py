import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def get_predition_score(x_train, x_test, y_train, y_test):
    clf = Perceptron(random_state=241)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)

    return accuracy_score(y_test, predictions)


train = pandas.read_csv('perceptron-train.csv',
                        header=0,
                        index_col=None,
                        names=['Y', 'A', 'B'])

X_train = train[['A', 'B']]
Y_train = train['Y']

test = pandas.read_csv('perceptron-test.csv',
                       header=0,
                       index_col=None,
                       names=['Y', 'A', 'B'])
X_test = test[['A', 'B']]
Y_test = test['Y']

without_scale = get_predition_score(X_train, X_test, Y_train, Y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with_scale = get_predition_score(X_train_scaled, X_test_scaled, Y_train, Y_test)

print round(with_scale - without_scale, 3)
