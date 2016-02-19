from __future__ import division
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas


def get_k_score(x, cv, y):
    res = []
    for k in range(14, 15):
        neigh = KNeighborsClassifier(n_neighbors=k)
        arr = cross_val_score(neigh, x, y, cv=cv)
        a = round(arr.sum() / len(arr), 2)
        print arr
        res.append(a)

    return np.array(res)


def cross_score_validate(classifier, x, y):
    # kf = KFold(len(y), n_folds=5, shuffle=True, random_state=42)
    kf = KFold(len(y))
    arr = cross_val_score(classifier, x, y, cv=kf)
    a = round(arr.sum() / len(arr), 2)
    print a


def columns_to_dictionary(src, column_names):
    for column_name in column_names:
        dates = {v: k for k, v in dict(enumerate(pandas.unique(src[column_name].ravel()))).items()}
        print 'total {} {}'.format(column_name, len(dates))
        src[column_name] = src[column_name].map(dates)

    return src


def pre_process_data(src, transform=True):
    y = []
    if 'Category' in src.columns:
        columns_to_dictionary(src, ['Category'])
        y = src['Category']

    columns = ['Address', 'PdDistrict', 'DayOfWeek']

    # src.Dates = src.Dates.str.split(' ').str.get(1)

    columns_to_dictionary(src, columns)

    # print src.head()

    src['Day'] = src['Dates'].dt.day
    src['Month'] = src['Dates'].dt.month
    src['Year'] = src['Dates'].dt.year
    src['Hour'] = src['Dates'].dt.hour
    src['WeekOfYear'] = src['Dates'].dt.weekofyear

    x = src[['Address', 'PdDistrict', 'DayOfWeek', 'Day', 'Month', 'Year', 'Hour', 'WeekOfYear']]

    if transform:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    return x, y


def k_fold(x, y):
    kf = KFold(len(y), n_folds=5, shuffle=True, random_state=42)
    ans_scaled = get_k_score(x, kf, y)
    print ans_scaled.max(), ans_scaled.argmax()


def k_tree(x, y):
    clf = DecisionTreeClassifier(random_state=241)
    clf.fit(x, y)
    print clf.feature_importances_


def support_vector_classification(x, y):
    svc = SVC(C=100000, kernel='linear', random_state=241)
    svc.fit(x, y)
    return svc


def k_neighbors(x, y):
    neigh = KNeighborsClassifier(n_neighbors=40)
    neigh.fit(x, y)
    return neigh


def random_forest(x, y):
    forest = RandomForestClassifier(n_estimators=10)
    forest.fit(x, y)
    return forest


if __name__ == '__main__':
    crimes_train = pandas.read_csv('train.csv', parse_dates=['Dates'])
    crimes_test = pandas.read_csv('test.csv', parse_dates=['Dates'])

    x_train, y_train = pre_process_data(crimes_train, transform=False)
    x_test, y_test = pre_process_data(crimes_test, transform=False)

    algo = random_forest(x_train, y_train)
    cross_score_validate(algo, x_train, y_train)
    # res = algo.predict(x_test)



    # x_test = pre_process_data(crimes_test)
    # k_tree(x_train, y_train)
    # support_vector_classification(x_train, y_train)
    # k_fold(x_train, y_train)
