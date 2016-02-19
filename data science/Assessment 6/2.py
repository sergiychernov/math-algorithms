from sklearn import datasets
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn import grid_search
from sklearn.feature_extraction.text import TfidfVectorizer

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)
X = newsgroups.data
y = newsgroups.target

vectorized = TfidfVectorizer(stop_words='english')
my_features = vectorized.fit_transform(X, y)


grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(my_features, y)
