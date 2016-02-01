from __future__ import division
from sklearn.tree import DecisionTreeClassifier
import os
import pandas

script_dir = os.path.dirname(__file__)
passengers = pandas.read_csv(os.path.join(script_dir, '../titanic.csv'), index_col='PassengerId')

passengers_cut = passengers[['Fare', 'Pclass', 'Age', 'Sex', 'Survived']].dropna(axis=0)
survived = passengers_cut['Survived']
passengers_cut = passengers_cut[['Fare', 'Pclass', 'Age', 'Sex']]

for i, r in passengers_cut.iterrows():
    passengers_cut.loc[i, "Sex"] = (1 if r['Sex'] == 'male' else 0)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(passengers_cut, survived)

print clf.feature_importances_
