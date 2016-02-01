from __future__ import division
import os
import pandas

script_dir = os.path.dirname(__file__)
passengers = pandas.read_csv(os.path.join(script_dir, '../titanic.csv'), index_col='PassengerId')
total_passengers = len(passengers)

print passengers['Sex'].value_counts()  # 577 314

survived = passengers['Survived'].value_counts()[1]
print round(survived / total_passengers * 100, 2)  # 38.38

first_class = passengers['Pclass'].value_counts()[1]
print round(first_class / total_passengers * 100, 2)  # 24.24

ages = passengers['Age']
print round(ages.mean(), 2)  # 29.70
print round(ages.median(), 2)  # 28.0

corr = passengers['SibSp'].corr(passengers['Parch'])
print round(corr, 2)  # 0.45

names = []
for i, r in passengers.iterrows():
    names.append(r['Name'].split('.')[1].strip().split(' ')[0].strip().replace('(', '').replace(')', ''))

passengers['FirstName'] = names
print passengers[passengers['Sex'] == 'female'].groupby('FirstName').size()