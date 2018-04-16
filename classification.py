import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB

from naive_bayes import NaiveBayes

df_train = pd.read_csv('data/training.csv')
train_X = df_train[['actual_weight','declared_horse_weight','draw','win_odds','recent_ave_rank','jockey_ave_rank','trainer_ave_rank','race_distance']].values
train_Y = np.ravel(df_train[['finishing_position']].values)

# 3.1.1
lr_model = LogisticRegressionCV(cv=10, random_state=3320)
lr_model.fit(train_X, train_Y)

# 3.1.2
nb_model = GaussianNB()
# skf = StratifiedKFold(n_splits=10, random_state=3320, shuffle=True)
# for train_index, test_index in skf.split(train_X, train_Y):
nb_model.fit(train_X, train_Y)

clf = NaiveBayes()
clf = clf.fit(train_X, train_Y)
# y_predict = clf.predict(X)

