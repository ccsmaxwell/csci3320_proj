import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from naive_bayes import NaiveBayes

def predictionToResult(df, y):
	raceSize = df.groupby('race_id').size()

	result = []
	for i in range(len(y)):
		recResult = []
		recResult.append(1 if y[i] == 1 else 0)
		recResult.append(1 if y[i] <= 3 else 0)
		recResult.append(1 if y[i] <= raceSize[df.iloc[i].race_id]/2 else 0)
		result.append(recResult)

	return np.array(result)

def resultToDf(df, result):
	df_out = df[['race_id','horse_id']]
	df_out.columns = ['RaceID', 'HorseID']
	df_result = pd.DataFrame(result, columns=['HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent'])
	df_out = pd.concat([df_out, df_result], axis=1)
	return df_out

df_train = pd.read_csv('data/training.csv')
train_X = df_train[['actual_weight','declared_horse_weight','draw','win_odds','recent_ave_rank','jockey_ave_rank','trainer_ave_rank','race_distance']].values
train_Y = np.ravel(df_train[['finishing_position']].values)

# 3.1.1
lr_model = LogisticRegressionCV(cv=10, random_state=3320)
lr_model.fit(train_X, train_Y)

# 3.1.2
skf_list = list(StratifiedKFold(n_splits=10, random_state=3320, shuffle=True).split(train_X, train_Y))
max_score, nb_model = -1, None
for train_index, test_index in skf_list:
	temp_nb_model = GaussianNB().fit(train_X[train_index], train_Y[train_index])
	temp_score = temp_nb_model.score(train_X[test_index], train_Y[test_index])
	if(temp_score > max_score):
		max_score = temp_score
		nb_model = temp_nb_model

clf = NaiveBayes()
clf = clf.fit(train_X, train_Y)
# y_predict = clf.predict(X)

# 3.1.3
max_score, svm_model = -1, None
for train_index, test_index in skf_list:
	temp_svm_model = SVC(kernel="rbf",random_state=3320).fit(train_X[train_index], train_Y[train_index])
	temp_score = temp_svm_model.score(train_X[test_index], train_Y[test_index])
	if(temp_score > max_score):
		max_score = temp_score
		svm_model = temp_svm_model

# 3.1.4
max_score, rf_model = -1, None
for train_index, test_index in skf_list:
	temp_rf_model = RandomForestClassifier(random_state=3320).fit(train_X[train_index], train_Y[train_index])
	temp_score = temp_rf_model.score(train_X[test_index], train_Y[test_index])
	if(temp_score > max_score):
		max_score = temp_score
		rf_model = temp_rf_model

# 3.2
df_test = pd.read_csv('data/testing.csv')
test_X = df_test[['actual_weight','declared_horse_weight','draw','win_odds','recent_ave_rank','jockey_ave_rank','trainer_ave_rank','race_distance']].values
test_Y = np.ravel(df_test[['finishing_position']].values)

predict_lr = predictionToResult(df_test, lr_model.predict(test_X))
df_lr = resultToDf(df_test, predict_lr)
df_lr.to_csv('predictions/lr_predictions.csv', index=False)

predict_nb = predictionToResult(df_test, nb_model.predict(test_X))
df_nb = resultToDf(df_test, predict_nb)
df_nb.to_csv('predictions/nb_predictions.csv', index=False)

predict_svm = predictionToResult(df_test, svm_model.predict(test_X))
df_svm = resultToDf(df_test, predict_svm)
df_svm.to_csv('predictions/svm_predictions.csv', index=False)

predict_rf = predictionToResult(df_test, rf_model.predict(test_X))
df_rf = resultToDf(df_test, predict_rf)
df_rf.to_csv('predictions/rf_predictions.csv', index=False)

# 3.3
predict_true = predictionToResult(df_test, test_Y)