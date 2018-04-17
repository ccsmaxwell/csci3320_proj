import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from naive_bayes import NaiveBayes

def cvTrain(modal, X, y):
	skf_list = list(StratifiedKFold(n_splits=10, random_state=3320, shuffle=True).split(X, y))
	max_score, max_model = -1, None
	for train_index, test_index in skf_list:
		temp_max_model = modal.fit(X[train_index], y[train_index])
		temp_score = temp_max_model.score(X[test_index], y[test_index])
		if(temp_score > max_score):
			max_score = temp_score
			max_model = temp_max_model
	return max_model

def predictionToResult(df, y):
	raceSize = df.groupby('race_id').finishing_position.max()

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

def predictEval(y_true, y_pred):
	result = {}
	result['precision'] = [precision_score(y_true[:,i], y_pred[:,i]) for i in range(len(y_true[0]))]
	result['recall'] = [recall_score(y_true[:,i], y_pred[:,i]) for i in range(len(y_true[0]))]
	result['f1'] = [f1_score(y_true[:,i], y_pred[:,i]) for i in range(len(y_true[0]))]
	result = pd.DataFrame(result, index=['HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent'])
	return result

df_train = pd.read_csv('data/training.csv')
train_X = df_train[['actual_weight','declared_horse_weight','draw','win_odds','recent_ave_rank','jockey_ave_rank','trainer_ave_rank','race_distance']].values
train_Y = np.ravel(df_train[['finishing_position']].values)

# 3.1.1
lr_model = LogisticRegressionCV(cv=10, random_state=3320)
lr_model.fit(train_X, train_Y)

# 3.1.2
skf_list = list(StratifiedKFold(n_splits=10, random_state=3320, shuffle=True).split(train_X, train_Y))
nb_model = cvTrain(GaussianNB(), train_X, train_Y)

clf = NaiveBayes()
clf = clf.fit(train_X, train_Y)

# 3.1.3
svm_model = cvTrain(SVC(kernel="rbf",random_state=3320), train_X, train_Y)

# 3.1.4
rf_model = cvTrain(RandomForestClassifier(random_state=3320), train_X, train_Y)

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

predict_clf = predictionToResult(df_test, clf.predict(test_X))
df_clf = resultToDf(df_test, predict_clf)

predict_svm = predictionToResult(df_test, svm_model.predict(test_X))
df_svm = resultToDf(df_test, predict_svm)
df_svm.to_csv('predictions/svm_predictions.csv', index=False)

predict_rf = predictionToResult(df_test, rf_model.predict(test_X))
df_rf = resultToDf(df_test, predict_rf)
df_rf.to_csv('predictions/rf_predictions.csv', index=False)

# 3.3
predict_true = predictionToResult(df_test, test_Y)

eval_lr = predictEval(predict_true, predict_lr)
print("lr\n", eval_lr)

eval_nb = predictEval(predict_true, predict_nb)
print("nb\n", eval_nb)

eval_clf = predictEval(predict_true, predict_clf)
print("clf\n", eval_clf)

eval_svm = predictEval(predict_true, predict_svm)
print("svm\n", eval_svm)

eval_rf = predictEval(predict_true, predict_rf)
print("rf\n", eval_rf)