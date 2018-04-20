import pandas as pd
import numpy as np
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from naive_bayes import NaiveBayes

TRAIN_FEATURE = ['actual_weight','declared_horse_weight','draw','win_odds','recent_ave_rank','jockey_ave_rank','trainer_ave_rank','race_distance']
PREDICT_COLUMN = ['HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent']

def yToPredictColumn(df, y):
	raceSize = df.groupby('race_id').finishing_position.max()

	result = []
	for i in range(len(y)):
		recResult = []
		recResult.append(1 if y[i] == 1 else 0)
		recResult.append(1 if y[i] <= 3 else 0)
		recResult.append(1 if y[i] <= raceSize[df.iloc[i].race_id]/2 else 0)
		result.append(recResult)

	return np.array(result)

def cvTrain(modal, X, y):
	skf_list = list(StratifiedKFold(n_splits=10, random_state=3320, shuffle=True).split(X, y))
	max_score, max_model = -1, None
	for train_index, test_index in skf_list:
		print("*", end="")
		temp_max_model = modal.fit(X[train_index], y[train_index])
		temp_score = temp_max_model.score(X[test_index], y[test_index])
		if(temp_score > max_score):
			max_score = temp_score
			max_model = temp_max_model
	print("")
	return max_model

def resultToDf(df, result):
	df_out = df[['race_id','horse_id']]
	df_out.columns = ['RaceID', 'HorseID']
	df_result = pd.DataFrame(result, columns=PREDICT_COLUMN)
	df_out = pd.concat([df_out, df_result], axis=1)
	return df_out

def predictEval(y_true, y_pred):
	result = {}
	result['precision'] = [precision_score(y_true[:,i], y_pred[:,i]) for i in range(len(y_true[0]))]
	result['recall'] = [recall_score(y_true[:,i], y_pred[:,i]) for i in range(len(y_true[0]))]
	result['f1'] = [f1_score(y_true[:,i], y_pred[:,i]) for i in range(len(y_true[0]))]
	result = pd.DataFrame(result, index=PREDICT_COLUMN)
	return result

def lr_model(train_X, train_Y, y_feature_index):
	model = LogisticRegressionCV(cv=10, random_state=3320)
	model.fit(train_X, train_Y[:,y_feature_index])
	return model

def nb_model(train_X, train_Y, y_feature_index):
	return cvTrain(GaussianNB(), train_X, train_Y[:,y_feature_index])

def nb_self_model(train_X, train_Y, y_feature_index):
	clf = NaiveBayes()
	clf = clf.fit(train_X, train_Y[:,y_feature_index])
	return clf

def svm_model(train_X, train_Y, y_feature_index):
	return cvTrain(SVC(kernel="rbf",random_state=3320), train_X, train_Y[:,y_feature_index])

def rf_model(train_X, train_Y, y_feature_index):
	return cvTrain(RandomForestClassifier(random_state=3320), train_X, train_Y[:,y_feature_index])

df_train = pd.read_csv('data/training.csv')
train_X = df_train[TRAIN_FEATURE].values
train_Y = yToPredictColumn(df_train,np.ravel(df_train[['finishing_position']].values))

# 3.1.1
print("Start LogisticRegression CV")
start = time.time()
lr = [lr_model(train_X, train_Y, i) for i in range(len(PREDICT_COLUMN))]
print("End LogisticRegression CV, Time: %s s" % (time.time() - start))

# 3.1.2
print("Start GaussianNB CV")
start = time.time()
nb = [nb_model(train_X, train_Y, i) for i in range(len(PREDICT_COLUMN))]
print("End GaussianNB CV, Time: %s s" % (time.time() - start))

print("Start self NaiveBayes")
start = time.time()
clf = [nb_self_model(train_X, train_Y, i) for i in range(len(PREDICT_COLUMN))]
print("End self NaiveBayes, Time: %s s" % (time.time() - start))

# 3.1.3
print("Start SVC CV")
start = time.time()
svm = [svm_model(train_X, train_Y, i) for i in range(len(PREDICT_COLUMN))]
print("End SVC CV, Time: %s s" % (time.time() - start))

# 3.1.4
print("Start RandomForestClassifier CV")
start = time.time()
rf = [rf_model(train_X, train_Y, i) for i in range(len(PREDICT_COLUMN))]
print("End RandomForestClassifier CV, Time: %s s" % (time.time() - start))

# 3.2
df_test = pd.read_csv('data/testing.csv')
test_X = df_test[TRAIN_FEATURE].values
test_Y = yToPredictColumn(df_test,np.ravel(df_test[['finishing_position']].values))

print("Start LogisticRegression predict")
start = time.time()
predict_lr = np.array([lr[i].predict(test_X) for i in range(len(PREDICT_COLUMN))]).transpose()
df_lr = resultToDf(df_test, predict_lr)
df_lr.to_csv('predictions/lr_predictions.csv', index=False)
print("End LogisticRegression predict, Time: %s s" % (time.time() - start))

print("Start GaussianNB predict")
start = time.time()
predict_nb = np.array([nb[i].predict(test_X) for i in range(len(PREDICT_COLUMN))]).transpose()
df_nb = resultToDf(df_test, predict_nb)
df_nb.to_csv('predictions/nb_predictions.csv', index=False)
print("End GaussianNB predict, Time: %s s" % (time.time() - start))

print("Start self NaiveBayes predict")
start = time.time()
predict_clf = np.array([clf[i].predict(test_X) for i in range(len(PREDICT_COLUMN))]).transpose()
df_clf = resultToDf(df_test, predict_clf)
print("End self NaiveBayes predict, Time: %s s" % (time.time() - start))

print("Start SVC predict")
start = time.time()
predict_svm = np.array([svm[i].predict(test_X) for i in range(len(PREDICT_COLUMN))]).transpose()
df_svm = resultToDf(df_test, predict_svm)
df_svm.to_csv('predictions/svm_predictions.csv', index=False)
print("End SVC predict, Time: %s s" % (time.time() - start))

print("Start RandomForestClassifier predict")
start = time.time()
predict_rf = np.array([rf[i].predict(test_X) for i in range(len(PREDICT_COLUMN))]).transpose()
df_rf = resultToDf(df_test, predict_rf)
df_rf.to_csv('predictions/rf_predictions.csv', index=False)
print("End RandomForestClassifier predict, Time: %s s" % (time.time() - start))

# 3.3
eval_lr = predictEval(test_Y, predict_lr)
print("LogisticRegression\n", eval_lr)

eval_nb = predictEval(test_Y, predict_nb)
print("GaussianNB\n", eval_nb)

eval_clf = predictEval(test_Y, predict_clf)
print("self NaiveBayes\n", eval_clf)

eval_svm = predictEval(test_Y, predict_svm)
print("SVC\n", eval_svm)

eval_rf = predictEval(test_Y, predict_rf)
print("RandomForestClassifier\n", eval_rf)