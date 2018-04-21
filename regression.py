import pandas as pd
import numpy as np
import math

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def predeictToEval(df, y):
	rmse = math.sqrt(mean_squared_error(df['finish_time_ms'].values , y))

	y_rank = [0 for i in y]
	df = df.reset_index(drop=True)
	raceIndex = df.groupby('race_id').indices
	for raceId in raceIndex:
		sortIndex = np.argsort(y[raceIndex[raceId]])
		for i in range(len(sortIndex)):
			y_rank[raceIndex[raceId][sortIndex[i]]] = i + 1

	top1, top3, avgRank = [], [], []
	for i in range(len(y_rank)):
		if y_rank[i] == 1:
			pos = df.iloc[i].finishing_position
			top1.append(1 if pos==1 else 0)
			top3.append(1 if pos<=3 else 0)
			avgRank.append(pos)

	df_out = df[['race_id','horse_id']]
	df_out.columns = ['RaceID', 'HorseID']
	df_result = pd.DataFrame(y_rank, columns=['HorseRank'])
	df_out = pd.concat([df_out, df_result], axis=1)

	return [(rmse, np.mean(top1), np.mean(top3), np.mean(avgRank)), df_out]

df_train = pd.read_csv('data/training.csv')
df_train['finish_time_ms'] = df_train[['finish_time']].apply(lambda x : np.multiply(np.array(x.finish_time.split('.')).astype(np.int), [6000,100,1]).sum(), axis=1)
train_X = df_train[['actual_weight','declared_horse_weight','draw','win_odds','jockey_ave_rank','trainer_ave_rank','recent_ave_rank','race_distance']].values
train_Y = np.ravel(df_train[['finish_time_ms']].values)

# 4.1.1
svr_model = SVR(kernel='rbf', C=5, epsilon=0.5)
svr_model.fit(train_X, train_Y)

# 4.1.2
gbrt_model = GradientBoostingRegressor(loss='quantile', learning_rate=0.03, n_estimators=300, max_depth=3, random_state=3320)
gbrt_model.fit(train_X, train_Y)

# 4.2
df_test = pd.read_csv('data/testing.csv')
df_test['finish_time_ms'] = df_test[['finish_time']].apply(lambda x : np.multiply(np.array(x.finish_time.split('.')).astype(np.int), [6000,100,1]).sum(), axis=1)
test_X = df_test[['actual_weight','declared_horse_weight','draw','win_odds','jockey_ave_rank','trainer_ave_rank','recent_ave_rank','race_distance']].values
test_Y = np.ravel(df_test[['finish_time_ms']].values)

result_svr = predeictToEval(df_test, svr_model.predict(test_X))
print("SVR: ", result_svr[0])
result_svr[1].to_csv('predictions/svr_predictions.csv', index=False)

result_gbrt = predeictToEval(df_test, gbrt_model.predict(test_X))
print("GBRT: ", result_gbrt[0])
result_gbrt[1].to_csv('predictions/gbrt_predictions.csv', index=False)

# Normalization
print("Normalization")
scaler = StandardScaler()
train_X_norm = scaler.fit_transform(train_X, train_Y)
test_X_norm = scaler.transform(test_X)

svr_model = SVR(kernel='rbf', C=5, epsilon=0.5)
svr_model.fit(train_X_norm, train_Y)

gbrt_model = GradientBoostingRegressor(loss='quantile', learning_rate=0.03, n_estimators=300, max_depth=3, random_state=3320)
gbrt_model.fit(train_X_norm, train_Y)

result_svr = predeictToEval(df_test, svr_model.predict(test_X_norm))
print("SVR: ", result_svr[0])
result_svr[1].to_csv('predictions/svr_norm_predictions.csv', index=False)

result_gbrt = predeictToEval(df_test, gbrt_model.predict(test_X_norm))
print("GBRT: ", result_gbrt[0])
result_gbrt[1].to_csv('predictions/gbrt_norm_predictions.csv', index=False)