import pandas as pd
import numpy as np

df_test = pd.read_csv('data/testing.csv')
allRace = df_test.race_id.unique()

def clsBasicBat(df_cls):
	resultAmt = 0
	for race_id in allRace:
		# find top horses
		df_race = df_cls[df_cls.RaceID == race_id]
		df_top = df_race[df_race.HorseWin == 1]
		if df_top.shape[0] == 0:
			df_top = df_race[df_race.HorseRankTop3 == 1]
			if df_top.shape[0] == 0:
				df_top = df_race[df_race.HorseRankTop50Percent == 1]
		# select unique win horse
		winID = ''
		if df_top.shape[0] > 1:
			df_test_top = df_test[(df_test.horse_id.isin(df_top.HorseID.values)) & (df_test.race_id == race_id)]
			winID = df_test_top.loc[df_test_top.declared_horse_weight.idxmax(), 'horse_id']
		elif df_top.shape[0] == 0:
			df_test_top = df_test[(df_test.race_id == race_id)]
			winID = df_test_top.loc[df_test_top.declared_horse_weight.idxmax(), 'horse_id']
		else:
			winID = df_top.iloc[0].HorseID
		# check if the horse win
		winIDRec = df_test[(df_test.race_id == race_id) & (df_test.horse_id == winID)]
		resultAmt = resultAmt - 1
		if winIDRec.finishing_position.values[0] == 1:
			resultAmt = resultAmt + winIDRec.win_odds.values[0]
	# return amount win/lose
	return resultAmt

def regBasicBat(df_reg):
	resultAmt = 0
	for race_id in allRace:
		winID = df_reg[(df_reg.RaceID == race_id) & (df_reg.HorseWin == 1)].HorseID.values[0]
		# check if the horse win
		winIDRec = df_test[(df_test.race_id == race_id) & (df_test.horse_id == winID)]
		resultAmt = resultAmt - 1
		if winIDRec.finishing_position.values[0] == 1:
			resultAmt = resultAmt + winIDRec.win_odds.values[0]
	# return amount win/lose
	return resultAmt

df_lr = pd.read_csv('predictions/lr_predictions.csv')
print("LogisticRegression: ", clsBasicBat(df_lr))

df_nb = pd.read_csv('predictions/nb_predictions.csv')
print("NaiveBayes: ", clsBasicBat(df_nb))

df_svm = pd.read_csv('predictions/svm_predictions.csv')
print("SVM: ", clsBasicBat(df_svm))

df_rf = pd.read_csv('predictions/rf_predictions.csv')
print("RandomForest: ", clsBasicBat(df_rf))

df_svr = pd.read_csv('predictions/svr_predictions.csv')
print("SVR: ", regBasicBat(df_svr))

df_svr_norm = pd.read_csv('predictions/svr_norm_predictions.csv')
print("SVR with normalize: ", regBasicBat(df_svr_norm))

df_gbrt = pd.read_csv('predictions/gbrt_predictions.csv')
print("GBRT: ", regBasicBat(df_gbrt))

df_gbrt_norm = pd.read_csv('predictions/gbrt_norm_predictions.csv')
print("GBRT with normalize: ", regBasicBat(df_gbrt_norm))