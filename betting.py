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
		df_top = df_reg[(df_reg.RaceID == race_id) & (df_reg.HorseRank.isin(list(range(1,3))))]
		df_test_top = df_test[(df_test.horse_id.isin(df_top.HorseID.values)) & (df_test.race_id == race_id)]
		winID = df_test_top.loc[df_test_top.declared_horse_weight.idxmax(), 'horse_id']
		# check if the horse win
		winIDRec = df_test[(df_test.race_id == race_id) & (df_test.horse_id == winID)]
		resultAmt = resultAmt - 1
		if winIDRec.finishing_position.values[0] == 1:
			resultAmt = resultAmt + winIDRec.win_odds.values[0]
	# return amount win/lose
	return resultAmt

df_lr = pd.read_csv('predictions/lr_predictions.csv')
df_nb = pd.read_csv('predictions/nb_predictions.csv')
df_svm = pd.read_csv('predictions/svm_predictions.csv')
df_rf = pd.read_csv('predictions/rf_predictions.csv')
df_svr = pd.read_csv('predictions/svr_predictions.csv')
df_svr_norm = pd.read_csv('predictions/svr_norm_predictions.csv')
df_gbrt = pd.read_csv('predictions/gbrt_predictions.csv')
df_gbrt_norm = pd.read_csv('predictions/gbrt_norm_predictions.csv')

print("Basic betting strategy:")
print("LogisticRegression: ", clsBasicBat(df_lr))
print("NaiveBayes: ", clsBasicBat(df_nb))
print("SVM: ", clsBasicBat(df_svm))
print("RandomForest: ", clsBasicBat(df_rf))
print("SVR: ", regBasicBat(df_svr))
print("SVR with normalize: ", regBasicBat(df_svr_norm))
print("GBRT: ", regBasicBat(df_gbrt))
print("GBRT with normalize: ", regBasicBat(df_gbrt_norm))

print("\nOur own betting strategy: ", end='')
df_lr = df_lr.set_index(['RaceID','HorseID']).rename(columns={'HorseWin':'lr_1', 'HorseRankTop3':'lr_3', 'HorseRankTop50Percent':'lr_50'})
df_nb = df_nb.set_index(['RaceID','HorseID']).rename(columns={'HorseWin':'nb_1', 'HorseRankTop3':'nb_3', 'HorseRankTop50Percent':'nb_50'})
df_svm = df_svm.set_index(['RaceID','HorseID']).rename(columns={'HorseWin':'svm_1', 'HorseRankTop3':'svm_3', 'HorseRankTop50Percent':'svm_50'})
df_rf = df_rf.set_index(['RaceID','HorseID']).rename(columns={'HorseWin':'rf_1', 'HorseRankTop3':'rf_3', 'HorseRankTop50Percent':'rf_50'})
df_svr = df_svr.set_index(['RaceID','HorseID']).rename(columns={'HorseRank':'svr_r'})
df_svr_norm = df_svr_norm.set_index(['RaceID','HorseID']).rename(columns={'HorseRank':'svr_norm_r'})
df_gbrt = df_gbrt.set_index(['RaceID','HorseID']).rename(columns={'HorseRank':'gbrt_r'})
df_gbrt_norm = df_gbrt_norm.set_index(['RaceID','HorseID']).rename(columns={'HorseRank':'gbrt_norm_r'})

df_mix = pd.concat([df_lr,df_nb,df_svm,df_rf, df_svr, df_svr_norm, df_gbrt, df_gbrt_norm], axis=1)

resultAmt = 0
for race_id in allRace:
	# find top horses
	df_race = df_mix.loc[race_id]
	df_top = df_race[(df_race.lr_1 == 1) | (df_race.gbrt_r.isin(range(1,4)))]
	winID = ''
	if df_top.shape[0] > 1:
		df_test_top = df_test[(df_test.horse_id.isin(df_top.index.values)) & (df_test.race_id == race_id)]
		winID = df_test_top.loc[df_test_top.declared_horse_weight.idxmax(), 'horse_id']
	elif df_top.shape[0] == 0:
		df_test_top = df_test[(df_test.race_id == race_id)]
		winID = df_test_top.loc[df_test_top.declared_horse_weight.idxmax(), 'horse_id']
	else:
		winID = df_top.iloc[0].name
	# check if the horse win
	winIDRec = df_test[(df_test.race_id == race_id) & (df_test.horse_id == winID)]
	if winIDRec.win_odds.values[0] < 30:
		resultAmt = resultAmt - 1
		if winIDRec.finishing_position.values[0] == 1:
			resultAmt = resultAmt + winIDRec.win_odds.values[0]
print(resultAmt)