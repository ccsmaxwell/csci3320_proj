import pandas as pd
import numpy as np

# 2.2.1
df_horse = pd.read_csv('data/race-result-horse.csv')
df_horse = df_horse[df_horse.finishing_position.isin(list(map(lambda x : str(x), list(range(1,15))))) == True]

# 2.2.2
df_horse = df_horse.sort_values(by=['race_id']).reset_index(drop=True)
df_horse = df_horse.assign(recent_6_runs = '')
df_horse = df_horse.assign(recent_ave_rank = 7.0)
allHorse = df_horse.horse_id.unique()
for i in allHorse:
	rank_rec = []
	df_i = df_horse[df_horse['horse_id']==i]

	for index, e in df_i.iterrows():
		df_horse.at[index,'recent_6_runs'] = '/'.join(rank_rec[-6:])
		if rank_rec:
			df_horse.at[index,'recent_ave_rank'] = sum(map(lambda x : int(x), rank_rec[-6:])) / float(6 if (len(rank_rec) >= 6) else len(rank_rec))
		rank_rec.append(str(df_i.at[index, 'finishing_position']))

# 2.2.3
df_horse['horse_id'] = df_horse['horse_id'].replace(allHorse ,list(range(len(allHorse))))
allJockey = df_horse.jockey.unique()
df_horse['jockey'] = df_horse['jockey'].replace(allJockey ,list(range(len(allJockey))))
allTrainer = df_horse.trainer.unique()
df_horse['trainer'] = df_horse['trainer'].replace(allTrainer ,list(range(len(allTrainer))))

df_horse = df_horse.assign(jockey_ave_rank = 7.0)
for i in range(len(allJockey)):
	mean = pd.to_numeric(df_horse[(df_horse.jockey==i) & (df_horse.race_id<="2016-327")].finishing_position).mean()
	mean = 7.0 if mean is np.nan else mean
	df_horse.loc[df_horse.jockey==i, 'jockey_ave_rank'] = mean

df_horse = df_horse.assign(trainer_ave_rank = 7.0)
for i in range(len(allTrainer)):
	mean = pd.to_numeric(df_horse[(df_horse.trainer==i) & (df_horse.race_id<="2016-327")].finishing_position).mean()
	mean = 7.0 if mean is np.nan else mean
	df_horse.loc[df_horse.trainer==i, 'trainer_ave_rank'] = mean

# 2.2.4
df_race = pd.read_csv('data/race-result-race.csv')
dist = {}
for i,e in df_race.iterrows():
	dist[e.race_id] = e.race_distance
df_horse = df_horse.assign(race_distance = 0)
for i in dist:
	df_horse.loc[df_horse.race_id==i, 'race_distance'] = dist[i]

# 2.2.5
pos = df_horse.loc[df_horse.race_id=="2016-327"].reset_index()['index'].max()
train = df_horse[:pos+1]
test = df_horse[pos+1:]
train.to_csv('data/training.csv')
test.to_csv('data/testing.csv')