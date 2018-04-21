import matplotlib.pyplot as plt
import pandas as pd

df_horse = pd.read_csv('data/training.csv')

raceCountByHorse = df_horse.groupby('horse_name').finishing_position.count().to_frame()
raceWinByHorse = df_horse[df_horse['finishing_position']==1].groupby('horse_name').finishing_position.count().to_frame()
racebyHorse = raceCountByHorse.join(raceWinByHorse, lsuffix='_count', rsuffix='_win').fillna(0)
racebyHorse['winRate'] = racebyHorse['finishing_position_win'] / racebyHorse['finishing_position_count']
horseArr = racebyHorse.reset_index().values

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))
ax1.scatter(horseArr[:,3], horseArr[:,2])
for i, e in enumerate(horseArr[:,0]):
	if horseArr[i][3] >= 0.5 and horseArr[i][2] >=4:
		ax1.annotate(e, (horseArr[i][3],horseArr[i][2]), rotation=15, ha='left', va='bottom')
ax1.set_title('Horse - Win Rate vs. Number of Wins')
ax1.set_xlabel('win rate')
ax1.set_ylabel('number of wins')


raceCountByJockey = df_horse.groupby('jockey').finishing_position.count().to_frame()
raceWinByJockey = df_horse[df_horse['finishing_position']==1].groupby('jockey').finishing_position.count().to_frame()
racebyJockey = raceCountByJockey.join(raceWinByJockey, lsuffix='_count', rsuffix='_win').fillna(0)
racebyJockey['winRate'] = racebyJockey['finishing_position_win'] / racebyJockey['finishing_position_count']
jockeyArr = racebyJockey.reset_index().values

ax2.scatter(jockeyArr[:,3], jockeyArr[:,2])
for i, e in enumerate(jockeyArr[:,0]):
	if jockeyArr[i][3] >= 0.15 and jockeyArr[i][2] >=100:
		ax2.annotate(e, (jockeyArr[i][3],jockeyArr[i][2]))
ax2.set_title('Jockey - Win Rate vs. Number of Wins')
ax2.set_xlabel('win rate')
ax2.set_ylabel('number of wins')

plt.show()