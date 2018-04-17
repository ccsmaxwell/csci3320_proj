import matplotlib.pyplot as plt
import pandas as pd

df_horse = pd.read_csv('data/training.csv')

horseID = input('horse ID: ')
race = df_horse[df_horse['horse_id']==horseID].tail(6).race_id.values
pos = df_horse[df_horse['horse_id']==horseID].tail(6).finishing_position.values.astype(int)

plt.plot(race, pos)
plt.gca().set_ylim([0,15])
plt.title(horseID)
plt.xlabel('race_id')
plt.ylabel('finishing position')
plt.show()