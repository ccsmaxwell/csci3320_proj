import matplotlib.pyplot as plt
import pandas as pd

df_horse = pd.read_csv('data/training.csv')

winByDraw = df_horse[df_horse['finishing_position']==1].groupby('draw').size().reset_index().values

plt.pie(winByDraw[:,1], labels=winByDraw[:,0], autopct='%.2f%%')
plt.title('Draw Bias Effect')
plt.show()