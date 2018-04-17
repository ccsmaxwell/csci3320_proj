import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv('data/training.csv')
feature = ['actual_weight','declared_horse_weight','draw','win_odds','recent_ave_rank','jockey_ave_rank','trainer_ave_rank','race_distance']
train_X = df_train[feature].values
train_Y = np.ravel(df_train[['finishing_position']].values)

rf_model = RandomForestClassifier(random_state=3320)
rf_model.fit(train_X, train_Y)

plt.bar(feature, rf_model.feature_importances_)
plt.xticks(rotation='vertical')
plt.title('Feature Importances')
plt.xlabel('feature')
plt.ylabel('feature importances')
plt.tight_layout()
plt.show()