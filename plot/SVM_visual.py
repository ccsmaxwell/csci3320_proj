import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.svm import SVC

df_train = pd.read_csv('data/training.csv')
feature = ['recent_ave_rank','jockey_ave_rank']
X = df_train[feature].values

raceSize = df_train.groupby('race_id').finishing_position.max()
# y = [1 if y[i] <= raceSize[df.iloc[i].race_id]/2 else 0 for i in range(raceSize.size)]np.ravel(df_train[['finishing_position']].values)

svm_model = SVC(kernel="linear",random_state=3320)
svm_model.fit(X, y)

X0, X1 = X[:, 0], X[:, 1]
x_min, x_max = X0.min() - 1, X0.max() + 1
y_min, y_max = X1.min() - 1, X1.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm)
plt.show()