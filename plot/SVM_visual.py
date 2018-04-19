import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.svm import SVC

df_train = pd.read_csv('data/training.csv')
feature = ['recent_ave_rank','jockey_ave_rank']
X = df_train[feature].values

raceSize = df_train.groupby('race_id').finishing_position.max()
y = np.ravel(df_train[['finishing_position']].values)
y = [1 if y[i] <= raceSize[df_train.iloc[i].race_id]/2 else 0 for i in range(len(y))]

svm_model = SVC(kernel="linear",random_state=3320)
svm_model.fit(X, y)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = np.meshgrid(np.arange(X0.min() - 1, X0.max() + 1, 0.05),np.arange(X1.min() - 1, X1.max() + 1, 0.05))
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm)
plt.title("SVM: recent_rank vs. jockey_ave_rank")
plt.xlabel('recent_rank')
plt.ylabel('jockey_ave_rank')
plt.show()