import numpy as np
from scipy.stats import norm

# 3.1.2
class NaiveBayes:	
	def fit(self, X, y):
		self.mean = np.mean(X, axis=0)
		self.std = np.std(X, axis=0)

		allClass = np.unique(y)

		self.stat = {}
		for i in allClass:
			classX = X[y==i]
			classStat = {}
			classStat["prior"] = len(classX) / len(y)
			classStat["mean"] = np.mean(classX, axis=0)
			classStat["std"] = np.std(classX, axis=0)
			self.stat[i] = classStat

		return self

	def predict(self, X):
		result = []
		for r in X:
			evid = norm(self.mean, self.std).pdf(r)
			like = dict([(c, norm(self.stat[c]['mean'], self.stat[c]['std']).pdf(r)) for c in self.stat])

			post = dict([(c, self.stat[c]['prior'] * np.prod(like[c]) / np.prod(evid)) for c in self.stat])
			result.append(max(post, key=post.get))

		return result