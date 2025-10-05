import numpy as np

class perceptron():
	def __init__(self, n_iters=1000, theta: int = 1, l_rate: float = 1):		
		self.n_iters = n_iters
		self.omega = None
		self.theta = theta
		self.l_rate = l_rate

	def fit(self, X, y):
		error = 1
		epoch = 0
		n_len, n_features = X.shape

		if self.omega is None:
			self.omega = np.zeros(n_features)
		if self.theta == None:
			self.theta = 0
		if self.l_rate == None:
			self.l_rate = 1
		
		while error > 0:
			if epoch > self.n_iters:
				break
			epoch += 1
			error = 0
			# print("epoch:", epoch)
			for i in range(n_len):
				# print("input:", X[i], y[i])
				if np.dot(self.omega, X[i]) >= self.theta:
					pred = 1
				else:
					pred = 0
				if pred != y[i]:
					self.theta -= self.l_rate * (y[i] - pred)
					self.omega += self.l_rate * (y[i] - pred) * X[i]
					error += np.sum(np.abs(y[i] - pred))
				# print("omega:", self.omega, " theta:", self.theta, " error:", error)
		print("Result:\n", "omega:", self.omega, "Theta:", self.theta)
	

	def predict(self, X):
		results = list()
		n_len, n_features = X.shape
		for i in range(n_len):
			if np.dot(self.omega, X[i]) >= self.theta:
				results.append(1)
			else:
				results.append(0)
		return results

	def output(self, x):
		return np.where(x>=0, 1, 0)
	