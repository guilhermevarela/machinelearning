'''
Created on Aug 24, 2017

@author: Varela

motive: LogisticRegression + softmax
course url: https://www.udemy.com/data-science-deep-learning-in-python/learn/v4/t/lecture/5239240?start=0

'''



import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle 
from utils import sigmoid, error_rate, get_facialexpression, cost, y2indicator, softmax 


class LogisticModelSoftmax(object):
	
	def __init__(self):
		pass 

	def fit(self, X, Y, learning_rate=10e-8, reg=10e-8, epochs=10000, show_figure=False):
		
		X, Y = shuffle(X, Y)
		K = len(set(Y))
		Xvalid, Yvalid = X[-1000:], Y[-1000:]
		Tvalid = y2indicator(Yvalid,K)
		X, Y = X[:-1000], Y[:-1000]

		N, D = X.shape

		T = y2indicator(Y, K)
		self.W = np.random.randn(D, K) / np.sqrt(D)
		self.b = np.zeros(K)

		costs = []
		best_validation_error = 1
		for i in xrange(epochs):
			pY = self.forward(X)
			# gradient descent step
			self.W -= learning_rate *(X.T.dot(pY-T) + reg*self.W)
			self.b -= learning_rate *((pY-T).sum(axis=0) 	+ reg*self.b)

			if i % 10 == 0:
				pYvalid = self.forward(Xvalid)
				
				c = cost(Tvalid, pYvalid) 
				costs.append(c)	
				e = error_rate(Yvalid, np.argmax(pYvalid,axis=1))

				print "i", i, "cost:", c, "error", e
				if e < best_validation_error:
					best_validation_error = e
		print "best_validation_error:", best_validation_error

		if show_figure:
			plt.plot(costs)
			plt.show()
	
	def forward(self, X):			
		return softmax(X.dot(self.W) + self.b)
  
	def predict(self, X):
		pY = self.forward(X)		
		return np.argmax(pY, axis=1)

	def score(self, X, Y):
		prediction = self.predict(X)
		return 1 - error_rate(Y, prediction)

def main():
	X, Y = get_facialexpression(balance_ones=True)
	
	model = LogisticModelSoftmax()
	model.fit(X, Y, show_figure=True)
	print model.score(X, Y)



if __name__ == '__main__':
	main()  	
