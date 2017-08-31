'''
Created on Aug 31, 2017

@author: Varela

motivation: Vanishing gradine problem
	
'''

import numpy as np 
import theano 
import theano.tensor as T 
import matplotlib.pyplot as plt

from sklearn.utils import shuffle 
from util import get_mnist, error_rate, relu , init_weight

class HiddenLayer(object): 
	def __init__(self, Mi, Mo): 
		W = init_weight(Mi, Mo)
		b = np.zeros(Mo)
		self.W = theano.shared(W)
		self.b = theano.shared(b)
		self.params= [self.W, self.b]

	def forward(self, X):
		return T.nnet.sigmoid(X.dot(self.W) + self.b)


class ANN(object): 
	def __init__(self, hidden_layer_sizes): 
		self.hidden_layer_sizes = hidden_layer_sizes

	def fit(self, X, Y, learning_rate=.01, mu=.99, epochs=30, batch_sz=100): 
		N, D  = X.shape
		K = len(set(Y))

		self.hidden_layers = [] 
		Mi=D 
		for Mo in self.hidden_layer_sizes: 
			h = HiddenLayer(Mi, Mo)
			self.hidden_layers.append(h)
			Mi=Mo 

		W = init_weight(Mi, K) 
		b = np.zeros(K)
		self.W = theano.shared(W)
		self.b = theano.shared(b)

		self.params = [self.W, self.b]
		self.allWs  = [] 
		for h in self.hidden_layers:
			self.params += h.params 
			self.allWs.append(h.W)
		self.allWs.append(self.W)

		X_in = T.matrix('X_in')
		targets = T.ivector('Targets') 
		pY = self.forward(X_in)

		cost= -T.mean( T.log( pY[T.arange(pY.shape[0]), targets ]) )
		prediction = self.predict(X_in)

		dparams= [theano.shared(p.get_value()*0) for p in self.params] 
		grads =T.grad(cost, self.params)

		updates=[
			(p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
		]		+  [
			(dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)		
		]
		# N, D = X.shape
		# K = len(set(Y))

		# self.hidden_layers = []
		# mi = D
		# for mo in self.hidden_layer_sizes:
		#     h = HiddenLayer(mi, mo)
		#     self.hidden_layers.append(h)
		#     mi = mo

		# # initialize logistic regression layer
		# W = init_weight(*(mo, K))
		# b = np.zeros(K)
		# self.W = theano.shared(W)
		# self.b = theano.shared(b)

		# self.params = [self.W, self.b]
		# self.allWs = []
		# for h in self.hidden_layers:
		#     self.params += h.params
		#     self.allWs.append(h.W)
		# self.allWs.append(self.W)

		# X_in = T.matrix('X_in')
		# targets = T.ivector('Targets')
		# pY = self.forward(X_in)

		# cost = -T.mean( T.log(pY[T.arange(pY.shape[0]), targets]) )
		# prediction = self.predict(X_in)

		# dparams = [theano.shared(p.get_value()*0) for p in self.params]
		# grads = T.grad(cost, self.params)

		# updates = [
		#     (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
		# ] + [
		#     (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
		# ]

		train_op =theano.function(
			inputs=[X_in, targets],
			outputs=[cost, prediction],
			updates=updates,
		)

		n_batches = N / batch_sz 
		costs= [] 
		lastWs = [W.get_value() for W in self.allWs]
		
		W_changes=[] 
		for i in xrange(epochs): 
			print "epoch", i 
			X, Y = shuffle(X, Y)
			for j in xrange(n_batches):
				Xbatch = X[j*batch_sz:(j+1)*batch_sz,:]
				Ybatch = Y[j*batch_sz:(j+1)*batch_sz]
				
				c, Yhat = train_op(Xbatch, Ybatch)				
				
				if j % 100 == 0: 										
					error = error_rate(Ybatch, Yhat) 
					print "i:%d\tj:%d\tnb:%d\tcost:%.6f\terror:%.3f\t" % (i,j,n_batches, c, error)
				costs.append(c)	
				W_change = [np.abs(W.get_value() - lastW).mean() for W, lastW in zip(self.allWs, lastWs)]
				W_changes.append(W_change)
				lastWs = [W.get_value() for W in self.allWs]

		W_changes = np.array(W_changes) 
		plt.subplot(2,1,1)
		for i in xrange(W_changes.shape[1]):  
			plt.plot(W_changes[:,1], label='layer %d' % i)

		plt.legend() 	

		plt.subplot(2,1,2)
		plt.plot(costs)
		plt.show() 

	def predict(self, X):
		return T.argmax(self.forward(X), axis=1)
	
	def forward(self, X):
		Z=X 
		for h in self.hidden_layers: 
		  Z = h.forward(Z) 
		Y = T.nnet.softmax(Z.dot(self.W) + self.b)  
		return Y  

def main(): 
	Xtrain, Ytrain, Xtest, Ytest =  get_mnist() 
	dnn = ANN([1000, 750, 500])
	dnn.fit(Xtrain, Ytrain) 

if __name__ == '__main__': 
	theano.config.floatX='float32'
	main()
