'''
Created on Ago 05, 2017

@author: Varela

Theano dropout regularization scheme


A 1-hidden-layer neural network in Theano.
This code is not optimized for speed.
It's just to get something working, using the principles we know.


For the class Data Science: Practical Deep Learning Concepts in Theano and TensorFlow
course url: https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
lecture url: https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow/learn/v4/t/lecture/5584716?start=0

'''

import numpy as np 
import theano  as th 
import theano.tensor as T 
import matplotlib.pyplot as plt 

from theano.tensor.shared_randomstreams import RandomStreams 
from util import get_normalized_data 
from sklearn.utils import shuffle

def error_rate(p, t):
	return np.mean(p !=t )

def relu(a):
	return a * ( a > 0 )

class HiddenLayer(object):
	
	def __init__(self, M1, M2, an_id):
		self.id = an_id
		self.M1 = M1 
		self.M2 = M2 

		W = np.random.randn(M1,M2) / np.sqrt(M1)
		b = np.zeros(M2)

		self.W = th.shared(W, 'W_%s' % self.id )
		self.b = th.shared(b, 'b_%s' % self.id )
		self.params = [self.W, self.b]

	def forward(self, X):
		return T.nnet.relu(X.dot(self.W) + self.b)

class ANN(object):

	def __init__(self, hidden_layer_sizes, p_keep):
		self.hidden_layer_sizes = hidden_layer_sizes 
		self.dropout_rates = p_keep		

	def fit(self, X, Y, learning_rate=1e-4, mu=0.9, decay=0.9, epochs=8, batch_size=100, show_fig=False):
		#Make validation set
		X, Y = shuffle(X, Y)
		X = X.astype(np.float32)	
		Y = Y.astype(np.int32)
		Xvalid, Yvalid = X[-1000:], Y[-1000:]
		X, Y = X[:-1000], Y[:-1000]

		self.rng = RandomStreams() 

		#initialize hidden layers
		N, D  = X.shape
		K = len(set(Y))
		self.hidden_layers=[]
		M1 = D 
		count=0
		for M2 in self.hidden_layer_sizes:
			h = HiddenLayer(M1, M2, count)
			self.hidden_layers.append(h)
			M1=M2
			count+=1
		
		W = np.random.randn(M1,K) / np.sqrt(M1)
		b = np.zeros(K)

		self.W = th.shared(W, 'W_logreg')
		self.b = th.shared(b, 'b_logreg')

		#collect parameters for latter use
		self.params = [self.W, self.b]
		for h in self.hidden_layers:
			self.params += h.params

		#setup theano functions and variables
		thX = T.matrix('X')
		thY = T.ivector('Y')
		pY_train = self.forward_train(thX)

		#this is cost for training
		cost = -T.mean(T.log(pY_train[T.arange(thY.shape[0]), thY]))

		#gradients wrt each param
		grads= T.grad(cost, self.params)

		#for momentum 
		dparams=[th.shared(np.zeros_like(p.get_value())) for p in self.params]

		#for rmsprop
		cache = [th.shared(np.ones_like(p.get_value())) for p in self.params]

		new_cache = [decay*c + (1-decay)*g*g for p, c, g in zip(self.params, cache, grads)]
		new_dparams = [mu*dp - learning_rate*(g/T.sqrt(new_c + 1e-10)) for p, new_c, dp, g in zip(self.params, new_cache, dparams, grads)]
		updates = [
			(c, new_c) for c, new_c in zip(cache, new_cache)
		] + [
			(dp, new_dp) for dp, new_dp in zip(dparams, new_dparams)
		] + [
			(p, p + new_dp) for p, new_dp in zip(self.params, new_dparams)
		]

		train_op = th.function(
			inputs=[thX, thY],
			updates=updates
		)

		#for evaluation and prediction
		pY_predict = self.forward_predict(thX)
		cost_predict = -T.mean(T.log(pY_predict[T.arange(thY.shape[0]),thY]))
		prediction = self.predict(thX)
		cost_predict_op = th.function(inputs=[thX, thY], outputs=[cost_predict, prediction])

		n_batches= N/batch_size
		costs = [] 
		for i in xrange(epochs):
			for j in xrange(n_batches):
			
				Xbatch = X[j*batch_size:((j+1)*batch_size)]
				Ybatch = Y[j*batch_size:((j+1)*batch_size)]
				train_op(Xbatch, Ybatch)
				if j % 20 == 0:
					
					c, p = cost_predict_op(Xvalid, Yvalid)
					costs.append(c)				
					err = error_rate(Yvalid, p)
					print "i:%d\tj:%d:\tnb:%d\tcost:%.3f\terror_rate:%.3f" % (i, j, n_batches, c, err)
				
		if show_fig:			
			plt.plot(costs)
			plt.show()

	def forward_train(self, X):
		Z = X 
		for h, p in zip(self.hidden_layers, self.dropout_rates[:-1]):
			mask = self.rng.binomial(n=1, p=p, size=Z.shape)
			Z = mask * Z 
			Z = h.forward(Z)

		mask = self.rng.binomial(n=1, p=self.dropout_rates[-1], size=Z.shape)
		Z = mask * Z 
		return T.nnet.softmax(Z.dot(self.W) + self.b)

	def forward_predict(self, X):
		Z = X
		for h, p in zip(self.hidden_layers, self.dropout_rates[:-1]):
			Z = h.forward(p * Z)	
		return T.nnet.softmax((self.dropout_rates[-1]*Z).dot(self.W) + self.b)

	def predict(self, X):
		pY = self.forward_predict(X)		
		return T.argmax(pY, axis=1)

def main():
	#step 1: get the data and define all usual variables
	X, Y  = get_normalized_data()

	ann = ANN([500, 300], [0.8, 0.5, 0.5])	
	ann.fit(X, Y, show_fig=True)

if __name__ == '__main__':
	main()	


