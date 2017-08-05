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