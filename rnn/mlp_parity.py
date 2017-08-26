'''
Created on Ago 26, 2017

@author: Varela
motivation: Feed forward the XOR parity problem using a RNN on theano
course url: https://www.udemy.com/deep-learning-recurrent-neural-networks-in-python/learn/v4/t/lecture/5359714?start=0
'''

import numpy as np 
import theano 
import theano.tensor as T 
import matplotlib.pyplot as plt 

from util import init_weight, all_parity_pairs
from sklearn.utils import shuffle


class HiddenLayer(object):

	def __init__(self, Mo, Mi, an_id):		
		self.Mo = Mo 
		self.Mi = Mi 
		self.id = an_id

		W = init_weight(Mo, Mi)
		b = np.zeros(Mi)
		self.W  = theano.shared(W, 'W_%s' % an_id)
		self.b  = theano.shared(b, 'b_%s' % an_id)
		self.params = [self.W, self.b]

	
	def forward(self, X):
		return T.nnet.relu(X.dot(self.W) + self.b)

class AnnTheano(object):	
	def __init__(self, hidden_layer_sizes):
		self.hidden_layer_sizes = hidden_layer_sizes 

	def fit(self, X, Y, learning_rate=10e-3, mu=0.99, reg=10e-12, eps=10e-10, epochs=400, batch_sz=20, print_period=1, show_fig=False): 
		X, Y = shuffle(X, Y)			
		X = np.int32(X)
		Y = np.int32(Y)

		N, D = X.shape
		K = len(np.unique(Y))

		self.hidden_layers = [] 
		self.params = [] 
		Mo = D 
		for i, Mi in enumerate(self.hidden_layer_sizes):
			h = HiddenLayer(Mo, Mi, i)
			self.hidden_layers.append(h)
			self.params +=  h.params 
			Mo = Mi 

		W = init_weight(Mo, K)
		b = np.zeros(K)
		self.W  = theano.shared(W, 'W_lgr')
		self.b  = theano.shared(b, 'b_lgr')
		self.params += [self.W, self.b]	


		#Momentum making zeros the same size as p
		dparams = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]

		thX = T.matrix(dtype='int32', name='X')
		thY = T.ivector('Y')
		pY = self.forward(thX)

		rcost = reg*T.sum([(p*p).sum() for p in self.params])
		cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + rcost
		prediction = T.argmax(pY, 1)
		grads = T.grad(cost, self.params)

		updates = [
			(p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
		] + [
			(dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
		]

		train_op = theano.function(
			inputs=[thX, thY],
			outputs=[cost, prediction],
			updates=updates,			
		)

		LL = [] 
		n_batches = int(N / batch_sz)
		for i in xrange(epochs):
			X, Y = shuffle(X, Y)
			for j in xrange(n_batches):
				Xbatch = X[j*batch_sz:(j+1)*batch_sz,:]
				Ybatch = Y[j*batch_sz:(j+1)*batch_sz]

				c, p = train_op(Xbatch, Ybatch)

				if j % print_period == 0:
					LL.append(c)
					e = np.mean(Ybatch != p )

					print "i:%d\tj:%d\tnb:%d\tcost:%.3f\terror:%.3f\t" % (i, j, n_batches, c, e)
		if show_fig:
			plt.plot(LL)			
			plt.show()
	
	def forward(self, X):
		Z = X 
		for h in self.hidden_layers: 
			Z = h.forward(Z)				
		return T.nnet.softmax(Z.dot(self.W) + self.b)


def wide():
	X, Y = all_parity_pairs(12)
	ann = AnnTheano([2048])
	ann.fit(X, Y, learning_rate=10e-5, print_period=10, epochs=300, show_fig=True)

def deep():
	X, Y = all_parity_pairs(12)
	ann = AnnTheano([1024]*2)
	ann.fit(X, Y, learning_rate=10e-4, print_period=10, epochs=100, show_fig=True)


if __name__ == '__main__':
	# wide()
	deep()





