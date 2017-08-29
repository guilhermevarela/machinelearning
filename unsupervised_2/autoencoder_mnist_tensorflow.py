'''
Created on Aug 28, 2017

@author: Varela

motivation: Unsupervised techniques auto encoder
	
'''

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt

from sklearn.utils import shuffle 
from util import relu, get_mnist, error_rate, init_weight

class AutoEncoder(object):
	def __init__(self, D, M, an_id): 
		self.M = M 
		self.id = an_id 
		self.build(D, M) 

	def set_session(self, session): 
		self.session = session 

	def build(self, D, M):
		self.W = tf.Variable(tf.random_normal(shape=(D,M)))
		self.bh = tf.Variable(np.zeros(M).astype(np.float32))
		self.bo = tf.Variable(np.zeros(D).astype(np.float32))

		self.X_in = tf.placeholder(tf.float32, shape=(None, D)) 
		self.Z = self.forward_hidden(self.X_in) 
		self.X_hat = tf.forward_output(self.X_in)

		#using the naive formulation for cross-entropy
		#will have numerical stability issues if X_hat = 0 or 1 
		logits= self.forward_logits(self.X_in)
		self.cost = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				labels=self.X_in,
				logits=logits,
			) 
		)

		self.train_op = tf.train.AdamOptimizer(1e-1).minimize(self.cost)

	def fit(self, X, epochs=1, batch_sz=100, show_fig=False):
		N, D = X.shape 
		n_batches = N // batch_sz

		costs = [] 
		print 'trainning autoencoder: %s' % self.ud 
		for i in xrange(epochs):
			print "epoch:", i 
			X = shuffle(X)
			for j in xrange(n_batches):
				batch = X[j*batch_sz:(j+1)*batch_sz]
				_, c = self.session.run(
					(self.train_op, self.cost),
					feed_dict={self.X_in: batch}
				)
				if j % 10 ==0: 
					print "i:%d\tj:%d\tnb:%d\tcost:%.6f\t" % (i,j,n_batches,c)
				costs.append(c)	

		if show_fig: 
			plt.plot(costs)
			plt.show()

		def transform(self, X):
			# accepts and returns a real numpy array
			# unlike forward_hidden and forward_output
			# which deal with tensorflow variables
			return self.session.run(self.Z, feed_dict={self.X_in: X})

		def predict(self, X):
		  # accepts and returns a real numpy array
      # unlike forward_hidden and forward_output
			# which deal with tensorflow variables
			return self.session.run(self.X_hat, feed_dict={self.X_in: X})

		def forward_hidden(self, X)
			Z = tf.nn.sigmoid(tf.matmul(X, self.W) + self.bh)
			return Z 

		def forward_logits(self, X): 
			Z= self.forward_hidden(X)
			return tf.matmul(Z, tf.transpose(self.W)) + self.bo

		def forward_output(self,X):
			return tf.nn.sigmoid(self.forward_logits(X))