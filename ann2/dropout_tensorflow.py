'''
Created on Ago 05, 2017

@author: Varela

Tensorflow dropout regularization scheme


A 1-hidden-layer neural network in Theano.
This code is not optimized for speed.
It's just to get something working, using the principles we know.


For the class Data Science: Practical Deep Learning Concepts in Theano and TensorFlow
course url: https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
lecture url: https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow/learn/v4/t/lecture/5584716?start=0

'''

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 


from util import get_normalized_data 
from sklearn.utils import shuffle

def error_rate(p, t):
	return np.mean(p !=t )

def relu(a):
	return a * ( a > 0 )

class HiddenLayer(object):
	
	def __init__(self, M1, M2):
		self.M1 = M1 
		self.M2 = M2 

		W = np.random.randn(M1,M2) / np.sqrt(M1)
		b = np.zeros(M2)

		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))
		self.params = [self.W, self.b]

	def forward(self, X):
		return tf.nn.relu(tf.matmul(X, self.W) + self.b)

class ANN(object):

	def __init__(self, hidden_layer_sizes, p_keep):
		self.hidden_layer_sizes = hidden_layer_sizes 
		self.dropout_rates = p_keep		

	def fit(self, X, Y, learning_rate=1e-4, mu=0.99, decay=0.99, epochs=300, batch_size=100, split=True, show_fig=False):
		#Make validation set
		X, Y = shuffle(X, Y)
		X = X.astype(np.float32)	
		Y = Y.astype(np.int32)
		if split:
			Xvalid, Yvalid = X[-1000:], Y[-1000:]
			X, Y = X[:-1000], Y[:-1000]
		else:
			Xvalid, Yvalid = X, Y 



		#initialize hidden layers
		N, D  = X.shape
		K = len(set(Y))
		self.hidden_layers=[]
		M1 = D 
		count=0
		for M2 in self.hidden_layer_sizes:
			h = HiddenLayer(M1, M2)
			self.hidden_layers.append(h)
			M1=M2
			count+=1
		
		W = np.random.randn(M1,K) / np.sqrt(M1)
		b = np.zeros(K)

		#In Tf variables are trainable params - 
		# we expect then to change
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))

		#collect parameters for latter use
		self.params = [self.W, self.b]
		for h in self.hidden_layers:
			self.params += h.params

		#In Tf placeholders are used to feed actual examples - 
		#we don't expect then to change
		inputs = tf.placeholder(tf.float32, shape=(None,D), name='inputs')
		labels = tf.placeholder(tf.int32, shape=(None,), name='labels')
		logits = self.forward(inputs)

		#this is cost for training
		cost = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=logits,
				labels=labels
			)
		)

		train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu).minimize(cost)
		prediction = self.predict(inputs)


		n_batches= N/batch_size
		costs = [] 
		init = tf.global_variables_initializer()
		with tf.Session() as session: 
			session.run(init)
			for i in xrange(epochs):
				print "epoch:%d\tn_batches:%d\t" % (i, n_batches)
				X, Y = shuffle(X, Y)

				for j in xrange(n_batches):
			
					Xbatch = X[j*batch_size:((j+1)*batch_size)]
					Ybatch = Y[j*batch_size:((j+1)*batch_size)]
					
					session.run(train_op, feed_dict={
						inputs: Xbatch,
						labels: Ybatch
					})
				
					if j % 20 == 0:
						c = session.run(cost, feed_dict={inputs: Xvalid, labels: Yvalid})
						p = session.run(prediction, feed_dict={inputs: Xvalid})
						costs.append(c)

						err = error_rate(Yvalid, p)
						print "i:%d\tj:%d\tnb:%d\tcost:%.3f\terror_rate:%.3f" % (i, j, n_batches, c, err)
				
		if show_fig:			
			plt.plot(costs)
			plt.show()

	def forward(self, X):
		# no need to define different functions for train and predict
		# tf.nn.dropout takes care of the differences for us
		Z = X 
		Z = tf.nn.dropout(Z, self.dropout_rates[0])
		for h, p in zip(self.hidden_layers, self.dropout_rates[1:]):			
			Z = h.forward(Z)
			Z = tf.nn.dropout(Z, p)

		return tf.matmul(Z, self.W) + self.b 

	def predict(self, X):
		pY = self.forward(X)		
		return tf.argmax(pY, axis=1)

def main():
	#step 1: get the data and define all usual variables
	X, Y  = get_normalized_data()

	ann = ANN([500, 300], [0.8, 0.5, 0.5])	
	ann.fit(X, Y, show_fig=True)

if __name__ == '__main__':
	main()	



