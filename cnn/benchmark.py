'''
Created on Ago 08, 2017

@author: Varela



For the class Data Science: Deep Learning convolutional neural networkds on theano an tensorflow

course url 1: https://deeplearningcourses.com/c/deep-learning-convolutional-neural-networks-theano-tensorflow
course url 2: https://udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow
lecture url: https://www.udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow/learn/v4/t/lecture/4847744?start=0

'''

import os

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 


from scipy.io import loadmat 
from sklearn.utils import shuffle 


from util import y2indicator
from datetime import datetime

#from sklearn.linear_model import LogisticRegression 

def get_svhnpath():
	cwdpath =  os.path.dirname(os.path.abspath(__file__)) 
	svhnpath  = cwdpath.replace('cnn','projects/svhn/')
	return svhnpath

def flatten(X):
	# input will be (32, 32, 3, N)
	# output will be (N, 3072)
	N = X.shape[-1]
	flat = np.zeros((N, 3072))
	for i in xrange(N):
	    flat[i] = X[:,:,:,i].reshape(3072)
	return flat


def error_rate(p, t):
	return np.mean(p != t)

def benchmark(): 
	
	train = loadmat(get_svhnpath() + '/train_32x32.mat')
	test  = loadmat(get_svhnpath() + '/test_32x32.mat')

	
  # Need to scale! don't leave as 0..255
  # Y is a N x 1 matrix with values 1..10 (MATLAB indexes by 1)
  # So flatten it and make it 0..9
	# Also need indicator matrix for cost calculation
	Xtrain = flatten(train['X'].astype(np.float32) / 255)
	Ytrain = train['y'].flatten() -1 #matlab arrays are 1 indexed
	Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
	Ytrain_ind = y2indicator(Ytrain)

	Xtest = flatten(test['X'].astype(np.float32) / 255)
	Ytest = test['y'].flatten() -1 #matlab arrays are 1 indexed
	Xtest, Ytest = shuffle(Xtest, Ytest)
	Ytest_ind = y2indicator(Ytest)

	max_iter = 20 
	print_period=10
	N,D  = Xtrain.shape 
	batch_size = 500
	n_batches = N / batch_size


	# initial weights
 	M1 = 1000 # hidden layer size
 	M2 = 500
 	K = 10
	W1_init = np.random.randn(D, M1) / np.sqrt(D + M1)
	b1_init = np.zeros(M1)
	W2_init = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
	b2_init = np.zeros(M2)
	W3_init = np.random.randn(M2, K) / np.sqrt(M2 + K)
	b3_init = np.zeros(K)

	# define variables and expressions
	X = tf.placeholder(tf.float32, shape=(None, D), name='X')
	T = tf.placeholder(tf.float32, shape=(None, K), name='T')
	W1 = tf.Variable(W1_init.astype(np.float32))
	b1 = tf.Variable(b1_init.astype(np.float32))
	W2 = tf.Variable(W2_init.astype(np.float32))
	b2 = tf.Variable(b2_init.astype(np.float32))
	W3 = tf.Variable(W3_init.astype(np.float32))
	b3 = tf.Variable(b3_init.astype(np.float32))

	Z1 = tf.nn.relu( tf.matmul(X, W1) + b1 )
	Z2 = tf.nn.relu( tf.matmul(Z1, W2) + b2 )
	Yish = tf.matmul(Z2, W3) + b3

	cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish, labels=T))

	train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

	# we'll use this to calculate the error rate
	predict_op = tf.argmax(Yish, 1)

	t0 = datetime.now()
	LL = []
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)

		for i in xrange(max_iter):
			for j in xrange(n_batches):
				Xbatch = Xtrain[j*batch_size:(j*batch_size + batch_size),]
				Ybatch = Ytrain_ind[j*batch_size:(j*batch_size + batch_size),]

				session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
				if j % print_period == 0:
				  test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest_ind})
				  prediction = session.run(predict_op, feed_dict={X: Xtest})
				  err = error_rate(prediction, Ytest)
				  
				  print "Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err)
				  LL.append(test_cost)
	print "Elapsed time:", (datetime.now() - t0)
	plt.plot(LL)
	plt.show()

if __name__ == '__main__':
	benchmark()
