'''
Created on Ago 10, 2017

@author: Varela

For the class Data Science: Deep Learning convolutional neural networkds on theano and tensorflow
lecture #21 CNN in tensorflow

New concepts and differences from Theano:
- stride is the interval at which to apply the convolution
- unlike previous course, we use constant-size input to the network
  since not doing that caused us to start swapping
- the output after convpool is a different size (8,8) here, (5,5) in Theano

course url: https://udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow
lecture url: https://www.udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow/learn/v4/t/lecture/4847770?start=0

'''
import os 
import numpy as np 
import matplotlib.pyplot as plt 

import tensorflow as tf 


from scipy.io import loadmat 


from datetime import datetime 
from util import y2indicator 

def error_rate(p, t):
	return np.mean(p != t)

def relu(a):	
	return a * (a>0)

def convpool(X, W, b):
	#just assume pool size is (2,2) because we need to augment it with 1s

	conv_out = tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')	
	conv_out = tf.nn.bias_add(conv_out, b)	
	pool_out = tf.nn.max_pool(
		conv_out,
		ksize=[1,2,2,1],
		strides=[1,2,2,1],
		padding='SAME'
	)
	return tf.nn.relu(pool_out) 



def init_filter(shape, poolsz):
	w = np.random.randn(*shape)	/ np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
	return w.astype(np.float32)

#input is (32,32,3,N)
#output is (N,32,32,3)
def rearrange(X):
	N = X.shape[-1]
	out = np.zeros((N,32,32,3), dtype=np.float32)
	for i in xrange(N):
		for j in xrange(3):
			out[i, :, :, j] = X[:, :, j, i]
	return out / 255

def get_svhnpath():
	cwdpath =  os.path.dirname(os.path.abspath(__file__)) 
	svhnpath  = cwdpath.replace('cnn','projects/svhn/')
	return svhnpath

# def flatten(X):
# 	# input will be (32, 32, 3, N)
# 	# output will be (N, 3072)
# 	N = X.shape[-1]
# 	flat = np.zeros((N, 3072))
# 	for i in xrange(N):
# 	    flat[i] = X[:,:,:,i].reshape(3072)
# 	return flat


def main():
	#same from benchmark
	train = loadmat(get_svhnpath() + '/train_32x32.mat')
	test  = loadmat(get_svhnpath() + '/test_32x32.mat')

	
  # Need to scale! don't leave as 0..255
  # Y is a N x 1 matrix with values 1..10 (MATLAB indexes by 1)
  # So flatten it and make it 0..9
	# Also need indicator matrix for cost calculation
	Xtrain = rearrange(train['X'])
	Xtrain = Xtrain[:73000,]
	Ytrain = train['y'].flatten() -1 #matlab arrays are 1 indexed
	Ytrain = Ytrain[:73000,]

	del train 
	# Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
	Ytrain_ind = y2indicator(Ytrain)

	Xtest = rearrange(test['X'])
	Xtest = Xtest[:26000,]
	Ytest = test['y'].flatten() -1 #matlab arrays are 1 indexed
	Ytest = Ytest[:26000,]
	del test
	Ytest_ind = y2indicator(Ytest)

	#Gradient descent parameters
	max_iter = 20
	print_period=10
	N 	= Xtrain.shape[0] 

	# lr = np.float32(1e-5)
	# reg = np.float32(1e-2)
	# mu = np.float32(1.0 - 1e-2)


	N  = Xtrain.shape[0] 
	batch_size = 500
	n_batches = N / batch_size

	# HIDDEN LAYER/OUTPUT
	M = 500 
	K = 10
	poolsz = (2,2)

	#1rst ConvPool layer
	# after conv will be of dimension 32 - 5 + 1 = 28
	# after downsample 28 / 2 = 14
	W1_shape = 	(5, 5, 3, 20) # (filter_width, filter_height, num_color_channels, num_feature_maps)
	W1_init  = 	init_filter(W1_shape, poolsz)
	b1_init  = 	np.zeros(W1_shape[-1], dtype=np.float32)

	#2nd ConvPool layer
	# after conv will be of dimension 14 - 5 + 1 = 10
	# after downsample 10 / 2 = 5
	W2_shape = 	(5, 5, 20, 50) # (filter_width, filter_height, num_color_channels, num_feature_maps)
	W2_init  = 	init_filter(W2_shape, poolsz)
	b2_init  = 	np.zeros(W2_shape[-1], dtype=np.float32)

	
	# vanilla ANN weights
	#1rst MLP Input-to-hidden layer
	W3_init  = 	np.random.randn(W2_shape[-1]*8*8, M) / np.sqrt(W2_shape[-1]*8*8 + M)
	b3_init  = 	np.zeros(M, dtype=np.float32)

	#2nd MLP Hidden-to-output layer
	W4_init  = 	np.random.randn(M,K) / np.sqrt(M + K)
	b4_init  = 	np.zeros(K, dtype=np.float32)



	# step 2: define tensorflow variables
	X = tf.placeholder(tf.float32, shape=(batch_size, 32, 32, 3), name='X')
	T = tf.placeholder(tf.float32, shape=(batch_size, K), name='T')

	# Y = T.matrix('T')

	W1 = tf.Variable(W1_init.astype(np.float32), name='W1')
	b1 = tf.Variable(b1_init.astype(np.float32), name='b1')
	W2 = tf.Variable(W2_init.astype(np.float32), name='W2')
	b2 = tf.Variable(b2_init.astype(np.float32), name='b2')
	W3 = tf.Variable(W3_init.astype(np.float32), name='W3')
	b3 = tf.Variable(b3_init.astype(np.float32), name='b3')
	W4 = tf.Variable(W4_init.astype(np.float32), name='W4')
	b4 = tf.Variable(b4_init.astype(np.float32), name='b4')

	#forward pass
	Z1 = convpool(X, W1, b1)
	Z2 = convpool(Z1, W2, b2)
	
	Z2_shape = Z2.get_shape().as_list()
	Z2r      = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])])
	Z3 			 = tf.nn.relu(tf.matmul(Z2r, W3) + b3)
	Yish     = tf.matmul(Z3, W4) + b4 
	
	
	cost      = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish, labels=T))
	train_op  = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)
	#we'll use this to calculate error rate
	prediction_op = tf.argmax(Yish, 1)

	t0 = datetime.now()
	LL = []
	init = tf.global_variables_initializer()
	with tf.Session() as session: 
		session.run(init)

		for i in xrange(max_iter):
			for j in xrange(n_batches):				
				Xbatch = Xtrain[j*batch_size:((j+1)*batch_size), :]
				Ybatch = Ytrain_ind[j*batch_size:((j+1)*batch_size), :]
				
				if len(Xbatch) == batch_size:
					session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})					
					if j % print_period == 0:
						#do due to RAM limitations we need to have a fixed size input 
						#so as result, we have this ugly total cost and prediction computation
						test_cost =0
						prediction= np.zeros(len(Xtest))
						for k in xrange(len(Xtest)/ batch_size):
							Xtestbatch = Xtest[k*batch_size:((k+1)*batch_size), :]	
							Ytestbatch = Ytest_ind[k*batch_size:((k+1)*batch_size), :]
							test_cost += session.run(cost, feed_dict={X:Xtestbatch, T: Ytestbatch})
							prediction[k*batch_size:((k+1)*batch_size), :] = session.run(
								prediction_op, feed_dict={X: Xtestbatch}
							)
						
						err = error_rate(prediction, Ytest)
						print "Cost at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err)
						LL.append(test_cost)
	print "elapsed time", datetime.now() - t0 				
	plt.plot(LL)
	plt.show()


if __name__ == '__main__':
	main()
