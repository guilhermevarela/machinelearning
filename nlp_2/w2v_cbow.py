'''
	Created on Dec 20, 2017

	@author: Varela

	Tensorflow cbow

	url: https://www.udemy.com/natural-language-processing-with-deep-learning-in-python/learn/v4/t/lecture/5505608?start=0

'''
import numpy as np 
import tensorflow as tf 

import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import sys
import os
sys.path.append(os.path.abspath('..'))
from rnn.util import get_wikipedia_data

def error_rate(p, t):
	return np.mean(p !=t )

def sentences2XY_list(sentences): 
	'''
		INPUT
			sentences<list<lists>>: list of lists of integer indexes
				expected to be a batch from indexes

		OUTPUT
			X_list[M,1] one-hot encoding for sentences
					M: examples

			Y_list[M,1] one-hot encoding for output sentences
					M: examples, V: vocabulary size

	'''
	examples_list = [item for sublist in sentences for item in ([0]+sublist+[1])]

	XX=examples_list[:-1]
	YY=examples_list[1:]
	
	Z = [(x,y) for x,y in zip(XX,YY) if not(x==1) and not(y==0)]
	X_tuple, Y_tuple = zip(*Z)
	
	return list(X_tuple), list(Y_tuple)

def cbow10(sentences, V, D, epochs=20, batch_sz=15000, print_period=10, lr=1e-4, reg=0.01):
	'''
		Performs continous bag-of-words with moving context of size 10(5-left, 5-right)
		INPUT
			sentences
			V <int>: Vocabulary size
			D <int>: Hidden layer size
			epochs<int>: number of iterations over the dataset
			batch_sz<int>: use stochast batch of size batch_sz
			print_period<int>: print results only once at every print_period
			lr<float>
			reg<float>

		OUTPUT
	'''
	X_list, Y_list=sentences2XY_list(sentences)		
	C=5
	n_batches =int((len(X_list)-2*C) / batch_sz)
	print('total words:%d\tvocabulary size:%d\tbatches:%d' % (len(X_list), V, n_batches))

	W1_init = np.random.randn(V+1,D) / np.sqrt(D+V+1)	
	W2_init = np.random.randn(D,V+1) / np.sqrt(D+V+1)	

	X = tf.placeholder(tf.float32, shape=(2*C, batch_sz, V+1), name='X')
	T = tf.placeholder(tf.float32, shape=(batch_sz, V+1), name='T')

	W1 = tf.Variable(W1_init.astype(np.float32), name='W1')
	W2 = tf.Variable(W2_init.astype(np.float32), name='W2')

	Z 	 = tf.reduce_mean( np.matmul(X, W1) ,axis=0)
	Yish = tf.matmul(Z, W3) 
	cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=T, logits=Yish))
	
	init = tf.global_variables_initializer()
	train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)
	predict_op = tf.argmax(Yish, 1)


	aux0=np.arange(2*C+1)
	aux1=np.arange(batch_sz)
	X_ind= np.zeros((2*C+1, batch_sz, V), dtype=np.int32)
	Y_ind= np.zeros((batch_sz), dtype=np.int32)
	with tf.Session() as session:
		session.run(init)

		for i in range(epochs):
			X_list, Y_list= shuffle(X_list, Y_list)

			for j in range(n_batches):
				auxx = X_list[j*batch_sz:(j+1)*batch_sz+(2*C+1)]
				auxy = Y_list[j*batch_sz+C:(j+1)*batch_sz+C]
				X_ind[aux0, aux1, auxx]=1
				Y_ind[aux1, auxy]=1
				
				session.run(train_op,
					feed_dict={
						X: X_ind,
						T: Y_ind,
					})

				if j % print_period == 0:
					test_cost = session.run(cost, feed_dict={X: X_ind, T: Y_ind})
					prediction_val = session.run(predict_op, feed_dict={X: X_ind})

					err = error_rate(prediction_val, Y_ind)
					print("Cost at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
					LL.append(test_cost)
				
			X_ind[aux0, aux1, auxx]=0
			Y_ind[aux1, auxy]=0
	
	plt.plot(LL)
	plt.show()



if __name__=='__main__':
	n_files=10
	V=2000
	D=200
	sentences, word2idx=get_wikipedia_data(n_files=n_files, n_vocab=V, by_paragraph=True)
	cbow10(sentences, V, D)