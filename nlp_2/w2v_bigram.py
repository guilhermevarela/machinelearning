'''
	Created on Dec 20, 2017

	@author: Varela

	Tensorflow bigram

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


def bigram():
	n_files=10
	N=2000
	D =300

	sentences, word2idx=get_wikipedia_data(n_files=10, n_vocab=N, by_paragraph=True)
	X_list, Y_list=sentences2XY_list(sentences)

	epochs = 20
	print_period = 10 
	lr = 1e-4
	reg =  0.01 

	
	
	batch_sz = 15000
	n_batches =int( len(X_list) / batch_sz)

	

	W1_init = np.random.randn(N+1, D) / np.sqrt(D+N)
	W2_init = np.random.randn(D,N+1) / np.sqrt(D+N)
	


	X = tf.placeholder(tf.float32, shape=(None, N+1), name='X')
	T = tf.placeholder(tf.float32, shape=(None, N+1), name='T')



	W1 = tf.Variable(W1_init.astype(np.float32), name='W1')
	W2 = tf.Variable(W2_init.astype(np.float32), name='W2')
	

	Z = tf.matmul( X,W1 ) 	
	Yish = tf.matmul( Z, W2 )
	cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=T, logits=Yish))

	train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)
	predict_op = tf.argmax(Yish, 1)

	LL = [] 
	init = tf.global_variables_initializer()

	aux= np.arange(batch_sz)
	X_ind = np.zeros((batch_sz, N+1), dtype=np.int32) # cononical base N
	Y_ind = np.zeros((batch_sz, N+1), dtype=np.int32) # cononical base N	
	with tf.Session() as session:
		session.run(init)

		for i in range(epochs):
			X_list, Y_list = shuffle(X_list, Y_list)
			
			for j in range(n_batches):				
				X_ind[aux,X_list[j*batch_sz:(j+1)*batch_sz]]=1
				Y_ind[aux,Y_list[j*batch_sz:(j+1)*batch_sz]]=1


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

				X_ind[aux,X_list[j*batch_sz:(j+1)*batch_sz]]=0
				Y_ind[aux,Y_list[j*batch_sz:(j+1)*batch_sz]]=0

	plt.plot(LL)
	plt.show()


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

if __name__ == '__main__':
	bigram()