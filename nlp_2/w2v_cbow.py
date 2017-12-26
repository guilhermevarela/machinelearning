'''
	Created on Dec 20, 2017

	@author: Varela

	Tensorflow eazy cbow

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


def sentences2XY_batch(X_batch, Y_batch, sentences, j):	
	stop=False
	batch_n=0	
	batch_sz, C, V= X_batch.shape	
	d=int((C-1)/2)
	
	aux1= np.arange(C)
	for jj in range(j, len(sentences)):
		sentence=[0]+sentences[jj]+[1]
		n_examples= len(sentence)-C+1
		for n in range(n_examples):
			if (batch_n < batch_sz):				
				X_batch[batch_n, aux1, sentence[n:n+C]]=1
				Y_batch[batch_n, n+d]=1
			else:
				stop=True
				break
			batch_n+=1
		if stop:
			break	

	return X_batch, Y_batch, jj, stop 

def cbow():
	n_files=10
	V=2000
	D=300
	C=3
	batch_sz = 500

	sentences, word2idx=get_wikipedia_data(n_files=n_files, n_vocab=V, by_paragraph=True)
	epochs = 20
	print_period = 10 
	lr = 1e-4
	reg =  0.01 

	
	
	n_examples=count_examples(sentences, C)
	n_batches =int( n_examples / batch_sz)

	

	W1_init = np.random.randn(V+1, D) / np.sqrt(D+V)
	W2_init = np.random.randn(D,V+1) / np.sqrt(D+V)
	


	X = tf.placeholder(tf.float32, shape=(batch_sz, C, V+1), name='X')
	T = tf.placeholder(tf.float32, shape=(batch_sz, V+1), name='T')



	W1 = tf.Variable(W1_init.astype(np.float32), name='W1')
	W2 = tf.Variable(W2_init.astype(np.float32), name='W2')
	

	X_2d= tf.reshape(X, [-1, V+1])
	Z_2d= tf.matmul( X_2d, W1, name='Z_2d') 	
	Z= tf.reshape(Z_2d, [-1, C, D])
	H= tf.reduce_mean(Z, axis=1, name='H')
	Yish= tf.matmul( H, W2 )
	cost= tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=T, logits=Yish))

	train_op= tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)
	predict_op= tf.argmax(Yish, 1)

	LL= [] 
	init= tf.global_variables_initializer()

	X_ind= np.zeros((batch_sz, C,V+1), dtype=np.int32) # batch
	Y_ind= np.zeros((batch_sz, V+1), dtype=np.int32) # batch 
	with tf.Session() as session:
		session.run(init)

		for i in range(epochs):
			sentences= shuffle(sentences)

			sc=0
			for j in range(n_batches):				
				X_ind, Y_ind, sc, process= sentences2XY_batch(X_ind, Y_ind, sentences, sc)
				
				#takes at least batch_sz examples to run
				if process:
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

				X_ind[X_ind]=0
				Y_ind[Y_ind]=0

	plt.plot(LL)
	plt.show()


def sentences2X_list(sentences): 
	'''
		INPUT
			sentences<list<lists>>: list of lists of integer indexes
				expected to be a batch from indexes

		OUTPUT
			X_list[M,1] one-hot encoding for sentences
					M: examples

	'''
	X_list = [item for sublist in sentences for item in ([0]+sublist+[1])]	
	return X_list

def count_examples(sentences, C):
	c=0
	for sentence in sentences:
		c+=len(sentence)-C+1	
	return c 	

if __name__ == '__main__':
	cbow()