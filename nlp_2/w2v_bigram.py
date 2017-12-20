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


def main():
	n_files=10
	N=2000
	D =300

	sentences, word2idx=get_wikipedia_data(n_files=10, n_vocab=N, by_paragraph=True)


	max_iter = 20
	print_period = 10 
	lr = 0.00004
	reg =  0.01 

	
	
	batch_sz = 500
	n_batches =int( N / batch_sz)

	

	W1_init = np.random.randn(D, N+1) / np.sqrt(D+N)
	W2_init = np.random.randn(N+1, D) / np.sqrt(D+N)
	


	X = tf.placeholder(tf.float32, shape=(N+1,1), name='X')
	T = tf.placeholder(tf.float32, shape=(N+1,1), name='T')



	W1 = tf.Variable(W1_init.astype(np.float32), name='W1')
	W2 = tf.Variable(W2_init.astype(np.float32), name='W2')
	

	Z = tf.matmul( W1,X ) 	
	Yish = tf.matmul( W2,Z )
	# Yish = tf.matmul( W2, tf.matmul( W1,X ) )
	cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=T, logits=Yish))

	train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)
	predict_op = tf.argmax(Yish, 1)

	LL = [] 
	init = tf.global_variables_initializer()

	Xind = np.zeros((N+1,1), dtype=np.int32) # cononical base N
	Yind = np.zeros((N+1,1), dtype=np.int32) # cononical base N
	with tf.Session() as session:
		session.run(init)

		for i in range(max_iter):
			sentences = shuffle(sentences)
			
			for j in range(n_batches):				
				# import code; code.interact(local=dict(globals(), **locals()))
				sb = sentences[j*batch_sz:((j+1)*batch_sz)]
				# sb = [item for sublist in sb for item in sublist]
				# Xbatch = [0] + sb
				# Ybatch = sb + [1]
				# n_words_in_batch=len(Xbatch)
				X_ind, Y_ind=sentences2mtrix(sentences, N+1)
				# for k in range(n_words_in_batch):
					# Xind[Xbatch[k]]=1 
					# Yind[Ybatch[k]]=1 

				session.run(train_op,
					feed_dict={
						X: X_ind,
						T: Y_ind,
					})

				if j % print_period == 0:
					test_cost = session.run(cost, feed_dict={X: X_ind, T: Y_ind})
						# prediction_val = session.run(predict_op, feed_dict={X: Xtest})

						# err = error_rate(prediction_val, Ytest)
						# print("Cost at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
					print("Cost at iteration i=%d, j=%d: %.3f" % (i, j, test_cost))
					LL.append(test_cost)

	plt.plot(LL)
	plt.show()

def sentences2mtrix(sentences, V):
	'''
		INPUT
			sentences<list<lists>>: list of lists of integer indexes
				expected to be a batch from indexes

		OUTPUT
			X[M,V] one-hot encoding for sentences
					M: examples, V: vocabulary size

			Y[M,V] one-hot encoding for output sentences
					M: examples, V: vocabulary size

	'''
	xidx = [item for sublist in sentences for item in ([0]+sublist)]
	yidx = [item for sublist in sentences for item in (sublist+[1])]

	M=len(xidx)

	index=np.arange(M)
	X=np.zeros((M,V), dtype=np.int32)
	Y=np.zeros((M,V), dtype=np.int32)

	X[index,np.array(xidx)]=1
	Y[index,np.array(yidx)]=1

	return X, Y

if __name__ == '__main__':
	main()