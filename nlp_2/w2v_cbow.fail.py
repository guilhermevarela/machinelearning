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

def count_examples(sentences, n):
	'''
		INPUT
			sentences<list<lists>>: list of lists of integer indexes
				expected to be a batch from indexes

			n<int>	: context size (1-sided)
	'''
	c=0
	for sentence in sentences:
		#adding BEGIN and END
		#missed examples from window		
		c+=len(sentence)+2-2*n
	return c


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


def cbow_fn( W, X_ind, prev, interval):
	return tf.matmul(W, X_ind[interval[0]:interval[1],:])

def cbow01(sentences, V, D, epochs=20, batch_sz=1000, print_period=100, lr=1e-4, reg=0.01):
	'''
		Performs continous bag-of-words with moving context of size 1(1-left, 1-right)
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
	
	C=1
	n_batches =int(len(sentences) / batch_sz)
	print('total number of examples: \tvocabulary size:%d\tbatches:%d' % (V, n_batches))

	W1_init = np.random.randn(V+1,D) / np.sqrt(D+V+1)	
	W2_init = np.random.randn(V+1,D) / np.sqrt(D+V+1)	


	# Y_bounds = tf.placeholder(tf.int32, shape=(None, 2), name='Y_bounds')
	X = tf.placeholder(tf.float32, shape=(2*C+1, V+1), name='X')
	T = tf.placeholder(tf.float32, shape=(V+1,1), name='T')

	W1 = tf.Variable(W1_init.astype(np.float32), name='W1')
	W2 = tf.Variable(W2_init.astype(np.float32), name='W2')




	# cbow_init=np.zeros((2*C+1, D), dtype=np.int32)
	Z 	 = tf.reduce_mean( tf.matmul(X, W1) , axis=0, keep_dims=True)
	Yish = tf.matmul(W1,	tf.transpose(Z)) 
	cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=T, logits=Yish))
	
	init 				= tf.global_variables_initializer()
	train_op 		= tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)
	predict_op 	= tf.argmax(Yish, 1)


	with tf.Session() as session:
		session.run(init)

		for i in range(epochs):
			sentences= shuffle(sentences)
			
			for s,sentence in enumerate(sentences):
				X_list= [0]+sentence+[1]
				Y_list= X_list[C:-C]
				
				lx= len(X_list)
				ly= len(Y_list) 

				X_ind=np.zeros((lx, V+1),dtype=np.int32)
				Y_ind=np.zeros((ly, V+1),dtype=np.int32)

				X_ind[np.arange(lx),X_list]=1
				Y_ind[np.arange(ly),Y_list]=1

				intervals=[[yy-C,yy+C+1] for yy in range(C,lx-(C))]
				for ii,interval in enumerate(intervals):
					session.run(train_op, feed_dict={
						X: X_ind[interval[0]:interval[1],:],
						T: Y_ind[ii,:].reshape((V+1,1))
					})

					if s % print_period == 0:
						test_cost = session.run(cost, feed_dict={X: X_ind, T: Y_ind})
						prediction_val = session.run(predict_op, feed_dict={X: X_ind})

						err = error_rate(prediction_val, Y_ind)
						print("Cost at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
						LL.append(test_cost)

				# Y_interval=[[yy-C,yy+C+1] for yy in range(C,lx-(C)) ]

				# import code; code.interact(local=dict(globals(), **locals()))
				
				# fn= lambda x,y : cbow_fn(W1, X_ind, x, y)
				# cbow_op= tf.scan(
				# 	fn=fn,
				# 	elems=Y_bounds, 		
				# 	initializer=cbow_init,
				# 	name='cbow_op'
				# )
				# Z = session.run(cbow_op,
				# 	feed_dict={
				# 		fn: fn,
				# 		Y_bounds:Y_interval
				# })

				





			# for j in range(n_batches):
			# 	batch_sentences= sentences[j*batch_sz:(j+1)*batch_sz]	
			# 	X_list=sentences2X_list(batch_sentences)
			# 	# Since we're shuffling on the sentences we are making variable
			# 	# number of examples - 
			# 	M= count_examples(sentences, C)
			# 	X_list2X_ind(M, V, X_list)



			# for j in range(M):
			# 	auxx = X_list[j*(2*C+1):(j+1)*(2*C+1)]
			# 	auxy = Y_list[j+C]
			# 	# import code; code.interact(local=dict(globals(), **locals()))
			# 	X_ind[aux0, auxx]=1
			# 	Y_ind[auxy,0]=1
				
			# 	session.run(train_op,
			# 		feed_dict={
			# 			X: X_ind,
			# 			T: Y_ind,
			# 		})

			# 	if j % print_period == 0:
			# 		test_cost = session.run(cost, feed_dict={X: X_ind, T: Y_ind})
			# 		prediction_val = session.run(predict_op, feed_dict={X: X_ind})

			# 		err = error_rate(prediction_val, Y_ind)
			# 		print("Cost at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
			# 		LL.append(test_cost)
				
			X_ind[aux0, aux1, auxx]=0
			Y_ind[aux1, auxy]=0
	
	plt.plot(LL)
	plt.show()



if __name__=='__main__':
	n_files=10
	V=2000
	D=200
	sentences, word2idx=get_wikipedia_data(n_files=n_files, n_vocab=V, by_paragraph=True)
	cbow01(sentences, V, D)