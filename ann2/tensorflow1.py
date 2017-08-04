'''
Created on Ago 04, 2017

@author: Varela

Tensorflow basics
For the class Data Science: Practical Deep Learning Concepts in Theano and TensorFlow
course url: https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
lecture url: https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow/learn/v4/t/lecture/4627284?start=0

'''

import numpy as np 
import tensorflow as tf 

def main():
	#You have to specify the type: shape & name are optional 
	A = tf.placeholder(tf.float32, shape=(5,5), name='A')

	#but the shape and name are optional
	v = tf.placeholder(tf.float32)

	w = tf.matmul(A, v)

	#Similar to theano we have to feed the variable names
	with tf.Session() as session:
		output = session.run(
			w,
			feed_dict={
				A: np.random.randn(5,5),
				v: np.random.rand(5,1)
			}
		)

	print output, type(output)

	#Tensorflow variables are like theano shared variables
	shape=(2,2)
	x = tf.Variable(tf.random_normal(shape))
	t = tf.Variable(0)


	init = tf.global_variables_initializer()
	with tf.Session() as session:
		out  = session.run(init)	
		print out
		print x.eval()
		print t.eval()


	#Let's minimize a function
	u = tf.Variable(20.0)

	cost = u*u + u +1 

	train_op = tf.train.GradientDescentOptimizer(0.3).minimize(cost)
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)

		for i in xrange(12):
			session.run(train_op)
			print "i = %d, cost %.3f, u= %.3f" % (i, cost.eval(), u.eval())
if __name__ == '__main__':
	main()
