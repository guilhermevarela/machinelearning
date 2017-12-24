'''
	Created on Dec 24, 2017

	@author: Varela

	Tensorflow

	matmul tutorial
	ref: https://www.tensorflow.org/api_docs/python/tf/matmul

'''

import numpy as np 
import tensorflow as tf 



a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
# `a` * `b`
# [[ 58,  64],
#  [139, 154]]
c = tf.matmul(a, b, name='2d2d_matmul')

a3= tf.stack([a,a])
c3= tf.tile(c, multiples=tf.constant([2,1], shape=[3, 2]))
# c3= tf.reshape(
# 	tf.tile(c, multiples=[2]), 
# 	[2,1,1]
# )
# d = tf.matmul(c3, a3, name='3d2d_matmul')

init= tf.global_variables_initializer()
with tf.Session() as session:
	session.run(init)

	C= session.run(c)
	print(C, C.shape)
	A3= session.run(a3)
	print(A3, A3.shape)
	C3= session.run(c3)
	print(C3, C3.shape)

	# D= session.run(d)
	# print(D, D.shape)
