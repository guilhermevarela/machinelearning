'''
Created on Ago 26, 2017

@author: Varela
motivation: Theano scan function
course url: https://www.udemy.com/deep-learning-recurrent-neural-networks-in-python/learn/v4/t/lecture/5359716?start=0
'''

import numpy as np 
import theano 
import theano.tensor as T 



x = T.ivector('X')

def square(x):
	return x*x 

outputs, updates = theano.scan(
	fn=square,
	sequences=x,
	n_steps=x.shape[0]
)	

square_op = theano.function( 
	inputs=[x], 
	outputs=[outputs]
)

o_val = square_op(np.array([1,2,3,4,5,6], dtype=np.int32))	

print "ouput:", o_val[0]

