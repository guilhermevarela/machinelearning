'''
Created on Ago 26, 2017

@author: Varela
motivation: Compute Fibonnaci using theano scan
course url: https://www.udemy.com/deep-learning-recurrent-neural-networks-in-python/learn/v4/t/lecture/5359716?start=0
'''

import numpy as np 
import theano 
import theano.tensor as T 


N = T.iscalar('N')



def recurrence(n, fn_1, fn_2):
	return fn_1+fn_2, fn_1 


outputs, updates = theano.scan(
	fn=recurrence,
	sequences=T.arange(N),
	n_steps=N,
	outputs_info=[1,1], 
)	

fibonnaci = theano.function(
	inputs=[N],
	outputs=outputs
)

o_val = fibonnaci(np.int32(8))

print "output:", o_val

