'''
Created on Ago 27, 2017

@author: Varela

motivation: Theano.Scan tutorial example 1 
  Scan url: http://deeplearning.net/software/theano/library/scan.html
'''

import numpy as np 
import theano  
import theano.tensor as T 


print 'Simple loop with accumulation A^k'
# How to represent in theano
# result = 1 
# for i in range(K):
# 	result  *= A 
# ------x Loop x---------   		----x Theano	x----
# A is an unchaging value 	==> non_sequences
# result initial value of 1 ==> outputs_info, it's going to be passed down as output
# accumulation on results   ==> happens automaticaly
# number of iterations      ==> number of steps 

k = T.iscalar('k')
A = T.vector('A')

# def prod(result, vector): 
# 	return result * vector 


# Symbolic description of the result
result, updates = theano.scan(
	fn=lambda prior, A : prior * A, #fn call must come first
	non_sequences=A, 								#followed by non sequences: immutable
	outputs_info=T.ones_like(A), 		#outputs startup with 1's
	n_steps=K  											#number of steps to perform  
)

# result, updates = theano.scan(
# 	fn=prod, 
# 	non_sequences=A,
# 	outputs_info=T.ones_like(A), 
# 	n_steps=k
# )


# We only care about A**k, but scan has provided us with A**1 through A**k.
# Discard the values that we don't care about. Scan is smart enough to
# notice this and not waste memory saving them.
final_result = result[-1]

# compiled function that returns A**k
power = theano.function(inputs=[A,k], outputs=final_result, updates=updates)

print(power(range(10),2))
print(power(range(10),4))



