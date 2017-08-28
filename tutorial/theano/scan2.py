'''
Created on Ago 27, 2017

@author: Varela

motivation: Theano.Scan tutorial example 2 generate the components of the polynomial
  Scan url: http://deeplearning.net/software/theano/library/scan.html
'''

import numpy as np 
import theano  
import theano.tensor as T 

print  'Iterating over the first dimension of a tensor: Calculating a polynomial'
coefficients = T.vector('coefficients')
x = T.scalar('x')

max_coeffients_supported = 10000
#generate the components of the polynomial
#In addition to looping a fixed number of times, scan can iterate over the leading dimension of tensors 
#Second, there is no accumulation of results, we can set outputs_info to None. This indicates to scan that it doesn’t need to pass the prior result to fn.
#The general order of function parameters to fn is:
components, updates = theano.scan(
	fn=lambda coefficient, power, free_variable: coefficient*(free_variable **power),
	outputs_info=None,
	sequences=[coefficients, T.arange(max_coeffients_supported)],  # sequences truncate to the smallest of the sequences
	non_sequences=x 
)
#First, we calculate the polynomial by first generating each of the coefficients, and then summing them at the end. (We could also have accumulated them along the way, and then taken the last one, which would have been more memory-efficient, but this is an example.)
# sum them up
polynomial = components.sum()

#Compile a function
calculate_polynomial = theano.function(inputs=[coefficients, x], outputs=polynomial)

#Test 
test_coefficients  = np.asarray([1,0,2], dtype=np.float32)
test_value = 3 
print(calculate_polynomial(test_coefficients, test_value))
print(1.0*(3 ** 0) + 0.0*(3 ** 1) + 2.0 * (3 ** 2))
