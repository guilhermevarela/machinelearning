'''
Created on Ago 26, 2017

@author: Varela
motivation: Compute Fibonnachi using theano scan
course url: https://www.udemy.com/deep-learning-recurrent-neural-networks-in-python/learn/v4/t/lecture/5359716?start=0
'''

import numpy as np 
import theano 
import theano.tensor as T 

import matplotlib.pyplot as plt 

X = 2*np.random.randn(300) + np.sin(np.linspace(0, 3*np.pi, 300))
plt.plot(X)
plt.title('original')
plt.show()

decay = T.scalar('decay')
sequence = T.vector('sequence', dtype='float32')



def recurrence(x, last, decay):
	return (1-decay)*x + decay*last 


outputs, _ = theano.scan(
	fn=recurrence,
	sequences=sequence,	
	n_steps=sequence.shape[0],
	outputs_info=[np.float32(0)], 
	non_sequences=[decay]
)	

lpf = theano.function(
	inputs=[sequence, decay],
	outputs=outputs,
	allow_input_downcast=True
)

Y = lpf(X, 0.99)
plt.plot(Y)
plt.title('filtered')
plt.show()



