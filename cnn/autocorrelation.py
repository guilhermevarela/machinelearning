'''
Created on Ago 09, 2017

@author: Varela

For the class Data Science: Deep Learning convolutional neural networkds on theano and tensorflow
lecture #11 Alternating ways to view convolution
				Autocorrelation
				Crosscorrelation
				Convolution of inverse filter <=> Cross correlation
course url: https://udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow
lecture url: https://www.udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow/learn/v4/t/lecture/6620952?start=0

'''

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import correlate, convolve

#Gaussian noise
X = np.random.randn(1000)

C1 = correlate(X, X)
plt.plot(C1)
plt.show()


#Cross correlations: with lag 100
Y = np.empty(1000)
Y[:900] = X[100:]
Y[900:] = X[:100]
C2 = correlate(X,Y)
plt.plot(C2)
plt.show()

#Convolution of the inverse filter
C3 = convolve(X,np.flip(Y, 0))

assert((np.sqrt(C3**2-C2**2)).sum() < 1e-10)
plt.plot(C3)
plt.show()


