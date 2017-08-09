'''
Created on Ago 09, 2017

@author: Varela

For the class Data Science: Deep Learning convolutional neural networkds on theano and tensorflow
lecture #10; writing a custom convolve2d function
course url: https://udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow
lecture url:https://www.udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow/learn/v4/t/lecture/5486160?start=0

'''

import numpy as np 
# from scipy.signal import convolve2d 

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
from datetime import datetime 

#slow
# def convolve2d(X, W):
# 	t0 = datetime.now()
# 	n1, n2 = X.shape 
# 	m1, m2 = W.shape 

# 	Y = np.zeros((n1 + m1-1, n2 + m2-1))
# 	for i in xrange(n1 + m1 -1):
# 		for ii in xrange(m1):
# 			for j in xrange(n2 + m2 -1):
# 				for jj in xrange(m2):
# 					if i >= ii and j >= jj and i - ii < n1 and j - jj < n2:
# 						Y[i,j] += W[ii,jj]*X[i - ii, j - jj]
	
# 	print "elapsed time:", (datetime.now() - t0) 
# 	return Y 

#10x faster
# def convolve2d(X, W):
# 	t0 = datetime.now()
# 	n1, n2 = X.shape 
# 	m1, m2 = W.shape 
# 	Y = np.zeros((n1 + m1-1, n2 + m2-1))

# 	for i in xrange(n1):	
# 		for j in xrange(n2):
# 			Y[i:i+m1,j:j+m2] += X[i,j]*W 
	
# 	print "elapsed time:", (datetime.now() - t0) 
# 	return Y 

#10x faster + keep same dimensions of input mode='same'
def convolve2d(X, W):
	t0 = datetime.now()
	n1, n2 = X.shape 
	m1, m2 = W.shape 
	Y = np.zeros((n1 + m1-1, n2 + m2-1))

	for i in xrange(n1):	
		for j in xrange(n2):
			Y[i:i+m1,j:j+m2] += X[i,j]*W 

	ret = Y[m1/2:-m1/2+1, m2/2:-m2/2+1]
	print "elapsed time:", (datetime.now() - t0) 
	return ret 

#10x faster + smaller output
def convolve2d(X, W):
	t0 = datetime.now()
	n1, n2 = X.shape 
	m1, m2 = W.shape 
	Y = np.zeros((n1 + m1-1, n2 + m2-1))

	for i in xrange(n1):	
		for j in xrange(n2):
			Y[i:i+m1,j:j+m2] += X[i,j]*W 

	ret = Y[m1:-m1+1, m2:-m2+1]
	print "elapsed time:", (datetime.now() - t0) 
	return ret 	


def main():
	img = mpimg.imread('lena.png')
	# plt.imshow(img)
	# plt.show()

	bw = img.mean(axis=2)
	# plt.imshow(bw, cmap='gray')
	# plt.show()

	W = np.zeros((20,20))

	for i in xrange(20):
		for j in xrange(20):
			dist = (i - 9.5)**2 + (j- 9.5)**2
			W[i,j] = np.exp(-dist/50)

	plt.imshow(W, cmap='gray')		
	plt.show()

	#mode='same' keeps output from the same size after convolution
	# out = convolve2d(bw, W, mode='same')
	out = convolve2d(bw, W)
	plt.imshow(out, cmap='gray')		
	plt.show()

	print "input shape", bw.shape
	print "output shape", out.shape

if __name__== '__main__':	
	main()