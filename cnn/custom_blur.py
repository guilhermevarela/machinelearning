'''
Created on Ago 08, 2017

@author: Varela

For the class Data Science: Deep Learning convolutional neural networkds on theano and tensorflow
lecture #8; Using convolution to apply gaussian blur to lena.png 
course url 1: https://deeplearningcourses.com/c/deep-learning-convolutional-neural-networks-theano-tensorflow
course url 2: https://udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow
lecture url: https://www.udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow/learn/v4/t/lecture/4847750?start=0

'''

import numpy as np 
from scipy.signal import convolve2d 

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

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

#mode='same' keeps output from the same size
out = convolve2d(bw, W, mode='same')
plt.imshow(out, cmap='gray')		
plt.show()

print "input shape", bw.shape
print "output shape", out.shape