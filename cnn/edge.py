'''
Created on Ago 08, 2017

@author: Varela

For the class Data Science: Deep Learning convolutional neural networkds on theano and tensorflow
lecture #9; Using convolution to find edges on lena.png 
course url: https://udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow
lecture url: https://www.udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow/learn/v4/t/lecture/4847754?start=0

'''

import numpy as np 
from scipy.signal import convolve2d 

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

img = mpimg.imread('lena.png')
bw  = img.mean(axis=2)

# Sobel operator - approximate gradient in Y dir
Hx = np.array([
	[ -1,  0,  1],
	[ -2,  0,  2],
	[ -1,  0,  1]
], dtype=np.float32)

Gx = convolve2d(bw, Hx)
plt.imshow(Gx , cmap='gray')
plt.show()


# Sobel operator - approximate gradient in Y dir
Hy = np.array([
	[-1., -2., -1.],
	[ 0.,  0.,  0.],
	[ 1.,  2.,  1.]
])

Gy = convolve2d(bw, Hy)
plt.imshow(Gy , cmap='gray')
plt.show()

#Gradient magnetude
G = np.sqrt(Gx*Gx + Gy*Gy)
plt.imshow(G, cmap='gray')
plt.show()

#The gradient's direction
theta = np.arctan2(Gy, Gx)
plt.imshow(theta, cmap='gray')
plt.show()