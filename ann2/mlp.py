'''
Created on July 6, 2017

@author: Varela

Contains: gradient descent/mlp
'''

import numpy as np 

def forward(X, W1, b1, W2, b2):
	#SIGMOID
	Z  = 1 / (1 + np.exp(-( X.dot(W1) + b1)))

	#ReLU
	#Z = X.dot(W1) + b1 
	#Z[Z<0] = 0 

	#Softmax 
	A = Z.dot(W2) + b2 
	expA = np.exp(A)
	Y  = expA / expA.sum(axis=1, keepdims=True)
	return Y, Z 


def derivative_w2(Z, T, Y):
	return Z.T.dot(Y-T)

def derivative_b2(T, Y):
	return (Y-T).sum(axis=0)	

def derivative_w1(X, Z, T, Y, W2):
	return X.T.dot(( Y-T ).dot((W2.T))*(Z*(1-Z))) # for sigmoid
	#return X.T.dot(( Y-T ).dot((W2.T))*np.sign(Z))

def derivative_b1(Z, T, Y, W2):
	return ((Y-T).dot((W2.T))*(Z*(1-Z))).sum(axis=0)   # for sigmoid
	#return ((Y-T).dot(W2.T)*np.sign(Z)).sum(axis=0) # for ReLU