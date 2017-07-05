'''
Created on May 21, 2017

@author: Varela
'''
#mlp file
#https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow/learn/v4/t/lecture/4626812?start=0
import numpy as np 

def forward(X, W1, b1, W2, b2):
    #Z = 1/(1+np.exp(-( X.dot(W1) + b)))
    
    #rectifier
    Z = X.dot(W1) + b1
    Z[Z <0] = 0 
    #print "Z:",Z
    
    A = Z.dot(W2) + b2
    expA = np.exp(A) 
    Y = expA / expA.sum(axis=1, keepdims=True)
    
    return Y, Z  

def derivative_w2(Z, T, Y):
    return Z.T.dot(Y - T)

def derivative_b2(Z, T, Y):
    return (Y - T).sum(axis=0)

def derivative_w1(X, Z, T, Y, W2):
    return X.T.dot( (Y-T).dot(W2.T)*Z*(1-Z) ) # for tanh
    #return X.T.dot( (Y-T).dot(W2.T)  + np.sign(Z) ) # for relu

def derivative_b1(Z, T, Y, W2):
    return ((Y-T).T.dot(W2.T)*(Z*(1-Z))).sum(axis=0) # for tanh
    #return ((Y - T).dot(W2.T) * np.sign(Z)).sum(axis=0) # for relu

