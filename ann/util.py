'''
Created on May 14, 2017

@author: Varela
'''
import numpy as np

# cross entropy (1): https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/3963012?start=0
# cross entropy (2): https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/5286984?start=0
#fnc to util: https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/5217324?start=0

#return as np.float32 for theano and tensor flow
def init_weight_and_bias(n, m):    
    # Initializes weight w(n,m)~N(0,1/(n+m)) and bias b~zeros((m,))
    # (n,m) fan-in/ fan-out dimensions .: (input size, output size)
    # (W,b) (W(n,m), b(m,))
    
    W = np.random.randn(n, m) / np.sqrt(n + m)
    b = np.zeros(m)
    return W.astype(np.float32), b.astype(np.float32) 

def init_filter(shape, poolz):
    # For the use of convolutional neural networks
    # shape ~ (A,B,C,D)
    w  = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:])) + shape[0]*np.prod(shape[2:] / np.prod(poolz))
    return w.astype(np.float32)

def relu(x):
    return x * (x > 0)
    
def sigmoid(z):
    return 1/(1+np.exp(-z))

def fwd(X, W, b):
    return sigmoid(X.dot(W) +b)

def classification_rate(Y,P):
    return np.mean( Y == P)

def xentropy(T,Y):
    eps= np.array([10e-3]*T.shape[0])
    Yb  = np.maximum(Y,eps)
    Y1b = np.maximum(1-Y,eps)
    return -np.mean(T*np.log(Yb)+(1-T)*np.log(Y1b))

def sigmoid_cost(T,Y):
    return -np.sum(T*np.log(Y)+(1-T)*np.log(Y))

def cost(T,Y):
    return -(T*np.log(Y)).sum()

def costs2(T,Y):
    # same as cost(), just uses T to index Y
    N = len(T)
    return -np.log(Y[np.arange(N),T]).sum()
    
def softmax(z):
    expz = np.exp(z)
    return expz / expz.sum(axis=1, keepdims=True)

def fwdprop(X, W1, b1, W2, b2 ):
# Computes forward propagation with settings
# structure:1 hidden layer
# activation fuction: sigmoid
    Z = fwd(X, W1, b1) 
    A = Z.dot(W2) + b2       # ouputs from hidden layer 
    P_Y_given_X = softmax(A) # binds all hidden layer outputs together 
    Y = np.argmax(P_Y_given_X, axis=1)
    return Y

#LAZY PROGRAMMER'S
def cross_entropy(T,Y):
    xe = 0
    n = T.shape[0]
    for i in xrange(n):
        if T[i] == 1:
            xe -= np.log(Y[i])
        else:  
            xe -= np.log(1-Y[i])
    return xe        
         
    
    
    
    