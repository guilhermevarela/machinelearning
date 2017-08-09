'''
Created on May 19, 2017

@author: Varela
'''

import numpy as np 
import matplotlib.pyplot as plt
import util as utl 

nclass = 500
# Build gaussian clouds
X1 = np.random.randn(nclass, 2) + np.array([0,-2])
X2 = np.random.randn(nclass, 2) + np.array([2, 2])
X3 = np.random.randn(nclass, 2) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])
Y = np.array([0]*nclass + [1]*nclass + [2]*nclass)

plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()

D = 2 # number of inputs
M = 3 # hidden layers
K = 3 # number of classes

W1  = np.random.randn(D, M)
b1 = np.random.randn(M)
W2  = np.random.randn(M, K)
b2 = np.random.randn(K)

def forward(X, W1, b1, W2, b2):
    Z = 1 / (1+ np.exp(-X.dot(W1) - b1) )
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y

def classification_rate(Y,P):    
    n_correct = 0 
    n_total = 0
    for i in xrange(len(Y)):
        n_total +=1
        if Y[i] == P[i]:
            n_correct +=1 
    return float(n_correct) / n_total        

P_Y_given_X = forward(X, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis=1)

# assert(len(P) == len(Y))
print "classification rate for random weights:",classification_rate(Y, P) 

Z = utl.fwd(X, W1, b1)
A = Z.dot(W2) + b2 
P_Y_given_X  = utl.softmax(A)
P = np.argmax(P_Y_given_X , axis=1)
print "classification rate for random weights (test):", utl.classification_rate(Y, P)

utl.fwdprop(X, W1, b1, W2, b2 )
print "classification rate for random weights (test-2):", utl.classification_rate(Y, P)

 