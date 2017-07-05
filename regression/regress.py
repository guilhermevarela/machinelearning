'''
Created on Apr 21, 2017

@author: guilhermevarela
'''
import numpy as np
from aux import numpfy, intercept, nintercept
from util import sigmoid, xentropy



def lin(X, Y, fill=False):
    
    X = numpfy(X)
    Y = numpfy(Y) 
    
    
    if fill:        
        X = intercept(X)
     
    d = nintercept(X)    
    if d==1:
        W   = linear1(X, Y)
        rsq = residual1d(W, X, Y)

    else:
        W   =linearn(X, Y)
        rsq =residualn(W, X, Y)        
    
    return W, rsq

def linl2(X,Y,l2, fill=False):
    X = numpfy(X)
    Y = numpfy(Y) 
    
    if fill:        
        X = intercept(X)
        
    W     =linearl2(X, Y, l2)
    resid = residualn(W, X, Y)
    
    return W, resid 

def linl1(X,Y,l1, T=1000, learning_rate=10e-2):
    n,d = X.shape
    mse = []
    w = np.random.randn(d) / np.sqrt(d)
    
    l1 = 10
    for _ in xrange(T):
        Yhat = X.dot(w)
        delta = Yhat - Y
        w = w - learning_rate*(X.T.dot(delta) + l1*np.sign(w))
    
        mse.append(delta.dot(delta) / n)
     
    return w, mse

def graddesc(X,Y, T=1000, learning_rate=10e-2):
    #Gradient descent for linear regression ( analytic method )
    #It garantees convergence even when a subset of columns of X are multiples 
    # won't work!
    # w = np.linalg.solve(X.T.dot(X), X.T.dot(Y)) 
    # https://deeplearningcourses.com/c/data-science-linear-regression-in-python
    # https://www.udemy.com/data-science-linear-regression-in-python     
    n,d  = X.shape
    
    
    # let's try gradient descent
    mse = [] # keep track of squared error cost
    w = np.random.randn(d) / np.sqrt(d) # randomly initialize w
    learning_rate = 0.001
    for _ in xrange(T):
        # update w
        Yhat = X.dot(w)
        delta = Yhat - Y
        w = w - learning_rate*X.T.dot(delta)

        # find and store the cost        
        mse.append(delta.dot(delta) / n)
    return w, mse    

def logisticl2(X,Y,l2, T=1000, learning_rate=10e-2):
    #Gradient descent for logistic regression adjusted for l2 regularization making weights smaller
    #It garantees convergence even when a subset of columns of X are multiples 
    # https://deeplearningcourses.com/c/data-science-linear-regression-in-python
    # https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/3963018?start=0     
            
    w    = np.random.randn(X.shape[1])
        
    xe   = []
    
    for _ in xrange(T):
            # recalculate Y
        Yh = sigmoid(X.dot(w))        
        delta = Y - Yh
        
        # gradient descent weight update
        w += learning_rate*(X.T.dot(delta)-l2*w)
    
        xe.append( xentropy(Y, Yh) )    
    return w, xe    

def logisticl1(X,Y,l1, T=1000, learning_rate=10e-2):
    #Gradient descent for logistic regression adjusted for l2 regularization making weights smaller
    #It garantees convergence even when a subset of columns of X are multiples
    #https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/6183984?start=0 
    # https://deeplearningcourses.com/c/data-science-linear-regression-in-python
    # https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/3963018?start=0     
            
    w    = np.random.randn(X.shape[1])
        
    xe   = []
    
    for _ in xrange(T):
        Yh   = sigmoid(X.dot(w))
        delta = Yh - Y
        # gradient descent weight update
        w -= learning_rate*(X.T.dot(delta)+l1*np.sign(w))
        
        xe.append( xentropy(Y, Yh) + l1*np.abs(w).mean())    
    return w, xe    


def logistic_graddesc(X,Y, T=1000, learning_rate=10e-2):
    #Gradient descent for logistic regression ( analytic method )
    #It garantees convergence even when a subset of columns of X are multiples 
    # https://deeplearningcourses.com/c/data-science-linear-regression-in-python
    # https://www.udemy.com/data-science-linear-regression-in-python     
            
    w    = np.random.randn(X.shape[1])
    Yh   = sigmoid(X.dot(w))    
    xe   = []
    
    for _ in xrange(T):
        delta = Y - Yh
        # gradient descent weight update
        w += learning_rate*X.T.dot(delta)
        # recalculate Y
        Yh = sigmoid(X.dot(w))
        xe.append( xentropy(Y, Yh) )    
    return w, xe    
      
    
    
def linlog(X, Y):
    raise NotImplementedError("linlog(X, Y) not yet implemented")
    w = []
    rsq=0
    return w, rsq

def r2(W, X, Y):
    
    X = numpfy(X)
    Y = numpfy(Y)
    
    dim   = np.shape(X)
    if (len(dim)==1):
        rsq = residual1d(W, X, Y)

    else:
        rsq =residualn(W, X, Y)        
    
    return rsq
    
    
def linear1(X,Y):
    #X,Y are both "numpfyed" arrays
    #Computes simple linear regression such that Yh = W*X + W[0]
    
    mux = X.mean()
    muy = Y.mean() 
    sumx = X.sum()  
 
    #cross product computations
    xx = (X.dot(X) - mux * sumx)
    xy = (Y.dot(X) - muy * sumx)
    a = xy  / xx
    b = muy - a * mux
    return [b,a];

def linearn(X,Y):
    #X,Y are both "numpfyed" arrays
    #Computes linear regression such that Yh = W*X + W[0]
    W = np.linalg.solve(np.dot(X.T, X), np.dot(X.T,Y))
    return W

def linearl2(X,Y, l2):
    
    if len(X.shape) == 1:
        n = 1
    else:
        n = X.shape[1]
    W = np.linalg.solve(l2*np.eye(n) + np.dot(X.T, X), np.dot(X.T,Y))
    return W    



def residual1d(W, X, Y):
    muy = Y.mean()
    Yh = W[1]*X + W[0]
    d1 = Y - muy
    d2 = Y - Yh
     
    rsq  = 1 - d2.dot(d2)/d1.dot(d1)
    return rsq 


def residualn(W, X, Y):
    muy = Y.mean()
    Yh = np.dot(X,W) 
    d1 = Y - muy
    d2 = Y - Yh
     
    rsq  = 1 - d2.dot(d2)/d1.dot(d1)
    return rsq 