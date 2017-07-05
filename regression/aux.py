'''
Created on May 10, 2017

@author: Varela
'''
import numpy as np

import os 

def get_dataset(filename):
    fullpath = os.getcwd()
    fullpath = fullpath.replace('samples', 'datasets')
    fullpath = fullpath + '/' + filename
    return fullpath  

def polyfy(X, deg):
    #X(n,0) or X(n,1) is an np.array
    #produces Y = [1,X,X^2,...,X^deg] 
    
    sh =  X.shape 
    if ( len(sh)>2):  
        raise ValueError("X must be either (n,0) (n,1) shaped")
    elif (len(sh)==2):
        if  (sh[1]>1):
            raise ValueError("X must be either (n,0) (n,1) shaped")        
    elif (len(sh) ==1):        
        X  = X.reshape((sh[0],1)) # necessary for concatenation
        sh = X.shape
    o = np.ones(sh, dtype=np.float64)
    Y = np.concatenate((o,X), axis=1)
    
    for d in xrange(2,deg+1):
        XX = np.power(X,d)
        Y  = np.concatenate((Y,XX), axis=1)
    
    return Y     

def numpfy(X):       
    if ~isinstance(X, np.ndarray):
        X = np.array(X)
    return X

def intercept(X):
    # adds a column to the left of X filled with 1s
    # handles the intercept of a regression 
    X = numpfy(X)    
    return polyfy(X,1)

def nintercept(X):
    #Computes the size of the intercept array
    if len(X.shape) == 1:
        n = 1
    else:
        n = X.shape[-1]
    return n     
def to2dim(X):
    #Converts a possible 1-indexed ndarray to a 2-indexed ndarray
    #If X is not np.ndarray it also converts
    X = numpfy(X)
    sh = X.shape
    if len(sh)==1:
        X = X.reshape((sh[-1],1))
    return X    