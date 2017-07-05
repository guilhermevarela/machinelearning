'''
Created on May 14, 2017

@author: Varela
'''
'''
Created on May 11, 2017

@author: Varela
'''
#https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/3963272?start=0
import numpy as np
from util import sigmoid
# if __name__ == "__main__":

N = 100
D = 2

X = np.random.randn(N,D)
ones  = np.array([[1]*N]).T # making arrays on numpy 2-dimensional
Xb    = np.concatenate((ones,X), axis=1)
w = np.random.randn(D+1)
                    
z = Xb.dot(w) 

print sigmoid(z)


