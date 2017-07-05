'''
Created on May 14, 2017

@author: Varela
'''
#logistic2.py
#https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/3963012?start=0
#logistic_vizualization.py
#https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/5288276?start=0
import numpy as np
from util import sigmoid, xentropy
import matplotlib.pyplot as plt
# if __name__ == "__main__":
    
n = 100 
d = 2
 
X = np.random.randn(n,d)
 
#Place a 2 bias on the first 0-49 rows
X[:50, :] = X[:50, :] - 2*np.ones((50, d))
X[:50, :] = X[50:, :] + 2*np.ones((50, d))
 
T = np.array([0]*50 + [1]*50)
 
ones  = np.array([[1]*n]).T # making arrays on numpy 2-dimensional
# ones  = np.array([[1]*N]).T 
Xb    = np.concatenate((ones,X), axis=1)
 
#randomly initialize the weights
w = np.random.randn(d+1)
 
#calculate the model output 
z = Xb.dot(w)
Y = sigmoid(z)
 
print xentropy(T, Y)
 
 
#Exact computation
#weights depend only on the means
# error happening 
w = np.array([0,4,4])
 
z = Xb.dot(w)
Y = sigmoid(z)
 
print xentropy(T, Y)

#vizualization
#y = -x
x_axis = np.linspace(-6,-6, 100)
y_axis = - x_axis
plt.plot(x_axis, y_axis)
plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha=0.5)
plt.show()


