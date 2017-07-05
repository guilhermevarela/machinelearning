'''
Created on May 15, 2017

@author: Varela
'''
#lecture Section 4 lecture 27
#https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/3963028?start=0

import numpy as np
import matplotlib.pyplot  as plt
from util.util import sigmoid
from regression.regress import logisticl2
n  = 4
d  = 2

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],                 
    ])

T = np.array([0, 1, 1, 0])
ones = np.array([[1]*n]).T

# plot X-or problem
# plt.scatter(X[:,0], X[:,1], c=T)
# plt.show()

ones = np.array([[1]*n]).T
xy   = np.matrix(X[:,0] * X[:,1]).T
Xb   = np.array(np.concatenate((ones,xy,X), axis=1))

#randomly initialize the weights
w = np.random.randn(d+2)

#calculate the model output
w, error  = logisticl2(Xb, T, 0.01, 10000, 0.001)

z = Xb.dot(w)
Y = sigmoid(z) 


plt.plot(error)
plt.title("Cross-entropy")
plt.show()

print "Final w:", w
print "Final classification rate", 1  - np.abs(T-np.round(Y)).sum() / n



