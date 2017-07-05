'''
Created on May 15, 2017

@author: Varela
'''


# the notes for this class can be found at: 
# https://deeplearningcourses.com/c/data-science-logistic-regression-in-python
# https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/6183984?start=0

import numpy as np
from util import sigmoid
from regress import  logisticl1
import matplotlib.pyplot as plt



n = 50 
d = 50
# uniformly distributed numbers between -5, +5
X = (np.random.random((n,d))-0.5)*10
# true weights - only the first 3 dimensions of X affect Y
true_w = np.array([1, 0.5, -0.5] + [0]*(d-3))
# generate Y - add noise with variance 0.5
Y = np.round(sigmoid(X.dot(true_w) + np.random.randn(n)*0.5))

w, costs = logisticl1(X,Y, 10, 5000, 0.001)

#LAZY PROGRAMMER'S
# costs = []
# w = np.random.randn(d) / np.sqrt(d)
# learning_rate = 0.001
# l1 = 3.0 
# for t in xrange(5000):
#     Yhat = sigmoid(X.dot(w))
#     delta = Yhat - Y 
#     w -= learning_rate*(X.T.dot(delta) + l1*np.sign(w))
#     
#     cost  = -(Y*np.log(Yhat) + (1-Y)*np.log(1-Yhat)).mean() + l1*np.abs(w).mean()
#     costs.append(cost)



plt.plot(costs)
plt.show()

plt.plot(true_w, label= 'true w')
plt.plot(w, label = 'map w')
plt.legend()
plt.show() 
