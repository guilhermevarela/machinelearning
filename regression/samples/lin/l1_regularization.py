'''
Created on May 11, 2017

@author: Varela
'''

import numpy as np 
import matplotlib.pyplot as plt 
from regress import linl1 
N = 50
D = 50

X = (np.random.random((N,D))-0.5)*10

true_w = np.array([1,0.5,-0.5] + [0]*(D-3))

Y = X.dot(true_w) + np.random.randn(N)*0.5
#LAZY'S PROGRAMMER
# costs = []
# w = np.random.randn(D) / np.sqrt(D)
# learning_rate = 0.001
# l1 = 10
# for t in xrange(500):
#     Yhat = X.dot(w)
#     delta = Yhat - Y
#     w = w - learning_rate*(X.T.dot(delta) + l1*np.sign(w))
#     
#     mse = delta.dot(delta) / N 
#     costs.append(mse)
#     
w, costs = linl1(X,Y,10,500,1e-3)
plt.plot(costs)    
plt.show()


print "Final w .:", w

plt.plot(true_w, label= 'true w')
plt.plot(w, label = 'map w')
plt.legend()
plt.show() 