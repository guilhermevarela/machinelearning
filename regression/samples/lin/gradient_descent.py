'''
Created on May 12, 2017

@author: Varela
'''
# notes for this course can be found at:
# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# https://www.udemy.com/data-science-linear-regression-in-python

import numpy as np
import matplotlib.pyplot as plt
from regress import graddesc

if __name__ == '__main__':
    pass

N = 10
D = 3
X = np.zeros((N, D))
X[:,0] = 1 # bias term
X[:5,1] = 1
X[5:,2] = 1
Y = np.array([0]*5 + [1]*5)

# print X so you know what it looks like
print "X:", X

# won't work!
# w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

w, costs = graddesc(X,Y,1000, 0.001)

# plot the costs
plt.plot(costs)
plt.show()

print "final w:", w

Yhat = X.dot(w)

# plot prediction vs target
plt.plot(Yhat, label='prediction')
plt.plot(Y, label='target')
plt.legend()
plt.show()

