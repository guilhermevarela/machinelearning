'''
Created on May 15, 2017

@author: Varela
'''


# the notes for this class can be found at: 
# https://deeplearningcourses.com/c/data-science-logistic-regression-in-python
# https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/3963018?start=0

import numpy as np
from util import sigmoid, xentropy
from regress import logistic_graddesc, logisticl2
import matplotlib.pyplot as plt



n = 100 
d = 2

X = np.random.randn(n,d)


# center the first 50 points at (-2,-2)
X[:50, :] = X[:50, :] - 2*np.ones((50, d))
# center the last 50 points at (2, 2)
X[50:, :] = X[50:, :] + 2*np.ones((50, d))

T = np.array([0]*50 + [1]*50)


ones  = np.ones((n,1)) 
Xb    = np.concatenate((ones,X), axis=1)

#randomly initialize the weights
w = np.random.randn(d+1)

#calculate the model output 
z = Xb.dot(w)

Y = sigmoid(z)

print xentropy(T, Y)

w,  xe   = logistic_graddesc(Xb, T, 1000,0.01)
wl2,xel2 = logisticl2(Xb,Y,10, T=1000, learning_rate=10e-2)

print "Final w:", w, xe[-1]
print "Final wl2:", wl2, xel2[-1]  



# plot the data and separating line

x_axis = np.linspace(-6, 6, 100)
y_axis = -(w[0] + x_axis*w[1]) / w[2]
y_axisl2 = -(wl2[0] + x_axis*wl2[1]) / wl2[2]
# plt.plot(x_axis, y_axis)
# plt.show()

plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha=0.5)
legend1, = plt.plot(x_axis,y_axis, label='gradient descend')
legend2, = plt.plot(x_axis,y_axisl2, label ='gradient descend l2')
plt.legend([legend1, legend2])        
plt.show()

legend1, = plt.plot(xe, label='cross entropy')
legend2, = plt.plot(xel2, label ='cross entropy l2')
plt.legend([legend1, legend2])        
plt.show()

