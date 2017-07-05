'''
Created on May 15, 2017

@author: Varela
'''

import numpy as np
import matplotlib.pyplot as plt
from util import sigmoid, xentropy
from regress import logisticl2

n = 1000
d = 2

R_inner = 5 
R_outer = 10

R1 = np.random.randn(n/2) + R_inner
theta = 2*np.pi*np.random.random(n/2)
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1*np.sin(theta)]]).T

R2 = np.random.randn(n/2) + R_outer
theta = 2*np.pi*np.random.random(n/2)
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2*np.sin(theta)]]).T

X = np.concatenate([X_inner, X_outer])
T = np.array([0]*(n/2) + [1]*(n/2))


plt.scatter(X[:,0], X[:,1], c=T)
plt.show()

r = np.zeros((n,1))
for i in range(n):
    r[i] = np.sqrt(X[i,:].dot(X[i,:]))

ones = np.array([[1]*n]).T
print ones.shape     
Xb = np.concatenate((ones, r, X), axis=1)

w = np.random.rand(d+2)



w, error  = logisticl2(Xb, T, 0.01, 5000, 0.0001)
z = Xb.dot(w)
Y = sigmoid(z) 

plt.plot(error)
plt.title("Cross-entropy")
plt.show()

print "Final w:", w
print "Final classification rate", 1  - np.abs(T-np.round(Y)).sum() / n


