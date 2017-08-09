'''
Created on May 16, 2017

@author: Varela
'''

if __name__ == '__main__':
    pass

import numpy as np
# a is the activation function on the last layer 
a = np.random.randn(5)

#numbers become positives
expa = np.exp(a)

#devide each element by the array modulus.: normalizing
ans = expa / expa.sum()

print "We are probabilities", ans
print "We sum to 1", ans.sum()
print "argmax",  np.argmax(expa)  


n = 100
d = 5

# A = np.random.randn((n,d)) / np.sqrt(n+d)
A = np.random.randn(n,d) 

expA = np.exp(A) 

ans = expA / np.sum(expA,axis=1,keepdims=True)
print "We are probabilities", ans[:2,:]
print "We sum to 1", np.sum(ans[:2,:],axis=1,keepdims=True)
print "argmax",  np.argmax(ans[:2,:]) 
