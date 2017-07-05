'''
Created on May 14, 2017

@author: Varela
'''

# the notes for this class can be found at: 
# https://deeplearningcourses.com/c/data-science-logistic-regression-in-python
# https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/3963016?start=0

import numpy as np 
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from ecommerce_preprocess import get_binary_data
from util import sigmoid, fwd, classification_rate, xentropy

X, Y =  get_binary_data()
X, Y =  shuffle(X, Y)

Xtrain = X[:-100]
Ytrain = Y[:-100]

Xtest  = X[-100:]
Ytest  = Y[-100:]

d = X.shape[1]
w = np.random.randn(d)
b=0 

train_costs = []
test_costs = []
learning_rate = 0.001

for i in xrange(10000):
    pYtrain = fwd(Xtrain, w, b)
    pYtest  = fwd(Xtest, w, b)
    
    ctrain =  xentropy(Ytrain, pYtrain)
    ctest = xentropy(Ytest, pYtest)
    
    train_costs.append(ctrain)
    test_costs.append(ctest)
    
    w -=learning_rate*Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate*(pYtrain - Ytrain).sum()
    if i % 1000 ==0:
        print i, ctrain, ctest, xentropy(Ytrain, pYtrain),xentropy(Ytest, pYtest) 
        
    
print "Final train classification_rate", classification_rate(Ytrain, np.round(pYtrain))
print "Final train classification_rate", classification_rate(Ytest, np.round(pYtest))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label = 'test cost')
plt.legend([legend1, legend2])        
plt.show()
     

