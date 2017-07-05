'''
Created on May 14, 2017

@author: Varela
'''

#https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/5286980?start=0

import numpy as np
import pandas as pd

from ecommerce_preprocess import get_binary_data
from util import sigmoid, fwd, classification_rate 

#randomly predicts data
X, Y = get_binary_data()

D = X.shape[1]
W = np.random.randn(D)
b = 0

P_Y_given_X = fwd(X,W, b)
predictions = np.round(P_Y_given_X)

print "Score:", classification_rate(Y, predictions)

