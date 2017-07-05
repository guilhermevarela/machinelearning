'''
Created on May 10, 2017

@author: Varela
'''

import numpy as np
import pandas as pd  
import regress as rg

from aux import polyfy, get_dataset




df = pd.read_csv(get_dataset('1dpoly.csv'), header=None)
X = np.array(df.iloc[:,0])
X = X.reshape((100,1))
XX = np.power(X,2)
# o = np.ones(X.shape, dtype=np.float64)
# 
# X =  np.concatenate((o,X), axis=1)
# X =  np.concatenate((X,XX), axis=1)

X = polyfy(X,2)
Y = np.array(df.iloc[:,1])
W, r2 = rg.lin(X, Y)

print W
print r2

#LAZY PROGRAMMER'S 
#load data 
# X = []
# Y = [] 
# for line in open('data_poly.csv'):
#     x, y = line.split(',')
#     x = float(x)
#     X.append([1 , x, x*x])
#     Y.append(float(y))
#turn X and Y into numpy arrays
# X = np.array(X)  
# Y = np.array(Y)

# plt.scatter(X[:,1], Y)
# plt.show()

#calculate weights
# w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T,Y))
# Yhat = np.dot(X, w)


# plt.scatter(X[:,1], Y)
# plt.plot(sorted(X[:,1]), sorted(Yhat))
# plt.show()

#compute r-squared
# d1 =  Y - Yhat 
# d2 =  Y - Y.mean()
# r2 = 1 - d1.dot(d1) / d2.dot(d2)
# print("r-squared",r2)


