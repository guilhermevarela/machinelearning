'''
Created on May 10, 2017

@author: Varela
'''
# from  mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
import pandas as pd  
import regress as rg
from aux import get_dataset

df = pd.read_csv(get_dataset('2d.csv'), header=None)
xcol= [0,1]
ycol = 2
X = df.iloc[:,xcol]
Y = df.iloc[:,ycol]

W, r2 = rg.lin(X, Y)

print W
print r2

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:,0],X[:,1], Y)
# plt.show()


#lazyprogrammer
#load data 
# X = []
# Y = [] 
# for line in open(filename):
#     x1, x2, y = line.split(',')
#     X.append([float(x1), float(x2), 1])
#     Y.append(float(y))

#turn X and Y into numpy arrays
# X = np.array(X)  
# Y = np.array(Y)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:,0],X[:,1], Y)
# plt.show()

#calculate weights
# w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T,Y))
# Yhat = np.dot(X, w)

#compute r-squared
# d1 =  Y - Yhat 
# d2 =  Y - Y.mean()
# 
# print("r-squared", 1 - d1.dot(d1) / d2.dot(d2))


# B, rsq = lin(XX,YY, True)
# print(w, B)
# print(1 - d1.dot(d1) / d2.dot(d2), rsq, r2(B,X,Y))
