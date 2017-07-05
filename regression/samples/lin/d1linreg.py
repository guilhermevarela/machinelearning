'''
Created on May 10, 2017

@author: Varela
'''


import pandas as pd  
import regress as rg
from aux import get_dataset
# import matplotlib.pyplot as plt


df = pd.read_csv(get_dataset('1d.csv'), header=None)
X = df.iloc[:,0]
Y = df.iloc[:,1]

W, r2 = rg.lin(X, Y)

print W
print r2

#plot the data to see what's look like
# plt.scatter(X, Y)
# plt.show()
 

 



