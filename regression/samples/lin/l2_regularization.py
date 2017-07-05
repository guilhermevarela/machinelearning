'''
Created on May 11, 2017

@author: Varela

'''
'''
Created on May 10, 2017

@author: Varela
'''
#https://www.udemy.com/data-science-linear-regression-in-python/learn/v4/t/lecture/6183990?start=0 
#performs L2 regularization or Ridge regularization
import numpy as np
import pandas as pd  
from regress import lin, linl2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pass 
 
N =50
X = np.linspace(0, 10, N)
Y = 0.5*X  + np.random.randn(N)
Y[-1] += 30 
Y[-2] += 30

plt.scatter(X,Y)
plt.show()
#Why??
X = np.vstack([np.ones(N), X]).T
# XX = np.concatenate((np.ones(N),np.array([N])))
wml, residml = lin(X,Y)
Yhat_ml = X.dot(wml)





print wml  
print residml  

l2 = 1000
wmap, residmap = linl2(X,Y,l2)
Yhat_map = X.dot(wmap)

plt.scatter(X[:,1],Y)
plt.plot(X[:,1],Yhat_ml, label=("ml r2=%.4f" % (residml)))
plt.plot(X[:,1],Yhat_map, label=("map r2=%.4f" % (residmap)))
plt.legend()
plt.show()

print wmap   
print residmap 



