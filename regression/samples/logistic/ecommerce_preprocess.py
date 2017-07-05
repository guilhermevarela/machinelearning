'''
Created on May 14, 2017

@author: Varela
'''
import pandas as pd 
import numpy as np
import os  
#https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/5286974?start=0    
def get_path():
    ecommerce_path = os.getcwd()
    ecommerce_path = ecommerce_path.replace('samples', 'datasets')
    ecommerce_path = ecommerce_path + '/' + 'ecommerce_data.csv'
    return ecommerce_path  
def get_data():    
    df = pd.read_csv(get_path())
    data =  df.as_matrix()
    X = data[:, :-1]
    Y = data[:, -1]
    
    #handling numerical data
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()
    
    #handling categorical data
    #time of the day may assume 4 values -> extend X by 3
    n, d = X.shape
    X2 = np.zeros((n,d+3))
    X2[:,0:(d-1)] = X[:,0:(d-1)]
    
    #hotcoding for the 4-categories time of the day may assume
    for n in xrange(n):
        t = int(X[n,d-1])
        X2[n,t+d-1]=1
# another way which is throwing error so far        
#     Z = np.zeros((n,4))
#     Z[np.arange(n), X[:,d-1].astype(np.int32)]=1
#     X2[:,-4:]=Z    
#     assert(np.abs(X2[:,-4:]-Z).sum() < 10e-10)
    return X2, Y

def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <=1]
    Y2 = Y[Y <=1]
    return X2, Y2
 
print get_binary_data()

    