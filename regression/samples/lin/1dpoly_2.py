'''
Created on May 10, 2017

@author: Varela
'''

import pandas as pd
import numpy as np
import regression.regress as reg 
from regression.aux import polyfy, get_dataset

df = pd.read_csv(get_dataset('1d.csv'), header=None)
X = np.array(df.iloc[:,0])
Y = np.array(df.iloc[:,1])
WW  = np.polyfit(X, Y, 2)
W   = WW[::-1] # itens come from highest order to lowest
XX  = polyfy(X, 2)
r2  = reg.r2(W, XX, Y)

print W
print r2
