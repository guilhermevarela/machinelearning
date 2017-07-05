
import numpy as np
import pandas as pd  
from aux import get_dataset
from regress import lin 

# import re 
# import matplotlib.pyplot as plt

df = pd.read_table(get_dataset('moore.csv'), header=None)


# \[(.*)\] matches anything enclosed in brackets
#  [^\d]+ matches anything which is not a decimal
Y = df.loc[:,1].str.replace(r'\[(.*)\]','')
Y = Y.str.replace(r'[^\d]+', '').astype('int')
X = df.loc[:,2].str.replace(r'\[(.*)\]','')
X = X.replace(r'[^\d]+', '').astype('int')

# plt.scatter(X, Y)
# plt.show()
# 
# Y = np.log(Y)
# plt.scatter(X, Y)
# plt.show()

Y = np.log(Y)
W, r2 = lin(X,Y)
print W 
print r2


 
#LAZY PROGRAMMER SOLUTION
# import re 
# import numpy as np
# import matplotlib.pyplot as plt
# 
# X = []
# Y = []
#  
# non_decimal = re.compile(r'[^\d]+')
#  
# for line in open('../src/datasets/moore.csv'):
#     r = line.split('\t')
#      
#     x = int(non_decimal.sub('',r[2].split('[')[0]))
#     y = int(non_decimal.sub('',r[1].split('[')[0]))
#     X.append(x)
#     Y.append(y)
#      
# X = np.array(X)
# Y = np.array(Y) 
#      
# plt.scatter(X, Y)
# plt.show()
#  
# Y = np.log(Y)
# plt.scatter(X, Y)
# plt.show()
#  
 
# denominator  = X.dot(X) - X.mean() * X.sum() 
# a = ( X.dot(Y) - Y.mean()* X.sum() ) / denominator 
# b = ( Y.mean() * X.dot(X) - X.mean()* X.dot(Y) ) / denominator
#  
# Yhat = a*X  + b 
# plt.scatter(X, Y)
# plt.plot(X, Yhat)
# plt.show()
#  
 
# d1 = Y - Yhat 
# d2 = Y - Y.mean() 
# r2 = 1 - d1.dot(d1) / d2.dot(d2)
# print("a:", a,"b:", b)
# print("the r-squared is:", r2)
  
#   
#  
#  
# 
# #log(tc) = a*year + b
# #tc= exp(b) * exp(a * year)
# #2*tc = 2*exp(b)*exp(a *year) = exp(ln(2)) *exp(b)*exp(a*year)
# #     = exp(b) * exp(a * year1 + ln2)
# # exp(b)*exp(a*year2) =  exp(b) * exp(a * year1 + ln2)
# #a*year2 =a*year1 + ln2
# #year2 = year1 + ln2/a
# print("time to double:", np.log(2)/a, "years")
 