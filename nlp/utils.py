'''
Created on Aug 16, 2017

@author: Varela

'''

import numpy as np 
import pandas as pd 

def error_rate(T,Y):
	return np.mean(np.round(Y)!=T)

def classification_rate(T,Y):
	return np.mean(Y==T)	

# calculate the cross-entropy error
def cross_entropy(T, Y):
	E = 0

	for i in xrange(T.shape[0]):
		# yi = min(max(Y[i],1e-8), 1-1e-8)		
		if T[i] == 1:
			
			E -= np.log(Y[i])
		else:			
			E -= np.log(1.0 - Y[i])
	return E
	
def sigmoid(Z):
	return 1.0/(1.0 + np.exp(-Z))

def softmax(A):
	expA = np.exp(A)
	return expA / expA.sum(axis=1, keepdims=True)

def forward(W, X):
	return sigmoid(X.dot(W))

def y2indicator(y, K):
	#one-hot_codefy :)
	N = len(y)
	ind = np.zeros((N,K))
	for i in xrange(N):
		ind[i, y[i]] = 1
	return ind 

def get_iris():
	X = []; Y =[]

	df = pd.read_csv('./datasets/iris/iris.data', sep=',')
	X=  df.values[:,:-1]
	Y=  df.values[:,-1]

	return X, Y 

def get_spambase():
	X = []; Y =[]

	df = pd.read_csv('./datasets/spambase/spambase.data', sep=',')
	X=  df.values[:,:-1]
	Y=  df.values[:,-1]

	return X, Y 	

def get_ecommerce(user_action=1):
	X = []; Y =[]
	if not( user_action in [None,1,2,3]): 
		raise ValueError( 'value to user_action - %d invalid' % (user_action))

	df = pd.read_csv('./datasets/ecommerce/ecommerce_data.csv', sep=',', header=0)
	X=  df.values[:,:-1]
	if user_action:
		Y=  np.array(df.values[:,-1] == user_action, dtype=np.int32)
	else:
		Y=  np.array(df.values[:,-1], dtype=np.int32)
				
	return X, Y 

def get_facialexpression(balance_ones=True):
	#images are 48x48 = 2304 size vectors
	#N = 35887
	Y = [] 
	X = []
	first= True 
	for line in open('./datasets/facial_recognition/fer2013.csv'):
		if first: 
			first = False 
		else:
			row = line.split(',')
			Y.append(int(row[0]))
			X.append([int(p) for p in row[1].split()])

	#	1/255 =? sigmoid/ tanh is more sensitive in interval -1..+1	
	X, Y = np.array(X)/ 255.0, np.array(Y)		

	if balance_ones:
		#class 1 is severely underrepresented
		X0, Y0 = X[Y!=1, :], Y[Y!=1] 
		X1 	= X[Y==1,:]
		X1 	= np.repeat(X1, 9, axis=0)
		X 	= np.vstack([X0, X1])
		Y 	= np.concatenate((Y0, [1]*len(X1)))

	return X, Y
	



