'''
Created on Jul 05, 2017

@author: Varela

'''
import os 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression 

def get_mnistpath():
	cwdpath =  os.path.dirname(os.path.abspath(__file__)) 
	mnistpath  = cwdpath.replace('ann2','projects/mnist/')
	return mnistpath

def get_transformed_data():
	print "Reading in and transforming the data ..."

	datapath = get_mnistpath()
	mnistpath =  datapath + 'train.csv'
	#Change that link
	df = pd.read_csv(mnistpath)
	data = df.as_matrix().astype(np.float32)
	np.random.shuffle(data)

	X = data[:, 1:]
	mu = X.mean(axis=0)
	X = X - mu 			#Centralize data
	pca = PCA()
	Z = pca.fit_transform(X)
	Y = data[:, 0]

	return Z,  Y, pca, mu 


def get_normalized_data():
	print "Reading in and transforming the data ..."

	mnistpath =  get_mnistpath() + 'train.csv'
	#Change that link
	df = pd.read_csv(mnistpath)
	data = df.as_matrix().astype(np.float32)
	np.random.shuffle(data)

	X = data[:, 1:]
	mu = X.mean(axis=0)
	std = X.std(axis=0)
	np.place(std, std == 0, 1)
	
	#NORMALIZE DATA
	X = (X - mu) / std 			
	Y = data[:, 0]

	return X,  Y


def plot_cumulative_variance(pca):
	P = [] 
	for p in pca.explained_variance_ratio_:
		if len(P) == 0:
			P.append(p)
		else:
			P.append(p + P[-1])
  
	plt.plot(P)
	plt.show()
	return P

def forward(X, W,b):
	a = X.dot(W) + b 
	expa = np.exp(a)
	y = expa / expa.sum(axis=1, keepdims=True)
	return y 

def predict(p_y):
	return np.argmax(p_y, axis=1)

def error_rate(p_y, t):
	prediction = predict(p_y)
	return np.mean(prediction != t)

def cost(p_y, t):
	tot = t * np.log(p_y)
	return -tot.sum()

def gradW(t, y, X):
	return X.T.dot((t-y))

def gradb(t, y):
	return (t-y).sum(axis=0)	

def y2indicator(y):
	N = len(y)
	y = y.astype(np.int32)
	ind = np.zeros((N,10))
	for i in xrange(N):
		ind[i, y[i]] =1 
	return ind 

def benchmark_full():
	X, Y = get_normalized_data()

	print("Performing logistic regression...")

	Xtrain = X[:-1000,]
	Ytrain = Y[:-1000,]
	Xtest = X[-1000:,]
	Ytest = Y[-1000:,]


	# convert Ytrain and Ytest to (N x K) matrices of indicator variables
	#import code; code.interact(local=dict(globals(), **locals()))
	N, D = Xtrain.shape
	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind  = y2indicator(Ytest)

	W = np.random.randn(D, 10) / 28
	b = np.zeros(10)
	LL = [] 
	LLtest  = [] 
	CRtest  = [] 


	lr = 0.00004
	reg = 0.01 
	for i in range(500):
		p_y = forward(Xtrain, W, b)
		ll = cost(p_y, Ytrain_ind)
		LL.append(ll)

		p_y_test = forward(Xtest, W, b)
		lltest = cost(p_y_test, Ytest_ind)
		LLtest.append(lltest)


		err = error_rate(p_y_test, Ytest)
		CRtest.append(err)

		W+=  lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
		b+=  lr*(gradb(Ytrain_ind, p_y) - reg*b)
		if i % 10 ==0:
			print("Cost a iteration %d: %.6f" %(i,ll))
			print("Error rate:", err)

	p_y = forward(Xtest, W, b)				
	print("Final error rate:", error_rate(p_y, Ytest))
	iters = range(len(LL))
	plt.plot(iters, LL, iters, LLtest)
	plt.show()
	plt.plot(CRtest)
	plt.show()


def benchmark_pca():
	X, Y, _, _  = get_transformed_data()

	print("Performing logistic regression...")

	X = X[:,:300]
	# normalize X first
	mu  = X.mean(axis=0)
	std = X.std(axis=0)
	X = (X-mu) / std 


	# convert Ytrain and Ytest to (N x K) matrices of indicator variables
	#import code; code.interact(local=dict(globals(), **locals()))

	print("Performing logistic regression...")	
	
	Xtrain = X[:-1000,]
	Ytrain = Y[:-1000]
	Xtest = X[-1000:,]
	Ytest = Y[-1000:]
	

	N, D = Xtrain.shape
	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)
	

	W = np.random.randn(D, 10) / 28
	b = np.zeros(10)
	LL = [] 
	LLtest  = [] 
	CRtest  = [] 


	lr = 0.0001
	reg = 0.01 
	for i in range(200):
		p_y = forward(Xtrain, W, b)
		ll = cost(p_y, Ytrain_ind)
		LL.append(ll)

		p_y_test = forward(Xtest, W, b)
		lltest = cost(p_y_test, Ytest_ind)
		LLtest.append(lltest)


		err = error_rate(p_y_test, Ytest)
		CRtest.append(err)

		W+=  lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
		b+=  lr*(gradb(Ytrain_ind, p_y) - reg*b)
		if i % 10 ==0:
			print("Cost a iteration %d: %.6f" %(i,ll))
			print("Error rate:", err)

	p_y = forward(Xtest, W, b)				
	print("Final error rate:", error_rate(p_y, Ytest))
	iters = range(len(LL))
	plt.plot(iters, LL, iters, LLtest)
	plt.show()
	plt.plot(CRtest)
	plt.show()	


if __name__ == '__main__':
	benchmark_pca()	
	#benchmark_full()	