'''
Created on Aug 16, 2017

@author: Varela

'''

import numpy as np 
import pandas as pd 

from sklearn.utils import shuffle

def get_mnist():
	#MNIST data:
	#column 0 	is labels
	#column 1-785 is data with values 0..255
	#total csv: (42000, 1, 28, 28)
	train = pd.read_csv('../projects/mnist/train.csv').as_matrix().astype(np.float32)

	train = shuffle(train)


	Xtrain = train[:-1000,1:] / 255 
	Ytrain = train[:-1000,0].astype(np.int32)

	Xtest = train[-1000:,1:] / 255 
	Ytest = train[-1000:,0].astype(np.int32)
	return Xtrain, Ytrain, Xtest, Ytest

	



