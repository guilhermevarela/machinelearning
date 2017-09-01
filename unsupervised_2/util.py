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


def error_rate(T,Y):
	return np.mean(Y != T)

def relu(A):
	return A * (A>0) 
	
def init_weight(Mi, Mo):
	return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)



def get_book_titles():
	titles = [line.rstrip() for line in open('../projects/book_titles/all_book_titles.txt')]
	return titles

def get_stopwords(): 
	stopwords = set(line.rstrip() for line in open('../projects/book_titles/stopwords.txt'))
	stopwords = stopwords.union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
		'third', 'second', 'fourth', })

	return stopwords

