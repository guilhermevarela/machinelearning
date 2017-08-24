'''
Created on Aug 20, 2017

@author: Varela

motivation: Multilayer perceptron for ecommerce data using scikit-learn
				
'''
import numpy as np 

from utils import get_ecommerce, sigmoid 

from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle


def main(): 
	X, Y  = get_ecommerce(user_action=None)

	X, Y = shuffle(X, Y)	

	#Split into train and test

	Ntrain = int(0.8*len(Y))
	Xtrain, Ytrain = X[:Ntrain,:], Y[:Ntrain]
	Xtest, Ytest   = X[:Ntrain,:], Y[:Ntrain]


	M = 10
	K = 4 
	# create the neural network
	model = MLPClassifier(hidden_layer_sizes=(Ntrain, M), max_iter=2000)

	#train ANN 
	model.fit(Xtrain, Ytrain)


	# print the train and test accuracy
	train_accuracy = model.score(Xtrain, Ytrain)
	test_accuracy = model.score(Xtest, Ytest)
	
	print "train accuracy:", train_accuracy, "test accuracy:", test_accuracy


if __name__ == '__main__':
	main()	