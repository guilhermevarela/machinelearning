'''
Created on Jul 07, 2017

@author: Varela

'''

import theano.tensor as T
from theano_ann import ANN
from util import get_spiral
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import numpy as np



def random_search():
	max_iter = 30
	X, Y = get_spiral()

	# plt.scatter(X[:,0],X[:,1])
	# plt.show()
	X, Y = shuffle(X,Y)
	Ntrain  =  int(0.7*len(X))
	Xtrain, Ytrain = X[:Ntrain,:], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:,:], Y[Ntrain:]


	#Starting parameters
	M = 20
	nHidden = 2
	log_lr = -4
	log_l2 = -2

	# best_M = M 
	# best_nHidden = nHidden
	# best_log_lr  = log_lr
	# best_log_l2  = log_l2

	#LOOP thrugh all possible hyperparameter settings
	best_validation_rate = 0 
	best_hls 	= None 
	best_lr 	= None
	best_l2  	=	None 
	for i in xrange(max_iter):
		#PARAMETER SPACE 
		hls = [M]*nHidden
		lr 	= 10**log_lr
		l2  = 10**log_l2

		model = ANN(hls)
		model.fit(Xtrain, Ytrain, learning_rate=lr, reg=12, mu=0.99, epochs=3000, show_fig=False)
		validation_accuracy = model.score(Xtest,Ytest)
		train_accuracy = model.score(Xtrain, Ytrain)
		print(
			"validation_accuracy: %.3f, train_accuracy %.3f, settings: %s, %s, %s" % (validation_accuracy, train_accuracy, hls, lr, l2)
		)
		if validation_accuracy > best_validation_rate:
			best_validation_rate = validation_accuracy
			best_M = M 
			best_nHidden = nHidden
			best_log_lr  = log_lr
			best_log_l2  = log_l2

			best_hls = hls 
			best_lr = lr
			best_l2 = l2


		M 			= max(best_M + np.random.randint(-10, 10)*10, 20)
		nHidden = max(best_nHidden + np.random.randint(-2, 2), 1)
		log_lr  = min(best_log_lr + np.random.randint(-1, 1), -1)
		log_l2  = min(best_log_l2 + np.random.randint(-1, 1), -1)
		print "M", M, "NHIDDEN", nHidden, "LOG-LR", log_lr, "LOG-L2", log_l2
	
	print("Best validation_accuracy", best_validation_rate)						
	print("Best settings:")						
	print("hidden_layer_size:", best_hls)						
	print("learning_rate:",best_lr)						
	print("l2:",best_l2)						



if __name__ == '__main__':
	random_search()		
