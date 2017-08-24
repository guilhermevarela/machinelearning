'''
Created on Aug 23, 2017

@author: Varela

motivation: NaiveBayes on spam detection
docs reference: 
	* naive bayes http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
	* tfidf http://scikit-learn.org/stable/modules/feature_extraction.html
'''

import numpy as np 

from utils import get_spambase
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle 

from sklearn.ensemble import AdaBoostClassifier
# from sklearn.feature_extraction.text import TfidfTransformer

def main():
	X, Y  = get_spambase()
	X, Y  = shuffle(X, Y)

	# Define dimensions 
	N, D  = X.shape 	
	M = 5
	K = len(np.unique(Y))

	Ntrain = N -300 
	Xtrain, Ytrain = X[:Ntrain,: ], Y[:Ntrain]
	

	Ntest  = 300
	Xtest, Ytest  = X[-Ntest:,: ], Y[-Ntest:]
	
	
	# model fit 
	nb = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
	nb.fit(Xtrain,Ytrain)

	Yhat = nb.predict(Xtest)
	print "NaiveBayes: classification rate:", np.mean(Yhat == Ytest)
	# print "Classification rate:", nb.score(Xtest, Ytest)

	ada = AdaBoostClassifier()
	ada.fit(Xtrain,Ytrain)	
	print "Adaboost: classification rate:", ada.score(Xtest, Ytest)

	# tfidf = TfidfTransformer()
	# tfidf.fit(Xtrain,Ytrain)
	# Yhat = tfidf.predict(Xtest)	
	# print "TFIdf: classification rate:", np.mean(Yhat == Ytest)




if __name__ == '__main__':
	main()	