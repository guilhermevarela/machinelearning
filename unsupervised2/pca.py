'''
Created on Aug 28, 2017

@author: Varela

motivation: Unsupervised techniques: PCA on mnist data
	
'''

import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.decomposition import PCA 
from utils import get_mnist 

def main(): 
	Xtrain, Ytrain, Xtest, Ytest = get_mnist()

	pca = PCA() 
	reduced = pca.fit_transform(Xtrain)
	plt.scatter(reduced[:,0], reduced[:,1], s=100, c=Ytrain, alpha=.5)
	plt.show()

	plt.plot(pca.explained_variance_ratio_)
	plt.show() 

	#cumulative variance
	#choose k = number of dimensions that gives us 95%-99% variance
	cumulative = [] 
	last = 0 
	for v in pca.explained_variance_ratio_:
		cumulative.append(last + v)
		last = cumulative[-1]

	plt.plot(cumulative)
	plt.show() 

if __name__ == '__main__':
	main()