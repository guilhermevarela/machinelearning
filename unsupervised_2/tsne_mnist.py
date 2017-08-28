'''
Created on Aug 28, 2017

@author: Varela

motivation: Unsupervised techniques: tSNE on mnist
	
'''

import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.manifold import TSNE 
from util import get_mnist 

def main():
	Xtrain, Ytrain, _, _ = get_mnist() 

	sample_size=100
	X = Xtrain[:sample_size]
	Y = Ytrain[:sample_size]

	tsne= TSNE(method='exact') 
	Z = tsne.fit_transform(X)
	plt.scatter(Z[:,0], Z[:,1], s=100, c=Y, alpha=.5)

if __name__ == '__main__':
	main()
