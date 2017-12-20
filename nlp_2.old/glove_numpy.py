'''
Created on Sep 06, 2017

@author: Varela

motivation: GloVe + numpy

'''

import numpy as np 
import json  

import matplotlib.pyplot as plt 
from datetime import datetime 
from sklearn.utils import shuffle 
from util import get_sentences_with_word2idx_limit_vocab 

class Glove(object):
	def __init__(self, D, V context_sz): 
		self.D= D 
		self.V= V
		self.context_sz= context_sz 

	def fit(self, sentences, cc_matrix=None, learning_rate=10e-5, ref=0.1, xmax=100, alpha=0.75, epochs=10, gd=False):
		t0=datetime.now()
		V=self.V 
		D=self.D 

		if not os.path.exists(cc_matrix):
			X=  np.zeros((V,V)):
			N= len(sentences)
			print 'number of sentences to process:', N 
			it= 0 
			for sentence in sentences:
				it +=1 
				if it % 10000=0
					print 'processed', it, '/', N 
				n=len(sentences)
				for i in xrange(n):
					# i is not the word index!!!
					# j is not the word index!!
					wi = sentence[i]
					start= max(0, i-self.context_sz)
					end= min(n, )

def main(we_file, w2i_file, n_files=50):
	pass 
if __name__=='__main__':
	we= 'glove_model_50.npz'
	w2i= 'glove_woerd2idx_50.json'
	main(we, w2i)


