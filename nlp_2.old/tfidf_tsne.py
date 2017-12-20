'''
Created on Sep 04, 2017

@author: Varela

motivation: Using unsupervised learning techniques to find analogies
					Term Frequency - Inverse Document Frequency + TSNE

'''

import json 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle 
from sklearn.manifold import TSNE 
from datetime import datetime 

import os 
import sys 
sys.path.append(os.path.abspath('.'))
from util import find_analogies, get_sentences_with_word2idx_limit_vocab
from sklearn.feature_extraction.text import TfidfTransformer 

def main(): 
	# sentences, word2idx = get_wikepedia_data(n_files=10, n_vocab=1500, by_paragraph=True)
	sentences, word2idx = get_sentences_with_word2idx_limit_vocab(n_vocab=2000)
	with open('w2v_word2idx.json', 'w') as f: 
		json.dump(word2idx, f)

	# build term document matrix	
	V = len(word2idx)
	N = len(sentences)

	#create raw counts first 
	A = np.zeros((V,N))
	j =0 
	for sentence in sentences: 
		for i in sentence:
			A[i, j] +=1 

		j+=1
	print 'finished getting raw counts'	

	transformer = TfidfTransformer()
	A = transformer.fit_transform(A) # sparse matrix

	A = A.toarray() # converts to array 

	idx2word = {v:k for k,v in word2idx.iteritems()}
	tsne = TSNE()	
	Z = tsne.fit_transform(A)
	plt.scatter(Z[:,0],Z[:,1])
	for i in xrange(V):
		try: 
			plt.annotate(s=idx2word[i].encode('utf8'), xy=(Z[i,0],Z[i,1]))
		except: 
			print 'bad string:', idx2word[i]

	

	#optionally get a higher dimensionality word embedding
	# tsne = TSNE(n_components=3)
	# We = tsne.fit_transform(A)
	# or 	
	We = Z
	find_analogies('king', 'man', 'woman', We, word2idx)
	find_analogies('france', 'paris', 'london', We, word2idx)
	find_analogies('france', 'paris', 'rome', We, word2idx)
	find_analogies('paris', 'france', 'italy', We, word2idx)
	plt.show()
	
if __name__ == '__main__': 
	main()






