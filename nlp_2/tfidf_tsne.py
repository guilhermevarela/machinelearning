'''
	Created on Dec 19, 2017

	@author: Varela

	motivation: tf_idf + t-sne performs better than PCA

'''

import json 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle 
from sklearn.manifold import TSNE
from datetime import datetime

import os  
import sys 
sys.path.append(os.path.abspath('..'))
from rnn.util import get_wikipedia_data

from util import find_analogies
from sklearn.feature_extraction.text import TfidfTransformer

def main():
	sentences, word2idx=get_wikipedia_data(n_files=10, n_vocab=3000, by_paragraph=True)
	with open('w2v_word2idx.json', 'w') as f:
		json.dump(word2idx, f)

	#build term document matrix
	V=len(word2idx)
	N=len(sentences)

	#create raw counts first
	A= np.zeros((V,N))
	j=0
	for sentence in sentences:
		for i in sentence:
			A[i,j] +=1
		j +=1
	print('finishing getting raw counts')

	transformer = TfidfTransformer() 
	A= transformer.fit_transform(A)
	A=A.toarray()
	idx2word= {v:k for k,v in word2idx.items()}

	tsne=TSNE() 
	Z= tsne.fit_transform(A) 
	plt.scatter(Z[:,0],Z[:,1])

	for i in range(V):
		try:
			plt.annotate(s=idx2word[i].encode('utf8'), xy=(Z[i,0], Z[i,1]))
		except:
			print('bad string:', idx2word[i])
	plt.show()

	#optional;y get a highter-dimentionaly word embedding 
	#tsne= TSNE(n_componentes=3)
	#We= tsne.fit_transform(A)
	We=Z
	find_analogies('king', 'man', 'woman', We, word2idx)
	find_analogies('france', 'paris', 'london', We, word2idx)
	find_analogies('france', 'paris', 'rome', We, word2idx)

if __name__ == '__main__':
	main()