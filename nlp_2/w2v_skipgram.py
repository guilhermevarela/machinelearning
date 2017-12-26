'''
	Created on Dec 26, 2017

	@author: Varela

	Tensorflow eazy skipgram

	url: https://www.udemy.com/natural-language-processing-with-deep-learning-in-python/learn/v4/t/lecture/5505610?start=0

'''

import numpy as np 
import tensorflow as tf 

import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import sys
import os
sys.path.append(os.path.abspath('..'))
from rnn.util import get_wikipedia_data

def error_rate(p, t):
	return np.mean(p !=t )


def sentences2XY_batch(X_ind, Y_ind, sentences, sc):
	batch_sz, V = Y_ind.shape

def skipgram():
	n_files=10
	V=2000
	D=300
	C=5
	batch_sz = 1500

	sentences, word2idx=get_wikipedia_data(n_files=n_files, n_vocab=V, by_paragraph=True)
	epochs= 20
	print_period= 10 
	lr= 1e-4
	reg=  0.01 
	
	n_examples=total_examples(sentences, C)
	n_batches= int(n_examples/ batch_sz)


	X_ind = np.zeros((batch_sz, V+1), dtype=np.int32)
	Y_ind = np.zeros((batch_sz, C, V+1), dtype=np.int32)

	for i in range(epochs):
		sentences= shuffle(sentences)

		sc=0
		for j in range(n_batches):				
			X_ind, Y_ind, sc= sentences2XY_batch(X_ind, Y_ind, sentences, sc)


def total_examples(sentences, C):
	te=0 
	for sentence in sentences:
		te+= len(sentence)+2-C
	return te

if __name__ == '__main__':
	skipgram()
