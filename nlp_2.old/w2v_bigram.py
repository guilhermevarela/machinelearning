'''
Created on Sep 04, 2017

@author: Varela

motivation: Using unsupervised techniques to create a word2vec model
						we want to predict the next word given the word that came before it 
'''


import numpy as np 
import matplotlib.pyplot as plt 
from nltk.corpus import stopwords as _stopwords
from nltk.corpus import brown as _brown
from sklearn.utils import shuffle 
import string 
import sys

def tokenizer(l, stopwords=None): 
	if stopwords is None: 
		stopwords= get_stopwords()

	tokens = [t.lower() for t in l] 		# to lowecase
	tokens = [remove_puctuation(t) for t in tokens] 
	tokens = [t for t in tokens if t not in stopwords] # remove stopwords
	tokens = filter(None, tokens)
	return tokens

def get_stopwords(lang='english'):	
	return set(_stopwords.words(lang))	

def remove_puctuation(s):
	'''
		s is a string with punctuation; converts unicode to string which might get data loss
			url: https://stackoverflow.com/questions/23175809/typeerror-translate-takes-one-argument-2-given-python
	'''	
	return 	str(s).translate(None, string.punctuation)


def word2indexify(sentence, word2idx={}):
	'''
		Converts a document to a list 
		Gets 
				sentence: i.e  the element of a document dict which is represented 
					list of strings (tokens). Information about of start and end periods is preserved

				word2idx: the previous word2idx dictionary if none exists then creates one

		Returns
			indexed_sentences: 	a flattened list of idx representing the tokens 
			word2idx :  				a token to index mapping 
			

	'''
	if not word2idx:
		idx= 0
	else:
		idx= max(word2idx.values())+1 

	idx_sentence=[] 
	for token in sentence: 
		token = token.lower()			
		if token not in word2idx: 
			word2idx[token]= idx 
			idx += 1 	
		idx_sentence.append(word2idx[token]) 	

	return idx_sentence, word2idx 	

def softmax(z):
	expz = np.exp(z)
	z = expz / expz.sum(axis=1, keepdims=1)
	return z 

class Bigram(object): 
	'''
		Simplest word2vec approach y = softmax(W1*W2*x)	
	'''
	def __init__(self, V, D):
		self.V=  V # vocabulary size 
		self.D=  D # hidden layer size
		self.W1= np.random.randn(V,D) / np.sqrt(V+D)
		self.W2= np.random.randn(D,V) / np.sqrt(V+D)

	def fit(self, X, Y, reg=10e-5, learning_rate=10e-3, epochs=500, batch_sz=10, show_fig=False):	
		N = len(X)
		X, Y = shuffle(X, Y)		
		
		
		n_batches= int(N / batch_sz) 
		costs=[] 
		rates=[] 
		best_classification_rate=-1
		for i in xrange(epochs):
			X, Y = shuffle(X, Y)		
			for j in xrange(n_batches):
				Xbatch = X[j*batch_sz:(j+1)*batch_sz,:]
				Ybatch = Y[j*batch_sz:(j+1)*batch_sz,:]

				pYbatch = self.forward(Xbatch)  
				W1 = self.W1
				W2 = self.W2
				# self.W -= learning_rate *(X.T.dot(pY-T) + reg*self.W)
				self.W1 -= learning_rate * ((Xbatch.T.dot(pYbatch-Ybatch)).dot(W2.T) + reg*W1)
				self.W2 -= learning_rate * ((Xbatch.T.dot(pYbatch-Ybatch)).dot(W1).T + reg*W2)
				if j % 5 == 0: 
					pY = self.forward(X)			
					Yhat = self.predict(X)	


					cost = -(Y*np.log(pY)).sum()
					classification_rate= np.mean(Yhat == (Y==1).sum(axis=1), axis=0)
					sys.stdout.write('epochs:%d of %d\tcost:%.3f\t:classification_rate:%.3f\r' % (i,epochs, cost, classification_rate))
					sys.stdout.flush()				
					if classification_rate > best_classification_rate:
						best_classification_rate=classification_rate
					costs.append(cost)					
					rates.append(classification_rate)
		if show_fig:
			plt.plot(costs)
			plt.show()

			plt.plot(rates)
			plt.show()

					
	def forward(self, X):
		return softmax(X.dot(self.W1.dot(self.W2)))

	def predict(self, X):
		pY = self.forward(X)
		return np.argmax(pY, axis=1)	
					
def toy():
	'''
		Toy example for gerating a valid dictionary
	'''
	sentences= [
		"It's rainning cats and dogs",
		"I love cats",
		"Dogs are cool, cats are not",
		"Snoop the dog is cool",
	]
	stopwords = get_stopwords()
	#preprocess
	for i, sentence in enumerate(sentences):
		sentences[i] = sentence.split(' ')

	#tokenizer	
	tokenized_sentences=[] 
	for sentence in sentences:
		tokens= tokenizer(sentence, stopwords=stopwords)
		tokenized_sentences.append(tokens)

	#maps the tokens to index	
	word2idx= {'START': 0, 'END': 1}		
	idx_sentences=[] 
	for tokens in tokenized_sentences:
		idx_sentence, word2idx  = word2indexify(tokens, word2idx)
		idx_sentences.append(idx_sentence)

	V = len(word2idx)
	print 'vocabulary size is '	, len(word2idx)
	
	#flattens list preserving starts & ends
	flat_sentences=[]
	for idx_sentence in idx_sentences:
		flat_sentences += [0] + idx_sentence + [1]

	#defines X,Y 	
	XX= flat_sentences[:-1]	
	YY= flat_sentences[1:]	

	

	
	Z = [(x,y) for x,y in zip(XX,YY) if x>=0 and y>=1]
	X, Y = zip(*Z)
	N = len(X)	

	print 'number of examples is', N 
	
	#converts the whole series into examples N of V size 
	Xind = np.array(X, dtype=np.int32)
	Yind = np.array(Y, dtype=np.int32)


	X = np.zeros((N,V), dtype=np.int32)
	Y = np.zeros((N,V), dtype=np.int32)

	# import code; code.interact(local=dict(globals(), **locals()))
	X[np.arange(N), Xind]= 1
	Y[np.arange(N), Yind]= 1

	
	D= int(V/4) 
	model= Bigram(V,D)
	model.fit(X,Y)


def brown():	
	sentences= _brown.sents(fileids=['ca01'])
	stopwords = get_stopwords()
	# import code; code.interact(local=dict(globals(), **locals()))
		#tokenizer	
	tokenized_sentences=[] 
	for sentence in sentences:
		tokens= tokenizer(sentence, stopwords=stopwords)
		tokenized_sentences.append(tokens)

	#maps the tokens to index	
	word2idx= {'START': 0, 'END': 1}		
	idx_sentences=[] 
	for tokens in tokenized_sentences:
		idx_sentence, word2idx  = word2indexify(tokens, word2idx)
		idx_sentences.append(idx_sentence)

	V = len(word2idx)
	print 'vocabulary size is '	, len(word2idx)
	
	#flattens list preserving starts & ends
	flat_sentences=[]
	for idx_sentence in idx_sentences:
		flat_sentences += [0] + idx_sentence + [1]

	#defines X,Y 	
	XX= flat_sentences[:-1]	
	YY= flat_sentences[1:]	

	

	
	Z = [(x,y) for x,y in zip(XX,YY) if x>=0 and y>=1]
	X, Y = zip(*Z)
	N = len(X)	

	print 'number of examples is', N 
	
	#converts the whole series into examples N of V size 
	Xind = np.array(X, dtype=np.int32)
	Yind = np.array(Y, dtype=np.int32)


	X = np.zeros((N,V), dtype=np.int32)
	Y = np.zeros((N,V), dtype=np.int32)

	# import code; code.interact(local=dict(globals(), **locals()))
	X[np.arange(N), Xind]= 1
	Y[np.arange(N), Yind]= 1

	
	D= int(V/2) 
	model= Bigram(V,D)
	model.fit(X,Y, epochs=2000, reg=10e-5 , learning_rate=10e-4, batch_sz=100, show_fig=True)



if __name__ == '__main__': 
		# toy()
		brown()


