'''
Created on Aug 23, 2017

@author: Varela

motivation: Singular value decomposition relating documents to a corpora

'''

import nltk
import numpy as np 
import matplotlib.pyplot as plt 

from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

def token2vec(word_index_map, tokens):  
	x = np.zeros(len(word_index_map))
	for t in tokens:
		i = word_index_map[t] 
		x[i] =1 
	
	return x	

def my_tokenizer(s, lemmatizer, stopwords): 
	s = s.lower() 
	tokens = nltk.tokenize.word_tokenize(s)
	tokens = [t for t in tokens if len(t) > 2] # remove words shorter then 3
	tokens = [t for t in tokens if t not in stopwords] # remove stopwords
	tokens = [lemmatizer.lemmatize(t) for t in tokens] # converts to baseform
	tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # removes digits 
	return tokens	

def main():
	wordnet_lemmatizer = WordNetLemmatizer() 
	titles = [line.strip() for line in open('all_book_titles.txt')] 
	stopwords = set(w.rstrip() for w in open('../aux/stopwords.txt'))
	stopwords = stopwords.union([ 
		'introduction', 'edition', 'series', 'application',
		'approach', 'card', 'access', 'package', 'plus', 'etext',
		'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
		'third', 'second', 'fourth',
	])

	word_index_map = {}
	current_index = 0
	all_tokens = []
	all_titles  = [] 
	index_word_map = []
	for title in titles:
		try: 
			title = title.encode('ascii', 'ignore')
			all_titles.append(title)
			tokens = my_tokenizer(title,wordnet_lemmatizer, stopwords)
			all_tokens.append(tokens)
			for token in tokens:
				if token not in word_index_map:
					word_index_map[token] = current_index
					current_index +=1
					index_word_map.append(token)

		except:
			pass

			N = len(all_tokens)
			D = len(word_index_map)
			X = np.zeros((D,N))
			i = 0 
			for token in all_tokens:
				X[:,i] = token2vec(word_index_map,tokens)
				i+=1

	
	svd = TruncatedSVD() 
	Z = svd.fit_transform(X)			
	plt.scatter(Z[:,0],Z[:,1])
	for i in xrange(D):
		plt.annotate(s=index_word_map[i], xy=(Z[i,0],Z[i,1]))
	plt.show()		

if __name__ == '__main__':
	main()	