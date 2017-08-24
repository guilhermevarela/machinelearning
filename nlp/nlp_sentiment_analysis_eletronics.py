'''
Created on Aug 23, 2017

@author: Varela

motivation: Sentiment analysis
			Eletronics category / Positive or Negative Reviews

'''

import nltk 
import numpy as np 

from nltk.stem import WordNetLemmatizer 
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup 

def my_tokenizer(s, lemmatizer, stopwords): 
	s = s.lower() 
	tokens = nltk.tokenize.word_tokenize(s)
	tokens = [t for t in tokens if len(t) > 2] # remove words shorter then 3
	tokens = [t for t in tokens if t not in stopwords] # remove stopwords
	tokens = [lemmatizer.lemmatize(t) for t in tokens] # converts to baseform
	return tokens	

def token2vec(word_index_map, tokens, label):  
	x = np.zeros(len(word_index_map) +1)
	for t in tokens:
		i = word_index_map[t] 
		x[i] +=1 
	x = x / x.sum() 
	x[-1] = label
	return x	


def main():
	#cat, cats, cat's -> cat Converts to baseform 
	wordnet_lemmatizer = WordNetLemmatizer()
	# from http://www.lextek.com/manuals/onix/stopwords1.html
	stopwords = set(w.rstrip() for w in open('./aux/stopwords.txt'))
	datadir = './datasets/sentiment_data/electronics/'
	# datadir = './electronics/'
	f =open(datadir + 'positive.review')
	positive_reviews = BeautifulSoup(f.read(), "lxml")
	positive_reviews = positive_reviews.findAll('review_text') # finds the key review text

	f =open(datadir + 'negative.review')
	negative_reviews = BeautifulSoup(f.read(), "lxml")
	negative_reviews = negative_reviews.findAll('review_text') # finds the key review text

	#Balance negative / positive reviews removing positive
 	np.random.shuffle(positive_reviews)
 	positive_reviews = positive_reviews[:len(negative_reviews)]
	

 	word_index_map = {} 
 	current_index = 0 

 	positive_tokenized = [] 
 	for review in positive_reviews: 
 		tokens = my_tokenizer(review.text, wordnet_lemmatizer, stopwords)
 		positive_tokenized.append(tokens)
 		for token in tokens: 
 			if token not in word_index_map: 
 				word_index_map[token] = current_index
 				current_index +=1

 	negative_tokenized = [] 			
 	for review in negative_reviews: 
		tokens = my_tokenizer(review.text, wordnet_lemmatizer, stopwords)
 		negative_tokenized.append(tokens)
 		for token in tokens: 
 			if token not in word_index_map: 
 				word_index_map[token] = current_index
 				current_index +=1 

 	N = len(positive_tokenized)	 + len(negative_tokenized)		 
 	# (N x D+1 matrix - keeping them together for now so we can shuffle more easily later
 	data = np.zeros((N, len(word_index_map)+1))
 	i = 0 
 	for tokens in positive_tokenized:
 		xy = token2vec(word_index_map, tokens, 1)
 		data[i, :] = xy 
 		i+=1

 	for tokens in negative_tokenized:
 		xy = token2vec(word_index_map, tokens, 0) 		
 		data[i, :] = xy 
 		i+=1

 	np.random.shuffle(data)	
 	X = data[:, :-1]
 	Y = data[:,-1]
 	
 	Xtrain = X[:-100,]
 	Ytrain = Y[:-100,]
 	Xtest = X[-100:,]
 	Ytest = Y[-100:,]

	
 	lgr = LogisticRegression()
 	lgr.fit(Xtrain, Ytrain)
 	print "LogisticRegression: Classification rate:", lgr.score(Xtest, Ytest)


	# let's look at the weights for each word
	# try it with different threshold values!
	threshold = 0.5
	for word, index in word_index_map.iteritems():
	    weight = lgr.coef_[0][index]
	    if weight > threshold or weight < -threshold:
				print word, weight

if __name__ == '__main__':
	main()



