'''
Created on Aug 24, 2017

@author: Varela

motivation: Use article spinner on amazon's eletronics positive reviews

'''
import nltk
import random 
import numpy as np 

from bs4 import BeautifulSoup

def spinner_randspin(positive_reviews, trigrams, trigrams_probabilities): 
	review = random.choice(positive_reviews)
	s = review.text.lower() 
	print "original:", s
	tokens = nltk.tokenize.word_tokenize(s)
	for i in xrange(len(tokens)-2):
		if random.random() < 0.2:		 # 20% chance of replacement
			k = (tokens[i], tokens[i+2])
			if k in trigrams_probabilities:
				w = random_sample(trigrams_probabilities[k])
				tokens[i+1] = w 
				
	print "Spun:"
	print  " ".join(tokens).replace(" .", ".").replace(" '", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!")


def random_sample(d):
	r = random.random() 
	cumulative = 0 
	for w, p in d.iteritems():
		cumulative += p 
		if r < cumulative: 
			return w 

def main():
	# from http://www.lextek.com/manuals/onix/stopwords1.html
	stopwords = set(w.rstrip() for w in open('./aux/stopwords.txt'))
	datadir = './projects/sentiment_data/electronics/'
	# datadir = './electronics/'
	f =open(datadir + 'positive.review')
	positive_reviews = BeautifulSoup(f.read(), "lxml")
	positive_reviews = positive_reviews.findAll('review_text') # finds the key review text

	# extract trigrams and insert into dictionary
	# (w1, w3) is the key, [ w2 ] are the values
	trigrams = {} 
	for review in positive_reviews:
		s = review.text.lower() 
		tokens = nltk.tokenize.word_tokenize(s)
		for i in xrange(len(tokens)-2):
			k = (tokens[i], tokens[i+2])
			if k not in trigrams:
				trigrams[k] = [] 
			trigrams[k].append(tokens[i+1])
	
	
	trigrams_probabilities = {}		
	for k, words in trigrams.iteritems():
		if len(set(words)) >1:
			d = {} 
			n = 0 
			for w in words:
				if w not in d:
					d[w] = 0 
				d[w] +=1 
				n +=1 
			
			for w,c in d.iteritems():
				d[w] = float(c)	/n 
			trigrams_probabilities[k]	= d


	spinner_randspin(positive_reviews, trigrams, trigrams_probabilities)

if __name__ == '__main__':
	main()