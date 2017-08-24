'''
Created on Aug 23, 2017

@author: Varela

motivation: Sentiment analysis
POS tagging 
EX Bob is great ->(noun, verb, adjective)
full table			https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

Stemming and lemmatizing
Both reduce the word to a base form.
Stemmers only truncate
Lemmatizers adjust to the right radical
stem('wolves')  -> wolv
lemmatize('wolves')  -> wolf

NER (Named entity recognition)
What "are" the entities
"Albert Einstein" -> person
"Apple" 		  -> Organization


'''
import nltk 
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

#POS tagging 
# Bob is great
# (noun, verb, adjective)

# text = word_tokenize("Machine learning is great")
print "Pos_tag('Machine learning is great'): ", nltk.pos_tag("Machine learning is great".split())

#Stemmer reduces to base form by truncation
stemmer = PorterStemmer()

print "Wolves stems to :", stemmer.stem('Wolves')
print "Jumping stems to :", stemmer.stem('Jumping')

#Stemmer reduces to base form with correct radical
lemmatizer = WordNetLemmatizer()

print "Wolves lemmatizes to :", lemmatizer.lemmatize('Wolves') #--> should be working
print "Jumping lemmatizes to :", lemmatizer.lemmatize('Jumping')

s = "Albert Einstein was born on March 14, 1879"
tags = nltk.pos_tag(s.split())
print tags 

print nltk.ne_chunk(tags)
print nltk.ne_chunk(tags).draw()


