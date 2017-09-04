'''
Created on Sep 04, 2017

@author: Varela

motivation: util					

'''
import json 
import numpy as np 

from nltk.corpus import brown 
import operator 

# From brown.py 
# we absolutely want to keep these words in order to make comparisons
KEEP_WORDS = set([
	'king', 'man', 'queen', 'woman',
	'italy', 'rome', 'france', 'paris',
	'london', 'britain', 'england',
])

def get_sentences(): 
	# return 57340  sentences of brown corpus
	#each sentence is represent by a list of individual string tokens
	return brown.sents() 

def get_sentences_with_word2idx_limit_vocab(n_vocab=2000, keep_words=KEEP_WORDS):
	# Returns sentences as indexes of words and word2idx mapping
	# but limits to n_vocab and forces to keep WORDS
	sentences = get_sentences() 
	indexed_sentences= [] 

	i=2
	word2idx=  {'START':0, 'END':1}
	idx2word=  ['START', 'END']

	word_idx_count={
		0: float('inf'),
		1: float('inf'),
	}
	for sentence  in sentences: 
		indexed_sentence= [] 
		# This loop converts words to index and 
		# it fills the word2idx dictionary
		for token in sentence: 
			token= token.lower()
			if token not in word2idx: 
				idx2word.append(token)
				word2idx[token]= i 
				i +=1 

			#keep track of counts for later sorting
			idx = word2idx[token]
			word_idx_count[idx] = word_idx_count.get(idx,0) +1

			indexed_sentence.append(idx)
		indexed_sentences.append(indexed_sentence)

		#restrict vocab size
		#set all the words I want to keep to infinity
		# so that they are included when I pick the most common words
  	for word in keep_words:
			word_idx_count[word2idx[word]] = float('inf')


	#remapping to new smaller vocabulary words
	#updates the dictionary	
	sorted_word_idx_count= sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
	word2idx_small = {}
	new_idx =0 
	idx_new_idx_map= {} 
	for idx, count in sorted_word_idx_count[:n_vocab]: 
		word= idx2word[idx]
		print word, count 

		word2idx_small[word]= new_idx
		idx_new_idx_map[idx]= new_idx 
		new_idx +=1 

	# let 'unknown' be the last token
	word2idx_small['UNKNOWN']= new_idx 
	unknown= new_idx 

	# sanity check
	assert('START' in word2idx_small)
	assert('END' in word2idx_small)
	for word in keep_words:
		assert( word in word2idx_small)

	#map old idx to new idx
	sentences_small = [] 
	for sentence in indexed_sentences: 
		if len(sentence) > 1:
			new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown] 
			sentences_small.append(new_sentence)

	return sentences_small, word2idx_small  
def find_analogies(w1, w2, w3, we_file='word_embeddings.npy', w2i_file='wikipedia_word2idx.json'):
	We= np.load(we_file)
	with(w2i_file) as f: 
		word2idx = json.load(f)

	king= We[word2idx[w1]]
	man= We[word2idx[w2]]
	woman= We[word2idx[w3]]
	v0 = king - man + woman

	def dist1(a, b):
		return np.linalg.norm(a-b)

	def dist2(a, b):
		return 1- a.dot(b) / (np.linalg.norm(a) + np.linalg.norm(b))

	for dist, name in [(dist1, 'Euclidean'), (dist2, 'cosine')]: 
		min_dist = float('inf')
		best_word= ''
		for word, idx in word2idx.iteritems(): 
			if word not in (w1, w2, w3): 
				v1 = We[idx]
				d = dist1(v0,v1)
				if d < min_dist: 
					min_dist= d 
					best_word= word 
		print "closest match by", name, "distance:", best_word
		print w1, "-", w2, "=", best_word, "-", w3 
