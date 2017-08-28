'''
Created on Ago 26, 2017

@author: Varela

'''
import os 
import string
import numpy as np 

from nltk import pos_tag, word_tokenize

def init_weight(Mi, Mo):
	return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)

def all_parity_pairs(nbit):
	# total number of samples (Ntotal) will be a multiple of 100
	N = 2**nbit 
	remainder = 100 - (N % 100)
	Ntotal = N + remainder
	X = np.zeros((Ntotal, nbit))
	Y = np.zeros(Ntotal)

	for ii in xrange(Ntotal):
		i = ii % N 
		# now generate the ith sample
		for j in xrange(nbit):
			if i % (2**(j+1)) !=0: 
				i -= 2**j
				X[ii, j] = 1
		Y[ii] = X[ii].sum() % 2

	return X, Y

def all_parity_pairs_with_sequence_labels(nbit):
	X, Y = all_parity_pairs(nbit)
	N, t = X.shape

	# we want every time step to have a label
	Y_t = np.zeros(X.shape, dtype=np.int32)
	for n in xrange(N):
	    ones_count = 0
	    for i in xrange(t):
	        if X[n,i] == 1:
	            ones_count += 1
	        if ones_count % 2 == 1:
	            Y_t[n,i] = 1

	X = X.reshape(N, t, 1).astype(np.float32)
	return X, Y_t
	
def remove_puctuation(s):
	return 	s.translate(None, string.punctuation)


def get_robert_frost():
	word2idx = {'START':0, 'END':1}
	current_idx = 2
	sentences = [] 
	for line in open('../projects/poems/robert_frost.txt'):
		line = line.strip()
		if line: 
			tokens = remove_puctuation(line.lower()).split()
			sentence = [] 
			for t in tokens: 
				if t  not in word2idx:
					word2idx[t] = current_idx
					current_idx +=1 
				idx = word2idx[t]
				sentence.append(idx)
			sentences.append(sentence)
	return sentences, word2idx

def get_tags(s): 
	tuples = pos_tag(word_tokenize(s))
	return [y for x, y in tuples]

def get_poetry_classifier_data(samples_per_class, load_cached=True, saved_cached=True):	

	# import code; code.interact(local=dict(globals(), **locals()))
	datafile = 'poetry_classifier_data.npz'
	if load_cached and os.path.exists(datafile):
		npz = np.load(datafile)
		X = npz['arr_0']
		Y = npz['arr_1']
		V = int(npz['arr_2'])
		return X, Y, V 

	word2idx = {} 
	current_idx = 0 
	X = [] 
	Y = [] 
	poem1_path = '../projects/poems/robert_frost.txt'
	poem2_path = '../projects/poems/edgar_alan_poe.txt'
	for fn, label in zip((poem1_path, poem2_path), (0,1)):
		count=0
		for line in open(fn):
			line = line.strip()
			if line: 
				print line 
				#tokens = remove_punctuation(line.lower()).split()
				tokens = get_tags(line)
				if len(tokens) > 1:
					#scan doesn't work nice here, technicaly could fix
					for token in tokens: 
						if token not in word2idx: 
							word2idx[token] = current_idx
							current_idx +=1
						sequence = np.array([word2idx[w] for w in tokens])
						X.append(sequence)
						Y.append(label)
						count +=1 
						print(count)
						#quit early because the tokenizer	is very slow
						if count >= samples_per_class: 
							break 
		if saved_cached:
			np.savez(datafile, X, Y, current_idx)	
		return X, Y, current_idx
