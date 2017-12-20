'''
	Created on Dec 19, 2017

	@author: Varela

	references: 
	https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
	https://udemy.com/natural-language-processing-with-deep-learning-in-python

'''
def find_analogies(w1, w2, w3, We, word2idx):
	'''
		w1 is to w2, what w3 is to result
		Paris is to France, London is to England

		INPUT
			w1<string>
			w2<string>
			w3<string>
			We<float>: DxN matrix where D is the embedding dimension, N the vocabulary size
			word2idx<dict<<string>,<int>>

		OUTPUT

	'''

	king= We[word2idx[w1]]
	man= We[word2idx[w2]] 
	woman= We[word2idx[w3]]
	v0= king-man+woman

	def dist1(a, b):
		return np.linalg.norm(a-b)

	def dist2(a, b):
		return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

	for dist, name in [(dist1, 'euclidean'),(dist2, 'cosine')]:
		min_dist=float('inf')
		best_word= ''

		for word, idx in word2idx.iteritems():
			if word not in (w1, w2, w3) :
				v1 = We[idx]
				d = dist(v0, v1)
				if d< min_dist:
					min_dist= d 
					best_word= word 

		print('closest match by', name, 'distance:', best_word)
		print(w1, '-', w2, '=', best_word, '-', w3)

