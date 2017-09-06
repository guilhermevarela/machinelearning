'''
Created on Sep 06, 2017

@author: Varela

motivation: 

'''

import json 
import re 
import numpy as np

import matplotlib.pyplot as plt 
from sklearn.utils import shuffle 

import os 
import sys 
sys.path.append(os.path.abspath('..'))
from util import get_wikipedia_data 
from util import find_analogies as _find_analogies
from datetime import datetime
# from util  import get_sentences_with_word2idx_limit_vocab 
# from util import find_analogies as _find_analogies

def sigmoid(x): 
	return 1 / (1 + np.exp(-1))

def init_weights(shape): 
	return np.random.randn(*shape).astype(np.float32) / np.sqrt(sum(shape))

class Model(object): 
	def __init__(self, D, V, context_sz): 
		self.D = D 
		self.V = V 
		self.context_sz = context_sz 

	def _get_pnw(self, X):
		#Creates the distributions for the negative sample

		word_freq = {} 
		word_count = sum(len(x) for x in X)
		for x in X:
			for xj in x:
				if xj not in word_freq:
					word_freq[xj] = 0
				word_freq[xj]	+=1
		self.Pnw= np.zeros(self.V)
		for j in xrange(2, self.V):
			self.Pnw[j] = (word_freq[j] / float(word_count)) **0.75

		assert(np.all(self.Pnw[2:]>0))

	def _get_negative_samples(self, context, num_neg_samples): 
		saved={} 
		for context_idx in context: 
			saved[context_idx] = self.Pnw[context_idx]
			self.Pnw[context_idx] = 0 

		neg_samples= np.random.choice(
			xrange(self.V),
			size=num_neg_samples,
			replace=False,
			p=self.Pnw / np.sum(self.Pnw)
		)
		for j, pnwj in saved.iteritems(): 
			self.Pnw[j] = pnwj
		
		assert(np.all(self.Pnw[2:]>0))
		return neg_samples 

	def fit(self, X, num_neg_samples=10, learning_rate=10e-4, mu=0.99, reg=0.1, epochs=10):
		N= len(X)
		V= self.V 
		D= self.D 
		self._get_pnw(X)

		self.W1 = init_weights((V,D))
		self.W2 = init_weights((D,V))
		dW1= np.zeros(self.W1.shape)
		dW2= np.zeros(self.W2.shape)

		costs= [] 
		cost_per_epoch= [] 
		sample_indices= range(N)
		for i in xrange(epochs):
			t0 = datetime.now() 
			sample_indices= shuffle(sample_indices)
			cost_per_epoch_i= [] 
			for it in xrange(N):
				j= sample_indices[it]
				x= X[j]

				#too short 
				if len(x) < 2 * self.context_sz + 1: 
					continue 

				cj= [] 
				n= len(x)
				for jj in xrange(n): 
					Z= self.W1[x[jj], :]
					start= max(0, jj -self.context_sz)
					end= max(n, jj+1 + self.context_sz)
					context= np.concatenate([x[start:jj], x[(jj+1):end]])
					context= np.array(list(set(context)), dtype=np.int32) 

					posA= Z.dot(self.W2[:,context])
					pos_pY= sigmoid(posA)

					neg_samples= self._get_negative_samples(context, num_neg_samples)
					negA= Z.dot(self.W2[:,neg_samples])
					neg_pY=sigmoid(-negA)
					c= -np.log(pos_pY).sum() - np.log(neg_pY).sum() 
					cj.append(c/ num_neg_samples + len(context))


					pos_err= pos_pY -1 
					print pos_err.shape 
					dW2[:, context]= mu*dW2[:, context] - learning_rate*(np.outer(Z, pos_err) + reg*self.W2[:, context])
					
					neg_err=1- neg_pY 
					print neg_err.shape 
					dW2[:, neg_samples]= mu*dW2[:, neg_samples] - learning_rate*(np.outer(Z, neg_err) + reg*self.W2[:, neg_samples])

					self.W2[:, context] += dW2[:, context]
					self.W2[:, neg_samples] += dW2[:, neg_samples]

					gradW1= pos_err.dot(self.W2[:,context].T) + neg_err.dot(self.W2[:, neg_samples].T)
					dW1[x[jj],:]= mu*dW1[x[jj],:] - learning_rate*(gradW1 + reg*self.W2[x[jj],:])

					self.W1[x[jj],:] += dW1[x[jj],:]

				cj= np.mean(cj)
				cost_per_epoch_i.append(cj)
				costs.append(cj)
				if it % 100 ==0: 
					sys.out.write('epoch: %d j: %d/%d cost: %f\r' % (i, it, N, cj))
					sys.out.flush()

			epoch_cost= np.mean(cost_per_epoch_i)		
			cost_per_epoch.append(epoch_cost)
			print 'time to complete epoch %d:' % i, (datetime.now()-t0), 'cost:', epoch_cost

		plt.plot(costs)
		plt.title('Numpy costs')
		plt.show() 

		plt.plot(cost_per_epoch)
		plt.title('Numpy cost at each epoch')
		plt.show()

	def save(self, fn):
		arrays= [self.W1, self.W2]
		np.savez(fn, *arrays)


def main():
	# sentences, word2idx= get_sentences_with_word2idx_limit_vocab(n_vocab=2000)
	sentences, word2idx= get_wikipedia_data(n_files=50, n_vocab=2000)
	with open('w2v_word2idx.json', 'w') as f: 
		json.dump(word2idx, f)

	V= len(word2idx)
	model = Model(80, V, 10)
	model.fit(sentences, learning_rate=10e-4, mu=0, epochs=7)
	model.save('w2v_model.npz')


def find_analogies(w1, w2, w3, concat=True, we_file='w2v_model.npz', w2i_file='w2v_word2idx.json'): 
	npz= np.load(we_file)
	W1= npz['arr_0']
	W2= npz['arr_1']

	with open(w2i_file) as f: 
		word2idx = json.load(f)

	V= len(word2idx)

	if concat:
		We= np.hstack([W1, W2.T])
		print 'We.shape', We.shape 
		assert(V== We.shape[0])

	else:
		We= (W1+ W2.T)/2

	_find_analogies(w1, w2, w3, we_file=We, w2i_file=w2i_file)	


if __name__ == '__main__':
	main()

	for concat in [True, False]:
		print '**concat:', concat
		find_analogies('king', 'man', 'woman', concat=concat)
		find_analogies('france', 'paris', 'london', concat=concat)
		find_analogies('france', 'paris', 'rome', concat=concat)
		find_analogies('paris', 'france', 'italy', concat=concat)
