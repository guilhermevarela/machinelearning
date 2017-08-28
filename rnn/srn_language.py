'''
Created on Ago 27, 2017

@author: Varela

motivation: Using a Simple Recurrent Unit and Unsupervised learning in order to generate poetry

'''

import theano 
import theano.tensor as T 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle 
from util import init_weight, get_robert_frost 

class SRN(object):
	def __init__(self, D, M, V):
		self.D = D  # word embedding dimension
		self.M = M  # hidden layer size
		self.V = V  # vocabulary size

	#unsupervised learning: learns use words to predict next words 	
	def fit(self, X, learning_rate=10e-5, mu=0.99, reg=1.0, activation=T.tanh, epochs=500, show_fig=False):
		N = len(X)
		D = self.D 
		M = self.M 
		V = self.V 
		self.f = activation

		#Initialize Weights
		We = init_weight(V,D) # Word embeddings
		Wx = init_weight(D,M) # Entry layer-x
		Wh = init_weight(M,M) # h-to-h layer
		bh = np.zeros(M)
		
		h0 = np.zeros(M)
		Wo = init_weight(M,V)
		bo = np.zeros(V)


		self.We = theano.shared(We) 
		self.Wx = theano.shared(Wx) 
		self.Wh = theano.shared(Wh) 
		self.Wo = theano.shared(Wo) 

		self.bh = theano.shared(bh) 
		self.h0 = theano.shared(h0) 
		self.bo = theano.shared(bo) 
		self.params = [
			self.We, self.Wx, self.Wh, self.bh, self.Wo,  self.bo, self.h0 
		]

		#Theano inputs 
		thX = T.ivector('X')
		Ei = self.We[thX] # The real word index - T {lenght of sequence} x D  {size of word index}
		thY = T.ivector('Y')

		def recurrence(x_t, h_t1): 
			# returns h(t), y(t)
			h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
			y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
			return h_t, y_t 

		[h, y], _ = theano.scan(
			fn=recurrence,
			outputs_info=[self.h0, None],
			sequences=Ei,
			n_steps=Ei.shape[0],
		)	

		py_x =y[:, 0, :]
		prediction = T.argmax(py_x, axis=1)
		cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
		grads = T.grad(cost, self.params)
		dparams = [theano.shared(p.get_value()*0) for p in self.params ]

		updates = [
			(p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
		] + [
			(dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
		]

		self.predict_op = theano.function(inputs=[thX], outputs=prediction)
		self.train_op = theano.function(
			inputs=[thX, thY],
			outputs=[cost, prediction],
			updates=updates,
		)

		costs =[]
		n_total = sum((len(sentence) +1) for sentence in X)
		for i in xrange(epochs):
			X = shuffle(X)
			n_correct = 0 
			cost = 0 
			for j in xrange(N):
				input_sequence= [0] + X[j]
				output_sequence= X[j] + [1]

				c, p = self.train_op(input_sequence, output_sequence)
				cost += c 
				for pj, xj in zip(p, output_sequence):
					if pj ==xj: 
						n_correct+=1
			print "i:%d\tcost:%.3f\tcorrect rate:%.3f\t" % (i, cost, float(n_correct) / n_total)
			costs.append(cost)

		if show_fig: 
			plt.plot(costs)
			plt.show()

	def save(self, filename):
		np.savez(filename, *[p.get_value() for p in self.params])
	
	def set(self,We, Wx, Wh, bh, h0, Wo, bo, activation): 
		self.f = activation 
		self.We = theano.shared(We) 
		self.Wx = theano.shared(Wx) 
		self.Wh = theano.shared(Wh) 
		self.Wo = theano.shared(Wo) 

		self.bh = theano.shared(bh) 
		self.h0 = theano.shared(h0) 
		self.bo = theano.shared(bo) 

		self.params = [
			self.We, self.Wx, self.Wh, self.bh, self.Wo,  self.bo, self.h0 
		]

		#Theano inputs 
		thX = T.ivector('X')
		Ei = self.We[thX] # The real word index - T {lenght of sequence} x D  {size of word index}
		

		def recurrence(x_t, h_t1): 
			# returns h(t), y(t)
			h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
			y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
			return h_t, y_t 

		[h, y], _ = theano.scan(
			fn=recurrence,
			outputs_info=[self.h0, None],
			sequences=Ei,
			n_steps=Ei.shape[0],
		)	

		py_x =y[:, 0, :]
		prediction = T.argmax(py_x, axis=1)

		self.predict_op = theano.function(
			inputs=[thX], 
			outputs=prediction, 
			allow_input_downcast=True
		)

		return self 

	def generate(self, pi , word2idx): 
		idx2word = 	{v:k for k, v in word2idx.iteritems()}
		V = len(pi)

		n_lines = 0 
		X = [ np.random.choice(V, p=pi)] 
		print idx2word[X[0]],

		while n_lines < 4: 
			P = self.predict_op(X)[-1]
			X += [P]

			P = P[-1]
			if P > 1: 
				word = idx2word[P]
				print word, 
			elif P == 1: 
				n_lines +=1 
				print ''
				if n_lines < 4: 
					X = [np.random.choice(V, p=pi)]
					print idx2word[X[0]] 


	@staticmethod 
	def load(filename, activation):
		npz= np.load(filename)
		# self.params = [
		# 	self.We, self.Wx, self.Wh, self.bh, self.Wo,  self.bo, self.h0 
		# ]
		We = npz['arr_0']
		Wx = npz['arr_1']
		Wh = npz['arr_2']
		bh = npz['arr_3']
		h0 = npz['arr_4']
		Wo = npz['arr_5']
		bo = npz['arr_6']

		V, D = We.shape
		_, M = Wx.shape

		srn = SRN(D, M, V)
		srn = srn.set(We, Wx, Wh, bh, h0, Wo, bo, activation)
		return srn 

def train_poetry(): 
	sentences , word2idx = get_robert_frost()
	D = 30 
	M = 50
	epochs = 500
	srn = SRN(D, M, len(word2idx))
	srn.fit(sentences, learning_rate=10e-5, show_fig=True,activation=T.nnet.relu, epochs=epochs)
	srn.save('RNN_D%d_M%d_epochs%d_relu.npz' % (D, M, epochs))

def generate_poetry(): 
	sentences , word2idx = get_robert_frost() 
	srn = SRN.load('RNN_D30_M30_epochs300_relu.npz', T.nnet.relu)

	V = len(word2idx)
	pi = np.zeros(V)
	for sentence in sentences: 
		pi[sentence[0]] +=1 

	pi /= pi.sum()

	srn.generate(pi, word2idx)

if __name__ == '__main__':
	train_poetry()
	generate_poetry()









		


