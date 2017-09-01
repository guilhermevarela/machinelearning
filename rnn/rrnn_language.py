'''
Created on Sep 01, 2017

@author: Varela

motivation: Using a Rated Recurrent Unit + Unsupervised 

'''

import theano 
import theano.tensor as T 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle 
from util import init_weight, get_robert_frost 

class RRNN(object):
	def __init__(self, D, M, V):
		self.D = D  # word embedding dimension
		self.M = M  # hidden layer size
		self.V = V  # vocabulary size

	#unsupervised learning: learns use words to predict next words 	
	def fit(self, X, learning_rate=10e-5, mu=0.99, reg=1.0, activation=T.nnet.relu, epochs=500, show_fig=False):
		N = len(X)
		D = self.D 
		M = self.M 
		V = self.V 
		
		#Initialize Weights
		We = init_weight(V,D) # Word embeddings
		Wx = init_weight(D,M) # Entry layer-x
		Wh = init_weight(M,M) # h-to-h layer
		bh = np.zeros(M)

		Wxz = init_weight(D,M) 
		Whz = init_weight(M,M) 
		bz = np.zeros(M)
		
		h0 = np.zeros(M)
		Wo = init_weight(M,V)
		bo = np.zeros(V)

		thX, thY, py_x, prediction = self.set(We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo, activation)

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
		for i in xrange(epochs):
			X = shuffle(X)
			n_correct = 0 
			cost = 0 
			n_total = 0 
			for j in xrange(N):
				if np.random.random() < 0.1:
					input_sequence= [0] + X[j]
					output_sequence= X[j] + [1]
				else: 
					input_sequence= [0] + X[j][:-1]
					output_sequence= X[j]
				n_total += len(output_sequence)		
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
	
	def set(self,We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo, activation): 
		self.f 		= activation 
		self.We 	= theano.shared(We) 
		self.Wx 	= theano.shared(Wx) 
		self.Wh 	= theano.shared(Wh) 
		self.bh 	= theano.shared(bh) 
		self.h0 	= theano.shared(h0) 

		self.Wxz 	= theano.shared(Wxz) 
		self.Whz 	= theano.shared(Whz) 
		self.bz 	= theano.shared(bz) 
				
		
		self.Wo 	= theano.shared(Wo) 
		self.bo 	= theano.shared(bo) 

		self.params = [
			self.We, self.Wx, self.Wh, self.bh, self.h0 , self.Wxz, self.Whz, self.bz , self.Wo,  self.bo 
		]

		#Theano inputs 
		thX = T.ivector('X')
		Ei = self.We[thX] # The real word index - T {lenght of sequence} x D  {size of word index}
		thY = T.ivector('Y')

		def recurrence(x_t, h_t1): 
			# returns h(t), y(t)			
			hhat_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)			
			z_t = T.nnet.sigmoid(x_t.dot(self.Wxz) +  h_t1.dot(self.Whz) + self.bz) 			
			h_t = (1-z_t) * h_t1 + z_t * hhat_t
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
			outputs=[py_x,prediction], 
			allow_input_downcast=True
		)

		return thX, thY, py_x, prediction

	def generate(self, pi , word2idx): 
		idx2word = 	{v:k for k, v in word2idx.iteritems()}
		V = len(word2idx)

		n_lines = 0 
		X = [ 0 ] 
		

		while n_lines < 4: 
			PY_X, _   = self.predict_op(X)[-1]
			PY_X = PY_X[-1].flatten()
			P = [np.random.choice(V, p=PY_X)]
			X += [P]

			P = P[-1]
			if P > 1: 
				word = idx2word[P]
				print word, 
			elif P == 1: 
				n_lines +=1 
				print ''
				if n_lines < 4: 
					X = [0]
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
		Wxz = npz['arr_5']
		Whz = npz['arr_6']
		bz = npz['arr_7']
		Wo = npz['arr_8']
		bo = npz['arr_9']

		V, D = We.shape
		_, M = Wx.shape

		srn = SRN(D, M, V)
		srn = srn.set(We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo, activation)
		return srn 

def train_poetry(): 
	sentences , word2idx = get_robert_frost()
	D = 50 
	M = 50
	epochs = 500
	rrnn = RRNN(D, M, len(word2idx))
	rrnn.fit(sentences, learning_rate=10e-5, show_fig=True,activation=T.nnet.relu, epochs=epochs)
	rrnn.save('RRNN_D%d_M%d_epochs%d_relu.npz' % (D, M, epochs))

def generate_poetry(): 
	sentences , word2idx = get_robert_frost() 
	rrnn = RRNN.load('RRNN_D30_M30_epochs500_relu.npz', T.nnet.relu)

	rrnn.generate(word2idx)

if __name__ == '__main__':	
	train_poetry()
	generate_poetry()









		


