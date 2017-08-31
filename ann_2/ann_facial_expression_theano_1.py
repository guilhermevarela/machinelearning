'''
Created on Aug 24, 2017

@author: Varela

motivation: Multilayer perceptron for facial expression recognition + theano + multiple layers
	features: batching
'''

import numpy as np 

import theano
import theano.tensor as T 

from sklearn.utils import shuffle 
from utils import get_facialexpression, sigmoid, softmax, classification_rate , error_rate, relu, init_weight_and_bias  

class HiddenLayer(object):
	def __init__(self, M1, M2, an_id):
		self.id = an_id 
		self.M1 = M1 
		self.M2 = M2
		W, b = init_weight_and_bias(M1, M2)
		self.W = theano.shared(W, 'W_%s' % an_id)
		self.b = theano.shared(b, 'b_%s' % an_id)
		self.params=[self.W,self.b]


	def forward(self, X):
		return relu(X.dot(self.W) + self.b)	

class AnnTheano2(object):
	def __init__(self, hidden_layer_sizes):
		self.hidden_layer_sizes = hidden_layer_sizes


	def fit(self, X, Y, learning_rate=10e-4, reg=10e-8, epochs=10000, show_figure=False):
		Nvalid = 1000
		N, D  = X.shape 
		K =  len(np.unique(Y))
		X, Y  = shuffle(X, Y)

		Xvalid, Yvalid = X[-Nvalid:,:],  Y[-Nvalid:,]
		X, Y = X[:-Nvalid,:], Y[:-Nvalid,]


		#Initialize Hidden layers 
		self.hidden_layers = [] 
		M1 = D 		
		for count, M2 in enumerate(self.hidden_layer_sizes):
			hidden_layer =  HiddenLayer(M1, M2, count)
			self.hidden_layers.append(hidden_layer)
			M1=M2

		#final layer
		W, b = init_weight_and_bias(M1, K)  
		self.W = theano.shared(W, 'W_logreg')
		self.b = theano.shared(b, 'b_logreg')

		#collect parameters for later use
		self.params = []
		for h in self.hidden_layers: 
			self.params += h.params
		self.params += [self.W, self.b]
		
		
		#Theano variables 
		thX = T.fmatrix('X')
		thY = T.ivector('Y')		
		pY =self.th_forward(thX)

		costs = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY]))
		prediction = self.th_predict(thX)


		#actual prediction functions and variabels
		self.predict_op=theano.function(inputs=[thX], outputs=prediction)
		cost_predict_op=theano.function(inputs=[thX, thY], outputs=[costs, prediction])

		#Streamline initializations
		updates = [
			(p, p - learning_rate*(T.grad(costs,p) + reg*p)) for p  in self.params
		]
		
		train_op = theano.function(
			inputs=[thX, thY],
			updates=updates,
			allow_input_downcast=True,
		)

		batch_sz=200
		n_batches = N / batch_sz
		costs = [] 
		for i in xrange(epochs):
			X,Y = shuffle(X,Y)
			for j in range(n_batches):
				Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz),:]
				Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

				train_op(Xbatch.astype(np.float32), Ybatch.astype(np.int32))
				
				if j % 100 == 0:
					c, p = cost_predict_op(Xvalid.astype(np.float32), Yvalid.astype(np.int32))
					costs.append(c)
					err = error_rate(Yvalid, p)

					print "i:%d\tj:%d\tnb:%d\tc:%.3f\terr:%.3f\t" % (i,j,n_batches,c,err)
			print "i:%d\tj:%d\tnb:%d\tc:%.3f\terr:%.3f\t" % (i,batch_sz,n_batches,c,err)		
		
		print "Final error rate", err 
			
		if show_fig: 
			plt.plot(costs)
			plt.show() 

	def th_forward(self, X):
		Z = X 
		for h in self.hidden_layers:
			Z = h.forward(Z)
		return T.nnet.softmax(Z.dot(self.W) + self.b)

	def th_predict(self, X):
		pY = self.th_forward(X)
		return T.argmax(pY, axis=1)		
		
	def predict(self, X):		
		return self.predict_op(X)

	def score(self, X, Y):		
		pY = self.predict_op(X)		
		return np.mean(pY == Y)

	


def main():
	X, Y =  get_facialexpression(balance_ones=True)

	ann = AnnTheano2([2000, 100, 500])
	ann.fit(X, Y)
	print "score:", ann.score(X,Y)

if __name__ == '__main__':
	theano.config.floatX = 'float32'
	main()