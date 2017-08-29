'''
Created on Aug 28, 2017

@author: Varela

motivation: Unsupervised techniques auto encoder
	
'''

import numpy as np 
import theano 
import theano.tensor as T 
import matplotlib.pyplot as plt

from sklearn.utils import shuffle 
from util import relu, get_mnist, error_rate, init_weight

class AutoEncoder(object):
	def __init__(self, M, an_id): 
		self.M = M 
		self.id = an_id 

	def fit(self, X, learning_rate=0.5, mu=0.99, epochs=1, batch_sz=100, show_fig=False):
		N, D = X.shape 
		n_batches = N / batch_sz 

		W0 = init_weight(D, self.M)
		self.W = theano.shared(W0, 'W_%s' % self.id)
		self.bh = theano.shared(np.zeros(self.M), 'bh_%s' % self.id)    
		self.bo = theano.shared(np.zeros(D), 'bo_%s' % self.id)    
		self.params = [self.W, self.bh, self.bo] 
		self.forward_params = [self.W, self.bh] 

		self.dW  = theano.shared(np.zeros(W0.shape), 'dW_%s' % self.id) 
		self.dbh = theano.shared(np.zeros(self.M), 'dbh_%s' % self.id) 
		self.dbo = theano.shared(np.zeros(D), 'dbo_%s' % self.id) 
		self.dparams = [self.dW, self.dbh, self.dbo] 
		self.forward_dparams = [self.dW, self.dbh] 

		X_in  = T.matrix('X_%s' % self.id)
		X_hat = self.forward_output(X_in)

		H = T.nnet.sigmoid(X_in.dot(self.W) + self.bh)
		self.hidden_op = theano.function(
			inputs=[X_in],
			outputs=H,
		)
		# # save this for later so we can call it on reconstructiond
		self.predict  = theano.function(
			inputs=[X_in],
			outputs=X_hat,
		)

		#cost = ((X_in - X_hat) * (X_in - X_hat)).sum() /N 
		cost = -(X_in*T.log(X_hat) + (1-X_in)*T.log(1-X_hat)).flatten().mean()
		cost_op = theano.function(
			inputs=[X_in],
			outputs=cost,
		) 
		updates= [
			(p, p + mu*dp - learning_rate*T.grad(cost,p)) for p, dp in zip(self.params, self.dparams)
		] + [
			(dp, mu*dp - learning_rate*T.grad(cost,p)) for p, dp in zip(self.params, self.dparams)
		]

		train_op = theano.function(
			inputs=[X_in],
			updates=updates,
		)

		costs = [] 
		print "training autoencoder: %s" % self.id 
		for i in xrange(epochs):
			print "epoch:", i 
			X = shuffle(X)
			for j in xrange(n_batches): 
				batch = X[j*batch_sz:(j+1)*batch_sz]
				train_op(batch)
				the_cost = cost_op(batch)
				print "i:%d\tj:%d\tnb:%d\tcost:%.6f\t" % (i,j,n_batches,the_cost)
				costs.append(the_cost)
	
		if show_fig: 
			plt.plot(costs) 
			plt.show() 

	def forward_hidden(self, X): 
		Z = T.nnet.sigmoid(X.dot(self.W) + self.bh) 
		return Z 

	def forward_output(self, X): 
		Z = self.forward_hidden(X) 
		Y = T.nnet.sigmoid(Z.dot(self.W.T) + self.bo)
		return Y

class DNN(object):
	def __init__(self, hidden_layer_sizes, UnsupervisedModel=AutoEncoder):
		self.hidden_layers = [] 
		count=0
		for M in hidden_layer_sizes:
			ae = UnsupervisedModel(M, count)
			self.hidden_layers.append(ae)
			count +=1 

	def fit(self, X, Y, Xtest, Ytest, pretrain=True, learning_rate=0.01, mu=.99, reg=.1,epochs=1,batch_sz=100,show_fig=False)
		pretrain_epochs=1
		if not pretrain: 
			pretrain_epochs = 0 

		current_input = X 
		for ae  in self.hidden_layers: 
			ae.fit(current_input, epochs=pretrain_epochs) 
			current_input = ae.hidden_op(current_input)

		N = len(Y)
		K = len(set(Y))
		W0 = init_weight(self.hidden_layers[-1].M, K) 

		self.W = theano.shared(W0, 'W_logreg')
		self.b = theano.shared(np.zeros(K), 'b_logreg')
		self.params = [self.W, self.b] 
		for ae in self.hidden_layers: 
			self.params +=ae.forward_params

		self.dW = theano.shared(np.zeros(W0.shape), 'dW_logreg')
		self.db = theano.shared(np.zeros(K), 'db_logreg')
		self.dparams = [self.dW, self.db]
		for ae in self.hidden_layers:
			self.dparams += ae.forward_dparams

			X_in = T.matrix('X_in')
			targets = T.ivector('Targets')
			pY = self.forward(X_in)

			squared_magnitude = [(p*p) for p in self.params] 
			reg_cost= -T.sum(squared_magnitude)
			cost = -T.mean(T.log(pY[T.arange(pY.shape[0]), targets])) + reg_cost 
			prediction = self.predict(X_in)
			cost_predict_op = theano.function(
				inputs=[X_in, targets],
				outputs=[costs,prediction]
			) 
			updates= [
				(p, p + mu*dp - learning_rate*T.grad(cost,p)) for p, dp in zip(self.params, self.dparams)
			] + [
				(dp, mu*dp - learning_rate*T.grad(cost,p)) for p, dp in zip(self.params, self.dparams)
			]

			train_op = theano.function(
				inputs=[X_in,targets],
				updates=updates,
			) 
			n_batches = N / batch_sz
			costs=[]
			print 'supervised training'
			for i in xrange(epochs):
				print "epoch:", i
				X, Y = shuffle(X, Y) 
				for j in xrange(n_batches):
					Xbatch =X[j*batch_sz:(j+1)*batch_sz]
					Ybatch =Y[j*batch_sz:(j+1)*batch_sz]
					train_op(Xbatch, Ybatch)
					the_cost, the_prediction = cost_predict_op(Xtest, Ytest) 
					error = error_rate(the_prediction, Ytest)
					print "i:%d\tj:%d\tnb:%d\tcost:%.6f\terror:%.3f\t" % (i,j,n_batches,the_cost, error)
					costs.append(the_cost)

			if show_fig:
				plt.plot(costs)
				plt.show() 

	def predict(self, X):
		return T.argmax(self.forward(X), axis=1)

	def forward(self, X):
		current_input = X 
		for ae in self.hidden_layers: 
			Z = ae.forward_hidden(current_input) 
			current_input = Z 

		Y = T.nnet.softmax(T.dot(current_input, self.W) + self.b) 
		return Y
					

def test_single_autoencoder():
	Xtrain, Ytrain, Xtest, Ytest = get_mnist()

	autoencoder = AutoEncoder(300,0) 
	autoencoder.fit(Xtrain, epochs=2, show_fig=True)

	done=False
	while not done: 
		i = np.random.choice(len(Xtest))
		x = Xtest[i]
		y = autoencoder.predict([x]) 
		plt.subplot(1,2,1)
		plt.imshow(x.reshape(28,28), cmap='gray')
		plt.title('Original')
		
		plt.subplot(1,2,2)
		plt.imshow(y.reshape(28,28), cmap='gray')
		plt.title('reconstructed')


		plt.show() 

		ans= input('Generate another?')
		if ans and ans[0] in ('n' or 'N'):
			done= True




if __name__ == '__main__': 
	test_single_autoencoder()









