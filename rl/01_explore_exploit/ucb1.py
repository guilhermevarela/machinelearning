import numpy as np
import matplotlib.pyplot as plt 
from comparing_epsilons import run_experiment as run_experiment_eps
from optimistic_initial_values import run_experiment as run_experiment_iov

class Bandit:
	def __init__(self, m):
		self.m = m
		self.mean = 0 
		self.N = 1

	def pull(self):
		return np.random.randn() + self.m

	def update(self, x):
		self.N +=  1
		self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x

def run_experiment(m1, m2, m3, N):
	bandits = [Bandit(m1), Bandit(m2), Bandit(m3)] 

	data = np.empty( N )
	n = [0] * 3
	for i in range(N):
		j = np.argmax([b.mean + np.sqrt( 2 * np.log( i ) / ( n[j] + 1e-5 ))
				for j, b in enumerate(bandits)])   
		x = bandits[j].pull()
		bandits[j].update(x) 
		
		# for the plot
		data[i] = x
		n[j] += 1
 
	cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

	#plot moving average ctr
	plt.plot(cumulative_average)
	plt.plot(np.ones(N) * m1 )
	plt.plot(np.ones(N) * m2 )
	plt.plot(np.ones(N) * m3 )
	plt.xscale('log')
	plt.show() 
	
	for b in bandits:
		print(b.mean)
	return cumulative_average

if __name__ == '__main__':
	c_1 = run_experiment_eps(1.0, 2.0, 3.0, 0.1, 10000) 
	 	      
	iov = run_experiment_iov(1.0, 2.0, 3.0, 10000) 

	ucb = run_experiment(1.0, 2.0, 3.0, 10000) 

	plt.plot(c_1, label='eps = 0.1') 
	plt.plot(iov, label='optimistic') 
	plt.plot(ucb, label='ucb') 
	plt.legend()
	plt.xscale('log')
	plt.show()
