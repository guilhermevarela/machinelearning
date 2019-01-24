'''
	Created on Feb 8, 2017
		
	@author: Varela

	Builds a markov model out of site_data

	*page_last_id, page_next_id
	*10 pages total
	#every sequence ends up on two states
		1) B (bounce)
		2) C (close)

	url https://www.udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python/learn/v4/t/lecture/5257360?start=0

'''

import numpy as np 

transitions= {}
row_sums= {}

#collect all counts

with open('site_data.csv','r') as f: 
	for line in f:		
		s, e = line.rstrip().split(',')
		transitions[(s,e)]= transitions.get((s,e), 0.0)+1.0
		row_sums[s]=row_sums.get(s, 0.0)+1.0

#normalize
for k, v in transitions.items():
	s, e= k 	
	transitions[k]= v / row_sums[s]

#initial state distribution
print('initial state distribution')
for k, v in transitions.items():
	s, e= k 	
	if s=='-1':
		print(e,v)

#which page has the hiest bounce
print('highest bounce rate')
for k, v in transitions.items():
	s, e= k 	
	if e=='B':
		print('bounce rate for {}: {}'.format(s,v))

