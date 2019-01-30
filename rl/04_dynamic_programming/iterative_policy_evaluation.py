import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid

SMALL_ENOUGH = 10e-4 # threshold for convergence

def print_vales(V, g):
	for i in range( g.width ):
		print("____________________")
		for j in range( g.height ):
			v = V.get((i, j ), 0)
			if v >= 0:
				print( " {:.2f}|".format(v), end=0)
			else:
				print( "{:.2f}|".format(v), end=0) # -ve signal takes up an extra space
		print("")

def print_policy(P, g):
	for i in range( g.width ):
		print("____________________")
		for j in range( g.height ):
			p = P.get((i, j ), ' ')
			print (" {:s} |".format(p), end=0)
		print("")

if __name__ == '__main__':
	# iterative policy evaluation
	# given a policy, let's find it's value function V(s)
	# we will do this for both a uniform random policy and fixed policy
	# NOTE:
	# there are 2 sources of randomness
	# p(a|s) - deciding what action to ake given the state
	# p(s', r|, s, a) - the next state and reward given your action-state pair
	# we are only modeling p(a|s) uniform
	# how would the code change if p(s',r|s,a) is not deterministic?
	grid = standard_grid()

	# states will be positions (i, j)
	# simples than tic-tac-toe because we only have one "game piece"
	# that can only be at one position at a time
	states = grid.all_states() 

	## uniformly random actions ###
	# initialize V(s) = 0
	V = {} 
	for s in states:
		V[s] = 0 
	gamma = 1.0 # discount factor
	# repeat until convergence
	while True:
		biggest_change = 0 
		for s in states:
			old_v = V[s]

			#V(s) only has value if it's not a terminal state
			if s in grid.actions:
				new_v = 0 # we willl accumulate the answer
				p_a = 1.0 / len(grid.actions[s])	# each action has equal probability
				for a in grid.actions[s]:
					grid.set_state(s)
					r = grid.move(a)
					new_v += p_a * ( r + gamma * V[grid.current_state()]) 
			V[s] = new_v
			biggest_change = max( biggest_change, np.abs( old_v - V[s]))
		if biggest_change < SMALL_ENOUGHT:
			break
	print("values for uniformly random actions:")
	print_values(V, grid)
	print("\n\n")

	## fixed policy ##
	policy = {
		(2,0): 'U',
		(1,0): 'U',
		(0,0): 'R',
		(0,1): 'R',
		(0,2): 'R',
		(1,2): 'R',
		(2,1): 'R',
		(2,2): 'R',
		(2,3): 'U',
	}
	print_policy(policy, grid) 

	# initialize V(s) = 0 
	V = {} 
	for s in states:
		V[s] = 0 
	
	# let's see how V9s) changes as we get further away from the reward
	gamma = 0.9 # discount factor

	# repeat until convergence
	while True:
		biggest_change = 0
		for s in states:
			old_v = V[s]
			# V(s) only has value if it's not a terminal state
			if s in policy:
				a = policy[s]
				grid.set_state(s) 
				r = grid.move(a)
				V[s] = r + gamma * V[grid.current_state()] 
				biggest_change = max( biggest_change, np.abs( old_v - V[s]))
		if biggest_change < SMALL_ENOUGH:
			break
	print("values for fixed policy:")
	print_values(V, grid)
