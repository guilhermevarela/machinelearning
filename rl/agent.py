import numpy as np

LENGTH = 3

class Agent:
	def __init__(self, eps=0.1, alpha=0.5):
		self.eps = eps # probability of choosing random action instead of greedy
		self.alpha = alpha # learning rate
		self.verbose = False
		self.state_history = [] 

	def setV(self, V):
		self.V = V

	def set_symbol(self, sym):
		self.sym = sym

	def set_verbose(self, v):
		# if true will print values for each position on the board
		self.verbose = v

	def reset_history(self):
		self.state_history = []

	def take_action(self, env):
		# choose an action based on epislon-greedy strategy
		r = np.random.rand() 
		best_state = None
		if r < self.eps:
			# take a random action
			if self.verbose:
				print("Taking a random action") 

			possible_moves = [] 
			for i in range( LENGTH):
				for j in range( LENGTH ):
					if env.is_empty(i, j):
						possible_moves.append((i, j))
			
			idx = np.random.choice(len(possible_moves))
			next_move = possible_moves[idx]
		else:
			# choose the best action based on current values of states
			# loop through all possible moves, get their values
			# keep track of the best value 
			pos2value = {} # for debugging
			next_move = None
			best_value = -1
			for i in range( LENGTH ):
				for j in range( LENGTH ):
					if env.is_empty(i, j):
						# what is the state if we made that move?
						env.board[i, j] = self.sym
						state = env.get_state()
						env.board[i, j] = 0 # don't forget to change it back!
						pos2value[(i,j)] = self.V[state]
						if self.V[state] > best_value:
							best_value = self.V[state]
							best_state = state
							next_move = (i, j)
			# if verbose, draw the board w/ the values
			if self.verbose:
				print("Taking a greedy action")
				for i in range( LENGTH ):
					print('-' * 10)

					for j in range(LENGTH):
						if env.is_empty(i, j):
							# print the value
							print( " {0:.2f}|".format(pos2value[(i,j)]), end="")
						else:
							print(" ", end="")
							if env.board[i, j] == env.x:
								print("x |", end="")	
							elif env.board[i, j] == env.o:
								print("o |", end="")
							else:
								print(" |", end="")
					print("")
				print('-' * 10)

		env.board[next_move[0], next_move[1]]= self.sym

	def update(self, env):
		# we want to BACKTRACK over the states, so that:
		# V(prev_state) = V(prev_state) + alpha * ( V(next_state) - V(prev_state) )
		# where V(next_state) = reawrd if it's the most current state
		#
		# NOTE: we ONLY do this at the end of an episode
		# not so for all the algorithms wewill study
		reward = env.reward( self.sym )
		target = reward
		for prev in reversed(self.state_history):
			value = self.V[prev] + self.alpha * (target - self.V[prev])
			self.V[prev] = value
			target = value
		self.reset_history() 

	def update_state_history(self, state):
		# canot put this in take_action, because take_action only happens 
		#once every other iteratio for each player
		# state histoy needs to be updated every iteration
		# s = env.get_state() # don't want to do this twice so pass it in
		self.state_history.append(state)
	
