import numpy as np

def get_state_hash_and_winner(env, i=0, j=0):
	results = [] 
	
	for v in (0, env.x, env.o):
		env.board[i,j] = v # if empty board it should already be 0
		if j ==2:
			# j goes back to 0, increase i, unless i=2, then we are done
			if i == 2:
				# the board s full, collect results and return
				state = env.get_state() 
				ended = env.game_over(force_recalculate=True) 
				winner = env.winner
				results.append((state, winner, ended))
			else:
				results += get_state_hash_and_winner(env, i + 1, 0)
		else:
			# increment j, i  stays the same
			results += get_state_hash_and_winner(env, i, j + 1)
	return results

def initialV_x(env, state_winner_triples):
	# initialize state values as follows
	# if x wins, V(s) = 1
	# if x loses or draw, V(s) = 0
	# otherwise, V(s) = 0.5
	V = np.zeros(env.num_states) 
	for state, winner, ended in state_winner_triples:
		if ended:
			if winner == env.x:
				v = 1
			else:
				v = 0
		else:
			v = 0.5
		V[state] = v 
	return V

def initialV_o(env, state_winner_triples):
	# initialize state values as follows
	# if o wins, V(s) = 1
	# if o loses or draw, V(s) = 0
	# otherwise, V(s) = 0.5
	V = np.zeros(env.num_states) 
	for state, winner, ended in state_winner_triples:
		if ended:
			if winner == env.o:
				v = 1
			else:
				v = 0
		else:
			v = 0.5
		V[state] = v 
	return V
