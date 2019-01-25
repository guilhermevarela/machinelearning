# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
# Simple reinforcement learning algorithm for learning tic-tac-toe
# Use the update rule: V(s) = V(s) + alpha*(V(s') - V(s))
# Use the epsilon-greedy policy:
#   action|s = argmax[over all actions possible from state s]{ V(s) }  if rand > epsilon
#   action|s = select random action from possible actions from state s if rand < epsilon
#
#
# INTERESTING THINGS TO TRY:
#
# Currently, both agents use the same learning strategy while they play against each other.
# What if they have different learning rates?
# What if they have different epsilons? (probability of exploring)
#   Who will converge faster?
# What if one agent doesn't learn at all?
#   Poses an interesting philosophical question: If there's no one around to challenge you,
#   can you reach your maximum potential?import numpy as np

import numpy as np

from human import Human
from agent import Agent
from environment import Environment

LENGTH = 3

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

def play_game( p1, p2, env, draw=False):
	# loops until the game is over
	current_player = None
	while not env.game_over():
		# alternate between players
		# p1 always starts first 
		if current_player == p1:
			current_player = p2
		else:
			current_player = p1
		
		# draw the board before the user who wants to see it makes a move
		if draw:
			if draw ==1 and current_player == p1:
				env.draw_board()
			
			if draw ==2 and current_player == p2:
				env.draw_board()
		#current player makes a move
		current_player.take_action(env) 

		#update state histories
		state = env.get_state() 
		p1.update_state_history(state)
		p2.update_state_history(state)

	if draw:
		env.draw_board() 

	# do the value function update
	p1.update(env)
	p2.update(env)

if __name__ == '__main__':
	# train the agent
	p1 = Agent()
	p2 = Agent() 

	# set initial V for p1 and p2
	env = Environment()
	state_winner_triples = get_state_hash_and_winner(env)

	Vx = initialV_x(env, state_winner_triples)
	p1.setV(Vx)
	Vo = initialV_o(env, state_winner_triples)
	p2.setV(Vo)

	#give each player their symbol
	p1.set_symbol(env.x)
	p2.set_symbol(env.o)
	
	T = 10000
	for t in range( T ):
		if t % 200 == 0:
			print(t)
		play_game(p1, p2, Environment())

	# play human vs. agent 
	# do you think the agent learned to play the game well?
	human = Human()
	human.set_symbol(env.o)
	while True:
		p1.set_verbose(True)
		play_game(p1, human, Environment(), draw=2)
		# Player 1 always start the game -- does agent chooses the middle?
		# Have the human start by passing the human as first parameter
		answer = input("Play again? [Y/n]: ")
		if answer and answer.lower()[0] == 'n':
			break
