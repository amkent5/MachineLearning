# minimax AI noughts and crosses

"""
# recursion revisit
def factorial(n):

	print 'n: ', n
	if n == 1:
		return 1
	else:
		return n * factorial(n - 1)
#print factorial(5)
"""

import pygame
import random
from time import time

# function for finding available spaces to move
def availableIXs(grid):
	return [ix for ix, elt in enumerate(grid) if elt == '-']

# check for a draw
def drawCheck(grid):
	draw = True
	for elt in grid:
		if elt == '-':
			draw = False
	if draw == True:
		return 1

# check for a win
def winCheck(grid, player):

	curr_pos = [ix for ix, elt in enumerate(grid) if elt == player]
	if len(curr_pos) < 3:
		return 0

	for win in winning_grid_ixs:
		if all(pos in curr_pos for pos in win):
			return 1

	return 0

# Implementation of recursive minimax algorithm
# http://neverstopbuilding.com/minimax
#
# Returns vals
# willWin: 	1  -> 'X' is in the winning state
#			-1 -> 'O' is in the winning state
#			0 the game is a draw
#
# nextMove: index of the next move
def minimax_nextMove(grid, player):

	print grid

	# if it is the first move, always go centre
	if len(availableIXs(grid)) == 9:
		return 0, 4

	# control switching of player for max/ min calcs on each recursion
	if player == 'X':
		nextPlayer = 'O'
	else:
		nextPlayer = 'X'

	# check for winning end states
	if player == 'X' and winCheck(grid, 'O') == 1:
		return -1, -1 	# 'O' has won
	if player == 'O' and winCheck(grid, 'X') == 1:
		return 1, -1 	# 'X' has won

	# check for drawing end state
	if drawCheck(grid) == 1:
		return 0, -1

	# create list to append scores for each state
	res_states = []

	# iterate through each of the available IX's/ states and call minimax on them
	available_ixs = availableIXs(grid)
	for ix in available_ixs:

		print ix

		# move into the index
		grid[ix] = player

		# play out the algorithm from this grid 'state'
		outcome, move = minimax_nextMove(grid, nextPlayer)

		print 'Debug'
		print outcome, move

		# for the possible move, store the outcome
		res_states.append(outcome)

		print res_states

		# set grid back to original for next iteration
		grid[ix] = '-'

	# return the best move for either 'X' or 'O' based
	# on the max or min of the results list
	if player == 'X':
		max_elt = max(res_states)
		return max_elt, available_ixs[res_states.index(max_elt)]
	else:
		min_elt = min(res_states)
		return min_elt, available_ixs[res_states.index(min_elt)]

def printGrid(grid):
	print grid[0], grid[1], grid[2]
	print grid[3], grid[4], grid[5]
	print grid[6], grid[7], grid[8]
	print '\n'

grid = ['-', '-', '-',
		'-', '-', '-',
		'-', '-', '-']

winning_grid_ixs = [[0, 1, 2], [3, 4, 5], [6, 7, 8],	# horizontals
					[0, 3, 6], [1, 4, 7], [2, 5, 8],	# verticals
					[0, 4, 8], [2, 4, 6]]				# diagonals

# test minimax algorithm
#print minimax_nextMove(list('XX-OO-XO-'),'O')	# should return (-1, 5)
#quit()

# pygame main
pygame.init()
fps = 100 # ms
run = True
human_turn = True

while run:

	if human_turn:
		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN:

				player_move = event.key - 49

				# update board
				grid[player_move] = 'O'

				# print board
				printGrid(grid)
				
				# check to see if won
				if winCheck(grid, 'O'):
					print 'You have won!'
					run = False
					break

				# check for draw
				if drawCheck(grid) == 1:
					print 'It''s a draw!'
					run = False
					break

				# give AI control
				human_turn = False


	else:
		# check available IX's and choose one at random
		#ai_move = random.choice(availableIXs(grid))

		# implement minimax algorithm
		_w, ai_move = minimax_nextMove(grid, 'X')

		# update board
		grid[ai_move] = 'X'

		# print board
		printGrid(grid)

		# check to see if won
		if winCheck(grid, 'X'):
			print 'You have lost!'
			run = False
			break

		# check for draw
		if drawCheck(grid) == 1:
			print 'It''s a draw!'
			run = False
			break

		# pass control back to human
		human_turn = True

	pygame.time.wait(fps)





