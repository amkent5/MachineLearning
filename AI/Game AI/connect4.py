# Connect 4 game AI

# board is 7 (horiz) * 6 (verical) = 42 positions


# put these functions in a connect4 class
def defineBoard():
	board = [
		['-', '-', '-', '-', '-', '-', '-'],
		['-', '-', '-', '-', '-', '-', '-'],
		['-', '-', '-', '-', '-', '-', '-'],
		['-', '-', '-', '-', '-', '-', '-'],
		['-', '-', '-', '-', '-', '-', '-'],
		['-', '-', '-', '-', '-', '-', '-']
		]
	return board

def printBoard(board):
	for row in board:
		print row
		print '\n'

def checkDraw(board):
	draw = True
	for row in board:
		for elt in row:
			if elt == '-':
				draw = False
	if draw == True:
		return 1
	else:
		return 0

def checkWin(board, player):
	# will be slightly harder as cant store all winning positions

	# check horizontal wins
	if player == 'X':
		# check for 4 contiguous X's in the board rows
		for row in board:
			c = 0
			for elt in row:
				if elt == 'X':
					c += 1
				else:
					c = 0
				if c == 4:
					return 1


	# check vertical wins

	# check diagonal wins


def miniMax():



board = defineBoard()
printBoard(board)