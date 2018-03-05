# Snake game with AI snake sprite

'''
Next steps
	- AI snake
		- write a function like in savePrincess2 and apply it to the second snake
		- investigate A* path finding
		- investigate A* path finding avoiding moving target
'''

import pygame
import random
import math as m
from time import time

# class for drawing blocks on screen
class blocks:
	def __init__(self, x, y, width, height):
		self.x = x
		self.y = y
		self.width = width
		self.height = height
	def drawBlock(self, screen, colour):
		s = pygame.Surface((self.width, self.height))
		s.fill(colour)
		screen.blit(s, (self.x, self.y))
	def eraseBlock(self, screen):
		col = (0, 0, 0)
		s = pygame.Surface((self.width, self.height))
		s.fill(col)
		screen.blit(s, (self.x, self.y))

# function for updating snake for general movement
def updateSnake(snakeArray, x, y):
	# add new element to head of snake based on new position
	snakeArray.append(blocks(x, y, blockWidth, blockHeight))
	# erase element to tail end of snake and then remove
	snakeArray[0].eraseBlock(screen)
	snakeArray.pop(0)

# function for extendP1ing snake for food eating
def extendSnake(snakeArray, x, y):
	snakeArray.append(blocks(x, y, blockWidth, blockHeight))

# function to round a number to a given base (used in random generation of coords for food)
def myRound(x, base):
	return int(base * round(float(x)/ base))

# simple path finder
def updatePath(currX, currY, targX, targY, prevMove, prevDict):

	print 'prevMove: ', prevMove

	# princess algorithm
	if targX > currX and prevMove != 'l':	# need to move right
		d = {'r': True, 'l': False, 'u': False, 'd': False}
		return d
	if targX < currX and prevMove != 'r':	# need to move left
		d = {'r': False, 'l': True, 'u': False, 'd': False}
		return d
	if targY > currY and prevMove != 'u':	# need to move down
		d = {'r': False, 'l': False, 'u': False, 'd': True}
		return d
	if targY < currY and prevMove != 'd':	# need to move up
		d = {'r': False, 'l': False, 'u': True, 'd': False}
		return d
	return prevDict


# pygame inits
fps = 50
width = 1100
height = 800
pygame.init()
screen = pygame.display.set_mode((width, height))

# game inits
moveRightP1 = True
moveDownP1 = False
moveLeftP1 = False
moveUpP1 = False

moveRightP2 = True
moveDownP2 = False
moveLeftP2 = False
moveUpP2 = False

foodWait = 0
spawnFood = True
extendP1 = False
extendP2 = False
prevAImove = 'r'
prevPathDict = {'r': True, 'l': False, 'u': False, 'd': False}

# the snake is just an array of blocks to draw
blockWidth = 20
blockHeight = 20
startingNumBlocks = 10
l_snake = []
l_snake2 = []
for x in range(0, startingNumBlocks * blockHeight, blockHeight):
	l_snake.append(blocks(x+100, 200, blockWidth, blockHeight))
	l_snake2.append(blocks(x+100, 800, blockWidth, blockHeight))



running = True
while running:

	foodWait += 1
	if foodWait > 40:
		if spawnFood:
			foodX = myRound(random.randint(0, width - blockWidth), blockWidth)
			foodY = myRound(random.randint(0, height - blockHeight), blockWidth)
			food = blocks(foodX, foodY, blockWidth, blockHeight)
			food.drawBlock(screen, (0, 255, 0))
			spawnFood = False
		else:
			if l_snake[-1].x == food.x and l_snake[-1].y == food.y:
				spawnFood = True
				extendP1 = True
			if l_snake2[-1].x == food.x and l_snake2[-1].y == food.y:
				spawnFood = True
				extendP2 = True

	currXP1 = l_snake[-1].x
	currYP1 = l_snake[-1].y

	currXP2 = l_snake2[-1].x
	currYP2 = l_snake2[-1].y

	prevXP1 = l_snake[-2].x
	prevYP1 = l_snake[-2].y

	prevXP2 = l_snake2[-2].x
	prevYP2 = l_snake2[-2].y

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

		if event.type == pygame.KEYDOWN:
			
			# P1
			if event.key == pygame.K_RIGHT and currXP1 + blockWidth != prevXP1:
				moveRightP1, moveLeftP1, moveDownP1, moveUpP1 = True, False, False, False
			if event.key == pygame.K_DOWN and currYP1 + blockHeight != prevYP1:
				moveDownP1, moveUpP1, moveRightP1, moveLeftP1 = True, False, False, False
			if event.key == pygame.K_LEFT and currXP1 - blockWidth != prevXP1:
				moveLeftP1, moveDownP1, moveUpP1, moveRightP1 = True, False, False, False
			if event.key == pygame.K_UP and currYP1 - blockHeight != prevYP1:
				moveUpP1, moveRightP1, moveLeftP1, moveDownP1 = True, False, False, False

	# respond to keystrokes
	# P1
	if moveRightP1: currXP1 += blockWidth
	if moveDownP1: currYP1 += blockWidth
	if moveLeftP1: currXP1 -= blockWidth
	if moveUpP1: currYP1 -= blockWidth

	''' AI movement '''
	if foodWait > 40:
		path_dict = updatePath(currXP2, currYP2, food.x, food.y, prevAImove, prevPathDict)
		print path_dict
		prevPathDict = path_dict
		for elt in path_dict.items():
			if elt[1] == True:
				if elt[0] == 'r':
					prevAImove = 'r'
					moveRightP2, moveLeftP2, moveDownP2, moveUpP2 = True, False, False, False
				elif elt[0] == 'd':
					prevAImove = 'd'
					moveDownP2, moveUpP2, moveRightP2, moveLeftP2 = True, False, False, False
				elif elt[0] == 'l':
					prevAImove = 'l'
					moveLeftP2, moveDownP2, moveUpP2, moveRightP2 = True, False, False, False
				else:
					prevAImove = 'u'
					moveUpP2, moveRightP2, moveLeftP2, moveDownP2 = True, False, False, False

	# AI response vectors
	if moveRightP2: currXP2 += blockWidth
	if moveDownP2: currYP2 += blockWidth
	if moveLeftP2: currXP2 -= blockWidth
	if moveUpP2: currYP2 -= blockWidth

	# deal with edges
	# P1
	if currYP1 < 0:
		currYP1 = height - blockHeight
	if currYP1 > height - blockHeight:
		currYP1 = 0
	if currXP1 < 0:
		currXP1 = width - blockWidth
	if currXP1 > width - blockWidth:
		currXP1 = 0

	# P2
	if currYP2 < 0:
		currYP2 = height - blockHeight
	if currYP2 > height - blockHeight:
		currYP2 = 0
	if currXP2 < 0:
		currXP2 = width - blockWidth
	if currXP2 > width - blockWidth:
		currXP2 = 0

	for block in l_snake:
		# P1 hitting himself
		if block.x == currXP1 and block.y == currYP1:
			pygame.time.wait(1000)
			running = False
		
		# P2 hitting P1...
		if block.x == currXP2 and block.y == currYP2:
			pygame.time.wait(1000)
			removeUpToIX = len(l_snake2) - 5
			currXP2 = 100
			currYP2 = 800
			moveRightP2 = True
			moveDownP2 = False
			moveLeftP2 = False
			moveUpP2 = False
			deadSection = l_snake2[:removeUpToIX]
			for b in deadSection:
				b.eraseBlock(screen)
			l_snake2 = l_snake2[removeUpToIX:]

	for block in l_snake2:
		# P2 hitting himself
		if block.x == currXP2 and block.y == currYP2:
			#pygame.time.wait(1000)
			#running = False

			''' wrap this code block in a function '''
			pygame.time.wait(1000)
			removeUpToIX = len(l_snake2) - 5
			currXP2 = 100
			currYP2 = 300
			moveRightP2 = True
			moveDownP2 = False
			moveLeftP2 = False
			moveUpP2 = False
			deadSection = l_snake2[:removeUpToIX]
			for b in deadSection:
				b.eraseBlock(screen)
			l_snake2 = l_snake2[removeUpToIX:]


		# P1 hitting P2...
		if block.x == currXP1 and block.y == currYP1:
			pygame.time.wait(1000)
			removeUpToIX = len(l_snake) - 5
			currXP1 = 100
			currYP1 = 800
			moveRightP1 = True
			moveDownP1 = False
			moveLeftP1 = False
			moveUpP1 = False
			deadSection = l_snake[:removeUpToIX]
			for b in deadSection:
				b.eraseBlock(screen)
			l_snake = l_snake[removeUpToIX:]


	if extendP1:
		extendSnake(l_snake, currXP1, currYP1)
		extendP1 = False
	else:
		updateSnake(l_snake, currXP1, currYP1)
	
	if extendP2:
		extendSnake(l_snake2, currXP2, currYP2)
		extendP2 = False
	else:
		updateSnake(l_snake2, currXP2, currYP2)


	for block in l_snake:
		block.drawBlock(screen, (255, 0, 0))
	for block in l_snake2:
		block.drawBlock(screen, (0, 0, 255))


	# refresh screen
	pygame.time.wait(fps)
	pygame.display.flip()
