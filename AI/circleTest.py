# It's another Mandelbrot set generating program!

from __future__ import division
import pygame
from random import *
import math
from Genetic import *

width = 800
height = 600

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

MIN_CIRCLE_RADIUS = 10
MAX_CIRCLE_RADIUS = 20
NUM_CIRCLES = 30

circles = []

# CIRCLE FUNCTIONS
class Circle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def draw(self, screen, colour):
        pygame.draw.circle(screen, colour, (self.x, self.y), self.r, 1)

def generateRandomCircle(radiuslower, radiusupper):
    r = randint(radiuslower, radiusupper)
    x = randint(r, width - r)
    y = randint(r, height - r)
    return Circle(x, y, r)

def drawCircles(screen, colour = RED):
    for circle in circles:
        circle.draw(screen, colour)

def circlesIntersect(c1, c2):
    circleCentreDistSq = abs(c1.x - c2.x) ** 2 + abs(c1.y - c2.y) ** 2
    return circleCentreDistSq < (c1.r + c2.r) ** 2
# END CIRCLE FUNCTIONS

# GENETIC ALGORITHM FUNCTIONS
# Chromosone format:
# [x, y, r]
def randomBase(i):
    c = generateRandomCircle(10, height / 2)
    return [c.x, c.y, c.r][i]

def randomChromosone():
    c = generateRandomCircle(10, height / 2)
    return [c.x, c.y, c.r]

def fitnessFunc(chromCircle):
    # Try to map a circle to a fitness value between 0 and 1 (where 1 is optimal and 0 is the worst possible).
    # Circles are ranked better if they do not intersect with many circles and have a large radius.
    # First count intersections with other circles.
    circle = Circle(chromCircle[0], chromCircle[1], chromCircle[2])
    
    # Is this circle actually valid (i.e. on the screen)?
    if (circle.x - circle.r < 0) or (circle.x + circle.r > width) or (circle.y - circle.r < 0) or (circle.y + circle.r > height):
        return VERY_UNFIT

    intersectionCount = 0
    for fixedCircle in circles:
        if circlesIntersect(circle, fixedCircle):
            intersectionCount += 1
    intersectionCount += 1 # Avoid division by zero.

    radiusProportion = float(circle.r) / height
    return radiusProportion / intersectionCount
# END GENETIC ALGORITHM FUNCTIONS

screen = pygame.display.set_mode((width, height))
screen.fill(BLACK)

# Initialise the circles that we need to avoid.
while len(circles) < NUM_CIRCLES:
    rcircle = generateRandomCircle(MIN_CIRCLE_RADIUS, MAX_CIRCLE_RADIUS)
    # Does this circle intersect with any of the others.
    intersect = False
    for c in circles:
        if circlesIntersect(rcircle, c):
            intersect = True
            break
    if intersect: continue # Intersect, try again.
    circles.append(rcircle)

drawCircles(screen)

# Apply GA.
solver = GeneticSolver(randomBase, randomChromosone, fitnessFunc)
bestEverFitness = VERY_UNFIT
bestEverCircle = None

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    bestChrom = solver.getNextPopulation(1000)

    bestCircle = Circle(bestChrom[0], bestChrom[1], bestChrom[2])

    fitness = fitnessFunc(bestChrom)
    if fitness > bestEverFitness:
        bestEverFitness = fitness
        bestEverCircle = bestCircle

    screen.fill(BLACK)

    drawCircles(screen)
    bestCircle.draw(screen, GREEN)
    bestEverCircle.draw(screen, BLUE)

    pygame.display.flip()

    #pygame.time.wait(50)

    

