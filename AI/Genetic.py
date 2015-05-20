# Here I am going to try and define a nice generic genetic algorithm class which I can reuse.
# For simplicity I am going to define chromosones to be arrays.
# The fitness function returns zero if the solution is optimal otherwise another positive integer indicating distance from solution.

from random import *
from copy import *
import math

CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.01
VERY_UNFIT = 0 # Worst possible fitness.
PERFECT_FITNESS = 1 # Best possible fitness.

class GeneticSolver:

    # Wrapper for pairs.
    class ChromosoneFitnessPair:
        def __init__(self, chromosone, fitness):
            self.chromosone = chromosone
            self.fitness = fitness

    def __init__(self, randomBase, randomChromosone, fitnessFunc):
        # Store the functions that we'll need for the current problem.
        self.randomBase = randomBase
        self.randomChromosone = randomChromosone
        self.fitnessFunc = fitnessFunc

        self.rouletteSize = 10000
        self.currentPopulation = []

    def findOptimalChromosone(self, popSize, maxGenerations):
        # Keep a record of the best so far.
        bestChrom = None
        bestFitness = VERY_UNFIT

        # Iterate through the generations.
        for gen in range(0, maxGenerations):
            chrom = self.getNextPopulation(popSize)
            fitness = self.fitnessFunc(chrom)
            if fitness > bestFitness:
                bestFitness = fitness
                bestChrom = chrom

        return bestChrom

    def getNextPopulation(self, popSize):
        # Initialise the population.
        if len(self.currentPopulation) == 0:
            self.currentPopulation = [self.randomChromosone() for i in range(0, popSize)]

        fitnessPairs = []

        # Apply mutation.
        for pop in self.currentPopulation:
            self.doMutation(pop)

        # Compute the chromosone pairs.
        fitnessSum = 0
        for chromosone in self.currentPopulation:
            fitness = self.fitnessFunc(chromosone)

            # Check for perfect solution.
            if fitness == PERFECT_FITNESS:
                return chromosone

            fitnessSum += fitness
            fitnessPairs.append(self.ChromosoneFitnessPair(chromosone, fitness))

        # Use roulette style selection for create the next population.
        currentRouletteIndex = -1
        rouletteDict = {}
        for pair in fitnessPairs:
            # Compute the weight of this pair (which translates to a number of elements on the roulette wheel for this pair).
            weight = pair.fitness / fitnessSum
            numEltsOnWheel = int(math.ceil(weight * self.rouletteSize)) + 1
            for i in range(0, numEltsOnWheel):
                currentRouletteIndex += 1
                rouletteDict[currentRouletteIndex] = pair
            
        # We have created out roulette wheel, now select the next generation!
        newPopulation = []
        while len(newPopulation) < popSize:
            # Choose two elements at random from the wheel (though they are weighted as some chromosones have more elements).
            r1 = randint(0, self.rouletteSize)
            r2 = randint(0, self.rouletteSize)
            pair1 = rouletteDict[r1]
            pair2 = rouletteDict[r2]

            if random() < CROSSOVER_RATE:
                # Perform crossover.
                newPopulation.extend(self.doCrossOver(pair1.chromosone, pair2.chromosone))
            else:
                # Just carry them across.
                newPopulation.append(pair1.chromosone)
                newPopulation.append(pair2.chromosone)

        # Compute average fitness of the current population.
        total = sum([self.fitnessFunc(popElt) for popElt in newPopulation])
        print 'mean fitness', (total + .0) / popSize

        #print newPopulation
        self.currentPopulation = deepcopy(newPopulation)

        # What is the fittest chromosone we've found?
        bestFit = VERY_UNFIT
        bestChrom = []
        for pair in fitnessPairs:
            if pair.fitness > bestFit:
                bestFit = pair.fitness
                bestChrom = pair.chromosone

        return bestChrom

    def doCrossOver(self, chrom1, chrom2):
        newChrom1 = []
        newChrom2 = []
        for i in range(0, len(chrom1)):
            # During crossover we have a 50% chance of swapping each base.
            if random() < 0.5:
                # Swap around these bases.
                newChrom2.append(chrom1[i])
                newChrom1.append(chrom2[i])
            else:
                # Keep the same way around.
                newChrom1.append(chrom1[i])
                newChrom2.append(chrom2[i])
        return [newChrom1, newChrom2]
                
    def doMutation(self, chrom1):
        for i in range(0, len(chrom1)):
            if random() < MUTATION_RATE:
                # Do a muation.
                chrom1[i] = self.randomBase(i)
