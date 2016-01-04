# Neural Net (no hidden layer) for boiler data

import random

# class describing single level neural net 
class neural:
	def __init__(self):
		self.rate = 0.01	# how fast the weights change
		self.weights = []	# internal weights

	# get the output for these inputs with the current weights
	def evaluate(self, inputs):
		s = 0
		for i in range(0, len(inputs)):
			s += self.weights[i] * inputs[i]
		return 0 if s < 0.5 else 1	# could use a sigmoid or tanh function here

	# trains network based on input data
	def train(self, data):
		# init weights
		# how many weights do we need?
		numWeights = 0
		for d in data:
			if len(d) > numWeights:
				numWeights = len(d)

		self.weights = [random.random() for i in range(0, numWeights-1)]

		for i in range(0, 100):	# go through 100 times
			for test in data:	# tune weights using data
				inputs = test[:-1]	# ignore last elt
				out = self.evaluate(inputs)

				for j in range(0, len(self.weights)):
					
					'''if the output is incorrect then adjust the weight (using the rate value)
					comparing evaluation to real value'''
					self.weights[j] += self.rate * (test[-1] - out)

				print self.weights

if __name__ == "__main__":
	# read in data
	f = open("test.txt", "r")

	data = []
	for line in f:
		dataSet = []
		line = line.rstrip()
		elts = line.split(" ")
		for e in elts:
			dataSet.append(float(e))
		data.append(dataSet)

	f.close()

	data = data[:10]

	# test network
	n = neural()
	n.train(data)

	# compare with actual values to see how good the fit is
	numRight = 0
	for d in data:
		actual = d[-1]
		computed = n.evaluate(d[:-1])
		if actual == computed:
			numRight += 1

	print "Proportion correct:", float(numRight)/ len(data)



