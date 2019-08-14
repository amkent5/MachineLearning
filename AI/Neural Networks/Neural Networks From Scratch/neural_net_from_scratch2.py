"""
Need to add a delta key to the input and hidden dicts
As we propagate the error backward we will store it in this dict
"""

### Neural networks from scratch
### http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html


# number_epoch:			number of times the entire dataset is ran through the network
# batch_size:			number of rows that forward and backpropagated before the weights are saved
# activation function:	function that activates or deactivates each neuron

import math
import random
import numpy as np



# the input layer is the first hidden layer!!!!!!!!
def create_input_layer(network, input_dims, num_neurons):	# creates weights matrix and values matrix
	d_input_layer = {'values': [], 'weights': []}

	# add placeholders for neuron values
	d_input_layer['values'] = [ 0 for _ in range(num_neurons) ]

	# randomly assign weights for weight matrix
	# the matrix dims are based on the number of inputs to the network and the number
	# of neurons in the first layer
	for i in range(input_dims):
		weights_row = [ round(np.random.uniform(low=-0.05, high=0.05), 4) for _ in range(num_neurons) ]
		d_input_layer['weights'].append(weights_row)

	network.append(d_input_layer)
	return network

def create_hidden_layer(network, input_dims, num_neurons):		# creates weights matrix and values matrix
	d_hidden_layer = {'values': [], 'weights': []}

	# add placeholders for neuron values
	d_hidden_layer['values'] = [ 0 for _ in range(num_neurons) ]

	# assign weights matrix
	for i in range(input_dims):
		weights_row = [ round(np.random.uniform(low=-0.05, high=0.05), 4) for _ in range(num_neurons) ]
		d_hidden_layer['weights'].append(weights_row)

	network.append(d_hidden_layer)
	return network

def create_output_layer(network, input_dims, num_outputs):		# creates weights matrix and values matrix

	d_output_layer = {'values': [], 'weights': []}

	# add placeholders for neuron values
	d_output_layer['values'] = [ 0 for _ in range(num_outputs) ]

	# assign weights matrix
	for i in range(input_dims):
		weights_row = [ round(np.random.uniform(low=-0.05, high=0.05), 4) for _ in range(num_outputs) ]
		d_output_layer['weights'].append(weights_row)

	network.append(d_output_layer)
	return network

### Usage
"""
# create a network which has 3 inputs, 3 hidden layers (well 1 input 2 hidden) of 5 neurons
# each, and 1 output in the output layer
network = []
print network
create_input_layer(network, input_dims=3, num_neurons=5)
create_hidden_layer(network, input_dims=5, num_neurons=5)
create_hidden_layer(network, input_dims=5, num_neurons=5)
create_output_layer(network, input_dims=5, num_outputs=1)
print network
"""

# create a more simple network for testing
network = []
create_input_layer(network, input_dims=5, num_neurons=3)
create_hidden_layer(network, input_dims=3, num_neurons=3)
create_output_layer(network, input_dims=3, num_outputs=1)
print (network)

### Neuron function
"""
### example of how this should work:
inp =[2, 3, 2, 3]
ws = [[1, 0, 1], [2, 2, 2], [2, 3, 1], [0, 1, 0]]
print inp
print ws

res = []
for i in range(len(inp)):
	res.append([0 for _ in range(len(ws[0]))])

print res

for i, elt in enumerate(inp):
	for j in range(len(ws[0])):
		print i, j
		res[i][j] = elt * ws[i][j]
		print res
		print '\n'

print res
quit()"""
# returns a list of the entire layers new neuron values
def update_neuron_values(inputs, weights, activation_function):
	"""
	inputs = [...]
	weights = [[...], [...], ..., [...]]
	"""

	# init matrix product
	matrix_prod = []
	for i in range(len(inputs)):
		matrix_prod.append( [0 for _ in range(len(weights[0]))] )

	# perform matrix multiplication
	for i, elt in enumerate(inputs):
		for j in range(len(weights[0])):
			matrix_prod[i][j] = elt * weights[i][j]

	# sum up columns of matrix_prod (these are our weighted sums)
	weighted_sums = []
	for col in range(len(matrix_prod[0])):
		col_sum = 0
		for row in range(len(matrix_prod)):
			col_sum += matrix_prod[row][col]
		weighted_sums.append(col_sum)

	# now apply activation function to each weighted sum
	neuron_values = []
	if activation_function == 'sigmoid':
		for val in weighted_sums:
			neuron_values.append( round(1 / ( 1 + math.exp(-val)), 4) )

	if activation_function == 'relu':
		for val in weighted_sums:
			neuron_values.append( round(max(0, val), 4) )

	if activation_function == 'tanh':
		for val in weighted_sums:
			neuron_values.append( round(math.tanh(val), 4) )

	return neuron_values



# backprop!
def backpropagate_like_a_motherfucker():
	return

# main
def run_network(network, input_X, input_y, number_epoch, batch_size):

	i = 0
	while i <= number_epoch:
		i += 1
		for j, row in enumerate(input_X):

			#if j % batch_size == 0:
				# save weights and perform backpropagation

			for k in range(len(network)):
				if k == 0:
					network[k]['values'] = update_neuron_values(row, network[k]['weights'], activation_function='relu')
				else:
					network[k]['values'] = update_neuron_values(network[k-1]['values'], network[k]['weights'], activation_function='relu')

			# calculate the delta from actual
			delta = abs( input_y[j] - network[-1]['values'][0] )

			# print some output (first hidden layers neuron values)
			print ( network[0]['values'], '\t', network[1]['values'], '\t', network[2]['values'], '\t', delta )




### let's see progress (it's working!)
# gen data
X = []
y = []
for i in range(10000):
	X.append( [ round(random.random(), 0) for _ in range(5) ] )
	y.append( round(random.random(), 0) )
for row in X: print (row)

run_network(network, X, y, 5, 5)
