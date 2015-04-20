# Experiment with nearest neighbour classification on the Seeds dataset 
# With a training dataset of 210 examples we achieve a 90.48% 

import numpy as np

# create features array
data = np.genfromtxt('/Users/Ash/Projects/Building Machine Learning Systems/1400OS_02_Codes/data/seeds.tsv')
features = np.ma.compress_cols(np.ma.masked_invalid(data))

# create targets array
targets = np.array(['Kama'])
for i, row in enumerate(open('/Users/Ash/Projects/Building Machine Learning Systems/1400OS_02_Codes/data/seeds.tsv')):
	if i > 0:
		targets = np.vstack((targets, row.replace('\n', '').split('\t')[7]))

'''
Now write functions 
and do plots :) 
'''
# p0 and p1 are vectors with 7 elements (the features)
# this generates a distance based on all 7 features! 
def distance(p0, p1):
	return np.sum((p0-p1)**2)

def nn_classify(training_set, training_labels, new_example):

	# generate a list of vector differences 
	l = []
	for x in training_set:
		l.append(distance(x, new_example))

	# return the label that fits the closest (nearest neighbour)
	for i in range(len(l)):
		if l[i] == min(l):
			return training_labels[i]

# "leave one out" testing - run the test 210 times
# (most extreme form of cross-validation)
iter_count = 0
success = 0
for i in range(len(features)):
	iter_count += 1

	# remove one example 
	trainingFeatures = np.delete(features, i, 0)
	trainingLabels = np.delete(targets, i, 0)

	# store the "new" data for validation
	new_example_feature = features[i]
	new_example_label = targets[i]

	if nn_classify(trainingFeatures, trainingLabels, new_example_feature) == new_example_label:
		success += 1

print str(round((success/ float(iter_count))*100,2)) + '% accuracy'

