import random
import numpy as np
from sklearn import svm
from sklearn.datasets import load_iris

data = load_iris()

features = data['data']
targets = data['target']
target_names = data['target_names']


# lets create a training set of data (take 75 random rows of the data)
# We will create X - a training set of feature information
# and y - the corresponding (training) set of target data for this subset of features
l_random = [random.randint(1, 149) for r in range(75)]
X = features[0]
for rand in l_random:
	X = np.vstack((X, features[rand]))
y = targets[0]
for rand in l_random:
	y = np.hstack((y, targets[rand]))

# use a support vector machine as our classifier
classifier = svm.SVC()

# now we fit our classifier to the model - this is done 
# by passing our training data set to the fit method. 
classifier.fit(X, y)

# now lets test our classifier
print classifier.predict(features[5]), targets[5]	# what is the sample species based on its 4 features?
print classifier.predict(features[7]), targets[7]
print classifier.predict(features[46]), targets[46]
print classifier.predict(features[115]), targets[115]
print classifier.predict(features[132]), targets[132]	# passes with flying colours!

# passed with about 98% accuracy 
for i in range(1, 150):
	print classifier.predict(features[i]), targets[i]