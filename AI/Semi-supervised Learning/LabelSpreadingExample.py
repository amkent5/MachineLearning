# Scikit-learns LabelSpreading method for semi-supervised learning

import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import LabelSpreading

label_propagation_model = LabelSpreading()
iris = datasets.load_iris()

'''
Unlabeled entries in y

It is important to assign an identifier to unlabeled points along
with the labeled data when training the model with the fit method. 
The identifier that this implementation uses is the integer value -1.
'''

# generate boolean matrix where less than 30% are 'True'
rand_numgen = np.random.RandomState(42)
random_unlabeled_points = rand_numgen.rand(len(iris['target'])) < 0.3

# create the unlabelled data in the labels (setting to -1)
full_labels = iris['target']
cutdown_labels = np.copy(iris['target'])
cutdown_labels[random_unlabeled_points] = -1

print full_labels
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
'''

print cutdown_labels
'''
[ 0  0  0  0 -1 -1 -1  0  0  0 -1  0  0 -1 -1 -1  0  0  0 -1  0 -1 -1  0  0
  0 -1  0  0 -1  0 -1 -1  0  0  0  0 -1  0  0 -1  0 -1  0 -1  0  0  0  0 -1
  1  1  1  1  1  1 -1 -1 -1  1  1 -1  1  1 -1  1 -1  1 -1  1  1 -1 -1  1  1
  1  1 -1  1 -1  1  1  1 -1  1  1  1  1  1  1 -1  1  1  1  1  1  1  1 -1 -1
 -1  2  2  2  2 -1  2  2 -1 -1 -1 -1  2  2  2  2  2 -1  2  2  2  2  2 -1 -1
  2  2  2 -1  2  2 -1 -1  2  2  2  2  2  2  2  2 -1  2  2 -1 -1  2  2 -1 -1]
'''

# fit LabelSpreading model
label_propagation_model.fit(iris['data'], cutdown_labels)

# quick test
print 'y: ', full_labels[-1]
print 'y_hat: ', label_propagation_model.predict(iris['data'][-1])
'''
y:  2
y_hat:  [2]
'''

# overall accuracy
correct = 0.0
for i in range(len(iris['data'])):

	if label_propagation_model.predict(iris['data'][i])[0] == full_labels[i]:
		correct += 1

print 'Overall accuracy: ', correct/ len(iris['data'])
'''
Overall accuracy:  0.98
'''