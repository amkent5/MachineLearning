"""
Mutual information (MI) between two random variables is a non-negative value, which measures the dependency between the variables. 
It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.

In a feature selection context we can view the MI as the "amount of information" held in each feature. 
The implication being that you should include features that hold a lot of information, and possibly drop features from your dataset 
that hold little to no information describing the output as they will not be useful to the algorithm, could add noise and therefore 
difficulty in isolating the predictive signal, and will just make training time slower.

As our features in this probelm are discrete, I will use Scikit-learn’s mutual_info_classif class with the discrete_features=True flag:
"""

import numpy as np
from sklearn.feature_selection import mutual_info_classif
import pandas as pd

datafile = '/model_data/training_data_newfeatures.csv'
data = pd.read_csv(datafile, delimiter='\t', dtype=str)
l_features = list(data.columns)

discrete_dataset = np.loadtxt(datafile, dtype=str, delimiter='\t')
X = discrete_dataset[:, :-1]
y = discrete_dataset[:, -1]

l_importance = mutual_info_classif(X, y, discrete_features=True)

resdict = {}
for i, res in enumerate(l_importance):
	
	# exclude data which is 1:1 with the job feature (as these will not be useful)
	if l_features[i] in ['... list of features ...']:
		continue
	resdict[l_features[i]] = res
print 'MI:'
for elt in sorted(resdict.items(), key=lambda x: x[1], reverse=True):
    print elt

print '\n'

resdict2 = {}
for i, res in enumerate(l_importance):
	
	# exclude data which is 1:1 with the job feature (as these will not be useful)
	if l_features[i] in ['... list of features ...']:
		continue

	# mutual information divided by the features entropy
	# https://stackoverflow.com/questions/42303752/are-there-feature-selection-algorithms-that-can-be-applied-to-categorical-data-i#comment73734435_42303752
	resdict2[l_features[i]] = res/ data[l_features[i]].value_counts().size
print 'MI Divided by Feature Entropy:'
for elt in sorted(resdict2.items(), key=lambda x: x[1], reverse=True):
    print elt
