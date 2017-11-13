"""
Model notes:

- using inbuilt Scikit-learn encoding functions (far more efficient)
- upsample minority up to 50%

"""

### Imports
import time
import os
import operator
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier # wrap keras in scikit-learn for GridSearchCV

### Begin pre-processing
print 'Starting pre-processing...'
t0 = time.time()
datafile =    '/model_data/training_data.csv'
data = pd.read_csv(datafile, delimiter='\t', dtype=str)

# pick features using list(data.columns) from univariate selection
data = data[[
	'... my list of features ...'
]]

for i, feature in enumerate(list(data.columns)):
	if feature != 'response_variable':
		col_data = data[feature]

		# encode labels
		import sklearn.preprocessing
		le = sklearn.preprocessing.LabelEncoder()
		le.fit(col_data)
		t1 = le.transform(col_data)
		print t1

		# one-hot encode encoded labels
		lb = sklearn.preprocessing.LabelBinarizer()
		lb.fit(range(max(t1) + 1))
		t2 = lb.transform(t1)
		print t2
		print t2.shape
		print type(t2)

		# concatenate one-hot encoded columns
		if i == 0:
			encoded_data = t2
		else:
			encoded_data = np.concatenate((encoded_data, t2), axis=1)
			print encoded_data.shape

# append the response variable
le.fit(data['response_variable'])
t3 = le.transform(data['response_variable'])
t3 = np.expand_dims(t3, axis=1)
encoded_data = np.concatenate((encoded_data, t3), axis=1)
print encoded_data
print encoded_data.shape

# split
X = encoded_data[:, 0:-1]
y = encoded_data[:, -1]

# upsample minority class
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(ratio=0.5)
X_resampled, y_resampled = ros.fit_sample(X, y)

print '\n'
print X_resampled.shape
print y_resampled.shape

print 'Pre-processing took: ', time.time() - t0, 's'
### End pre-processing


### Begin modelling
print 'Starting modelling...'
seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits = 2, shuffle = True, random_state = seed)
cross_val_scores = []
model_input_dims = np.shape(X_resampled)[1]

for train, test in kfold.split(X_resampled, y_resampled):
    
    print 'Training indexes:'
    print train
    print '\n'
    print 'Testing indexes:'
    print test
    print '\n'

    # create network topology
    model = Sequential()
    model.add(Dense(model_input_dims + 1, input_dim=model_input_dims, init='uniform', activation='relu'))
    model.add(Dense(25, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit model
    model.fit(X_resampled[train], y_resampled[train], nb_epoch=35, batch_size=100)

    # evaluate the model
    scores = model.evaluate(X_resampled[test], y_resampled[test])
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cross_val_scores.append(scores[1] * 100)

    print '\n'
    print 'cross_val_scores:'
    print cross_val_scores

    print '\nConfusion Matrix:'
    print confusion_matrix(y_resampled[test], model.predict_classes(X_resampled[test]))

print("%.2f%% (+/- %.2f%%)" % (np.mean(cross_val_scores), np.std(cross_val_scores)))

"""
cross_val_scores:
[91.893662994580424, 91.843776935379296]

Confusion Matrix:
[[203257  22591]
 [  5040 107884]]
91.87% (+/- 0.02%)
"""
