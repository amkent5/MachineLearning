"""
Another way to assess which features have more importance is to perform model based ranking on them. 
We can use an arbitrary machine learning method to build a predictive model for the response variable using each individual feature, 
and measure the performance of each model.

Tree based methods are probably among the easiest to apply, since they can model non-linear relations well and don’t require much tuning, 
so we will use a Random Forest classifier in the below.
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# load data
datafile = '/model_data/training_data.csv'
data = pd.read_csv(datafile, delimiter='\t', dtype=str)

# pick features from list(data.columns)
data = data[[
	'...pick features we want to explore...'
]]

print 'Data shape: ', data.shape
print 'Data features: ', list(data.columns)

y = data[['no_access_event']]
scores = []
for feature in list(data.columns):

	# onehot encode the feature
	feature_data = data[[feature]]
	encoded_feature_data = pd.get_dummies(feature_data)

	print '\n'
	print feature
	print feature_data.shape
	print encoded_feature_data.shape
	print y.shape

	# upsample minority class
	from imblearn.over_sampling import RandomOverSampler
	ros = RandomOverSampler(ratio=0.5)
	X_resampled, y_resampled = ros.fit_sample(encoded_feature_data, y)

	print '\n'
	print X_resampled.shape
	print y_resampled.shape

	# create train and test split
	X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=0, test_size=0.2)

	print '\n'
	print 'Training data'
	print X_train.shape
	print y_train.shape

	print 'Testing data'
	print X_test.shape
	print y_test.shape

	# create and train classifier
	clf = RandomForestClassifier(n_jobs=2, n_estimators=40)
	clf.fit(X_train, y_train)

	# use model and acquire predictive power of each feature
	predicted = clf.predict(X_test)
	accuracy = accuracy_score(y_test, predicted)
	print 'Accuracy score: ',  accuracy

	scores.append((accuracy, feature))

	# release memory for more python processes
	del encoded_feature_data

print sorted(scores, reverse=True)
