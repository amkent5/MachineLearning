"""
Example of benchmarking a suite of the most common machine learning algorithms on a data set.
Includes a Random Forest with additional PCA features.
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


datafile = '/model_data/training_data_newfeatures.csv'
data = pd.read_csv(datafile, delimiter='\t')
print data.shape
print list(data.columns)

# create dummy variables (or 'one-hot encoded' them)
data2 = pd.get_dummies(data)
X = data2.iloc[:,1:]
y = data2.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



########################################################################
#	Logistic Regression
########################################################################

print 'Beginning logistic regression...'
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print confusion_matrix
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))



########################################################################
#	Random Forest
########################################################################

print 'Beginning random forest...'
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print 'Mean accuracy score: ',  accuracy

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, predicted)
print confusion_matrix



########################################################################
#	XGBoost
########################################################################

print 'Beginning XGBoost...'
# https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

numpy_X_train = X_train.as_matrix()
numpy_y_train = y_train.as_matrix()
numpy_X_test = X_test.as_matrix()
numpy_y_test = y_test.as_matrix()

# fit model to training data
#model = XGBClassifier(max_depth=5, n_estimators=250)
model = XGBClassifier()
model.fit(numpy_X_train, numpy_y_train)
print model

# make predictions for test data
y_pred = model.predict(numpy_X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(numpy_y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

confusion_matrix = confusion_matrix(numpy_y_test, y_pred)
print confusion_matrix



########################################################################
#	SVM
########################################################################

print 'Beginning support vector machine...'
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print accuracy_score(y_test, predicted)



########################################################################
#	Random Forest With PCA additional features
########################################################################

### PCA 100-components + Random Forest
# Analyse how many principle components we require to describe the variance in the data
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.ylabel('cumulative explained variance');
plt.xlabel('number of components')
plt.show()

pca = PCA(n_components=100)
pca.fit(X)

X_pca = pca.transform(X)
print X_pca
print X_pca.shape
print y.shape

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, random_state=0)

print 'Beginning random forest with purely PCA components...'
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print 'Mean accuracy score: ',  accuracy

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, predicted)
print confusion_matrix

### (Data + PCA 3-components) + Random Forest
pca = PCA(n_components=3)
pca.fit(X)
X_pca = pca.transform(X)
print X.shape
print X_pca.shape

X_new = np.concatenate((X, X_pca),axis=1)
print X_new.shape

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state=0)

print 'Beginning random forest with data combined with PCA components...'
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print 'Mean accuracy score: ',  accuracy

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, predicted)
print confusion_matrix
