""" Output class distribution plots (before resampling) """

import datetime
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

datafile = '/model_data/training_data.csv'
pd_data = pd.read_csv(datafile, delimiter='\t')


### class distribution
sns.countplot(x='no_access_event', data=pd_data, palette='hls').set_title('Class Distribution')
plt.show()


### how many no accesses does the client have per month?
pd_data_0class = pd_data[pd_data.no_access_event == 0]
pd_data_1class = pd_data[pd_data.no_access_event == 1]
num_0class = pd_data_0class.shape[0]
num_1class = pd_data_1class.shape[0]

monthly_data = pd_data_1class[['feature_1', 'output_class']]
monthly_data['feature_1'] = pd.to_datetime(monthly_data['feature_1'])
print monthly_data
print monthly_data.dtypes
print monthly_data.index

# set dataframe index to the timestamp (not the row int in the dataframe row)
monthly_data = monthly_data.set_index(['feature_1'])

# group by the month
g = monthly_data.groupby(pd.TimeGrouper("M"))

# then sum up the number of no accesses
monthly_sum = g.sum()
print monthly_sum

# plot
monthly_sum.plot()
plt.title('Number of No Accesses per Month')
plt.show()


### Now assess each feature in turn to try and expose differences in the class distributions
col_headers = list(pd_data_0class.columns)
l_dont_chart = ['...my list of features...']
for feature in l_dont_chart:
	col_headers.remove(feature)

for feature in col_headers:
	
	# create categorical label ordering
	if feature == 'feature_x':
		import calendar
		sorted_cats = list(calendar.day_name)
		print sorted_cats
	else:
		unique = pd_data_0class[feature].unique()
		sorted_cats = np.sort(unique)
		print sorted_cats

	### Counts chart
	fig = plt.figure()
	ax1 = fig.add_subplot(2, 1, 1)
	ax2 = fig.add_subplot(2, 1, 2)

	sns.countplot(x=feature, data=pd_data_0class, ax=ax1, order=sorted_cats)
	sns.countplot(x=feature, data=pd_data_1class, ax=ax2, order=sorted_cats)

	fig.suptitle('Response Variable Class Distribution - %s' % feature, fontsize=16)
	ax1.title.set_text('Access Jobs (0-class)')
	ax1.set_xlabel('')
	ax2.title.set_text('No Access Jobs (1-class)')
	ax2.set_xlabel('')

	plt.show()


	### Normalised % axis chart
	fig = plt.figure()
	ax1 = fig.add_subplot(2, 1, 1)
	ax2 = fig.add_subplot(2, 1, 2)

	fig.suptitle('Response Variable Class Distribution - %s' % feature, fontsize=16)
	ax1.title.set_text('Access Jobs (0-class)')
	ax2.title.set_text('No Access Jobs (1-class)')

	sns.barplot(
		x=feature,
		y=feature,
		data=pd_data_0class,
		estimator=lambda x: (float(len(x)) / len(pd_data_0class)) * 100,
		ax=ax1,
		order=sorted_cats
		)
	ax1.set(ylabel="% Representation in class")
	ax1.set(ylim=(0, 100))
	ax1.set_xlabel('')

	sns.barplot(
		x=feature,
		y=feature,
		data=pd_data_1class,
		estimator=lambda x: (float(len(x)) / len(pd_data_1class)) * 100,
		ax=ax2,
		order=sorted_cats
		)
	ax2.set(ylabel="% Representation in class")
	ax2.set(ylim=(0, 100))
	ax2.set_xlabel('')

	plt.show()
