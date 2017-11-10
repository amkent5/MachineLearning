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

monthly_data = pd_data_1class[['job_cdate', 'no_access_event']]
monthly_data['job_cdate'] = pd.to_datetime(monthly_data['job_cdate'])
print monthly_data
print monthly_data.dtypes
print monthly_data.index

# set dataframe index to the timestamp (not the row int in the dataframe row)
monthly_data = monthly_data.set_index(['job_cdate'])

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
l_dont_chart = ['... list of features ...']
for feature in l_dont_chart:
	col_headers.remove(feature)

for feature in col_headers:

	print '\n'
	print feature

	# include the job_version as an arbitrary '1' to count
	df0_feature = pd_data_0class[[feature, 'job_version']]
	df1_feature = pd_data_1class[[feature, 'job_version']]
	total0 = df0_feature.shape[0]
	total1 = df1_feature.shape[0]

	# add counts as a new column to the dataframes
	df0_feature['counts'] = df0_feature.groupby([feature])['job_version'].transform('count')
	df1_feature['counts'] = df1_feature.groupby([feature])['job_version'].transform('count')

	# drop duplication
	df0_feature = df0_feature.drop_duplicates()
	df1_feature = df1_feature.drop_duplicates()

	# add percentage column to dataframes
	df0_feature['perc'] = df0_feature['counts'].divide(total0)*100
	df0_feature['perc'] = df0_feature['perc'].round(2)
	
	df1_feature['perc'] = df1_feature['counts'].divide(total1)*100
	df1_feature['perc'] = df1_feature['perc'].round(2)

	# do sorting
	#sorted_cats = np.sort(df0_feature[feature])

	# do sorting
	if feature == 'dayofweek_job_created':
		import calendar
		sorted_cats = list(calendar.day_name)
		print sorted_cats
	else:
		sorted_cats = np.sort(df0_feature[feature])

	# do charting
	# counts
	fig = plt.figure()
	ax1 = fig.add_subplot(2, 1, 1)
	ax2 = fig.add_subplot(2, 1, 2)

	sns.barplot(x=df0_feature[feature], y=df0_feature['counts'], order=sorted_cats, ax=ax1)
	sns.barplot(x=df1_feature[feature], y=df1_feature['counts'], order=sorted_cats, ax=ax2)
	
	fig.suptitle('Response Variable Class Distribution - %s' % feature, fontsize=16)
	ax1.title.set_text('Access Jobs (0-class)')
	ax1.set(ylabel="Count")
	ax1.set_xlabel('')
	ax2.title.set_text('No Access Jobs (1-class)')
	ax2.set(ylabel="Count")
	ax2.set_xlabel('')
	plt.show()
	
	# percentage
	fig = plt.figure()
	ax1 = fig.add_subplot(2, 1, 1)
	ax2 = fig.add_subplot(2, 1, 2)

	sns.barplot(x=df0_feature[feature], y=df0_feature['perc'], order=sorted_cats, ax=ax1)
	sns.barplot(x=df1_feature[feature], y=df1_feature['perc'], order=sorted_cats, ax=ax2)

	fig.suptitle('Response Variable Class Distribution - %s' % feature, fontsize=16)
	ax1.title.set_text('Access Jobs (0-class)')
	ax2.title.set_text('No Access Jobs (1-class)')

	ax1.set(ylabel="% Representation in class")
	ax1.set(ylim=(0, 100))
	ax1.set_xlabel('')

	ax2.set(ylabel="% Representation in class")
	ax2.set(ylim=(0, 100))
	ax2.set_xlabel('')

	plt.show()
