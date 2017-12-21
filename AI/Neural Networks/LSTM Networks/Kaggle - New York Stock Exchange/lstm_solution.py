# lstm-solution adapted from
# https://www.kaggle.com/benjibb/lstm-stock-prediction-20170507

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
import keras
import h5py
import requests
import os

# read data into pandas!
df = pd.read_csv('prices-split-adjusted.csv')

print df.size
print df.shape
print df.head()
print df.describe()

# move 'close' to the RHS as this is the target (and rename it adj_close)
df['adj_close'] = df.close
df.drop(['close'], 1, inplace=True)
print df.head()
print '\n'

# pick a stock
"""
data from: 	2010-01-04
	   to:	2016-12-30
row per day, but some days missing
"""
df = df[df.symbol == 'GOOG']
print df.head()
print df.shape
print '\n'

# plot the time series
df_plot = df[['date', 'adj_close']]

# format date properly and reset index
df_plot.date = pd.to_datetime(df_plot['date'], format='%Y-%m-%d')
df_plot = df_plot.set_index('date')

# grab control in matplotlib
ax = df_plot.plot()
ax.set_title('Google Stock Close')
ax.set_ylabel('$')
plt.show()



# normalise the data
def normalize_data(df):
	min_max_scaler = preprocessing.MinMaxScaler()
	df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
	df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
	df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
	df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))
	df['adj_close'] = min_max_scaler.fit_transform(df['adj_close'].values.reshape(-1,1))
	return df
df = normalize_data(df)

# get rid of symbol
df.drop('symbol', axis=1, inplace=True)

print df.head()



# create training and testing set
amount_of_features = len(df.columns)
data = df.as_matrix()	# convert from pandas to numpy representation

# create inputs which are timeseries of 22 days each
# We will create a dataset where X is a time series of close values
# And Y will be the next close value as time step 23.
window = 22
sequence_length = window + 1 	# index starting from 0

result = []
# create staggered timeseries inputs
for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
	result.append(data[index: index + sequence_length]) # index : index + 22days

result = np.array(result)
#print result
#quit()

row = round(0.9 * result.shape[0]) # 90% split
train = result[:int(row), :] # 90% date, all features 

x_train = train[:, :-1] 
y_train = train[:, -1][:,-1]

print '\n'
print train[0]
print '\n'
print x_train[0]
print '\n'
print y_train[0]

"""
Data Explained:

train[0]:

[['2010-01-04' 0.15704696284697284 0.16116745635467583 0.15638997547274563
  0.13172249913453232 0.15939907870000913]
 ['2010-01-05' 0.1572384301007388 0.15888449177913994 0.15499505903156935
  0.20246901961838756 0.15709185339304588]
 ['2010-01-06' 0.15613950705516955 0.14604928817903923 0.15334120789128863
  0.26818406396730377 0.14394234288381746]
 ['2010-01-07' 0.1424361736711216 0.1344569928575487 0.1400936188051209
  0.4325220904321288 0.1321052846061131]
 ['2010-01-08' 0.12795029172350836 0.13146378676523296 0.13445546079616427
  0.31849210323768934 0.13872602961480618]
 ['2010-01-11' 0.1383235509882924 0.1356323015637962 0.13546617289444324
  0.48640627573295864 0.1379653023031744]
 ['2010-01-12' 0.13265404483177834 0.13052526045920698 0.13020387894581792
  0.32719719554863924 0.12907917376216138]
 ['2010-01-13' 0.11503787040196306 0.11860320108963712 0.12203482192023124
  0.4380745276899239 0.12624526774310008]
 ['2010-01-14' 0.12120685395799757 0.12613695510283768 0.1268961265483322
  0.28582279927535986 0.12855249305006322]
 ['2010-01-15' 0.12906589265298174 0.12210374429097082 0.12636157658331465
  0.3664104434219415 0.12031836489932773]
 ['2010-01-19' 0.11895907578063747 0.12062405445575008 0.12373877072662653
  0.2909920780298931 0.12668831673605152]
 ['2010-01-20' 0.122938527633843 0.11977850979758192 0.12003013174931748
  0.2190658326807315 0.12066113758448932]
 ['2010-01-21' 0.12082391945380805 0.11720810182388625 0.12073176052871792
  0.42532946589721266 0.12280949823490567]
 ['2010-01-22' 0.10505596075952783 0.08559350077953165 0.10718348019495588
  0.45857350761443494 0.09524821622170992]
 ['2010-01-25' 0.09014547415327845 0.08614310780979728 0.08987641142737518
  0.2979527908391507 0.08688036960937046]
 ['2010-01-26' 0.08296911901043297 0.0868026350623452 0.08964255187321651
  0.29361032780218266 0.08890335874978927]
 ['2010-01-27' 0.08571647698844548 0.08597402982560692 0.0880137249258986
  0.2672194456301445 0.08863586999644141]
 ['2010-01-28' 0.08839717420940857 0.08199153158735578 0.08747078205286135
  0.21760714153673447 0.08210705433889715]
 ['2010-01-29' 0.08340203722006617 0.07777230421517684 0.08245073972967898
  0.2791007202707653 0.0784706812207096]
 ['2010-02-01' 0.08016356534314178 0.08173786303199809 0.07812398590165426
  0.15159968675111837 0.08104539528269883]
 ['2010-02-02' 0.08046326194124154 0.07946339353321047 0.07741401516789387
  0.27611275615322306 0.07945710734118588]
 ['2010-02-03' 0.07522667399436062 0.07998761764097534 0.08337793789759645
  0.20208586111512566 0.0875658129908668]
 ['2010-02-04' 0.08216157901196625 0.07773003514093463 0.07995325614056054
  0.22763312237208722 0.07582908201368749]]

x_train[0]:

[['2010-01-04' 0.15704696284697284 0.16116745635467583 0.15638997547274563
  0.13172249913453232 0.15939907870000913]
 ['2010-01-05' 0.1572384301007388 0.15888449177913994 0.15499505903156935
  0.20246901961838756 0.15709185339304588]
 ['2010-01-06' 0.15613950705516955 0.14604928817903923 0.15334120789128863
  0.26818406396730377 0.14394234288381746]
 ['2010-01-07' 0.1424361736711216 0.1344569928575487 0.1400936188051209
  0.4325220904321288 0.1321052846061131]
 ['2010-01-08' 0.12795029172350836 0.13146378676523296 0.13445546079616427
  0.31849210323768934 0.13872602961480618]
 ['2010-01-11' 0.1383235509882924 0.1356323015637962 0.13546617289444324
  0.48640627573295864 0.1379653023031744]
 ['2010-01-12' 0.13265404483177834 0.13052526045920698 0.13020387894581792
  0.32719719554863924 0.12907917376216138]
 ['2010-01-13' 0.11503787040196306 0.11860320108963712 0.12203482192023124
  0.4380745276899239 0.12624526774310008]
 ['2010-01-14' 0.12120685395799757 0.12613695510283768 0.1268961265483322
  0.28582279927535986 0.12855249305006322]
 ['2010-01-15' 0.12906589265298174 0.12210374429097082 0.12636157658331465
  0.3664104434219415 0.12031836489932773]
 ['2010-01-19' 0.11895907578063747 0.12062405445575008 0.12373877072662653
  0.2909920780298931 0.12668831673605152]
 ['2010-01-20' 0.122938527633843 0.11977850979758192 0.12003013174931748
  0.2190658326807315 0.12066113758448932]
 ['2010-01-21' 0.12082391945380805 0.11720810182388625 0.12073176052871792
  0.42532946589721266 0.12280949823490567]
 ['2010-01-22' 0.10505596075952783 0.08559350077953165 0.10718348019495588
  0.45857350761443494 0.09524821622170992]
 ['2010-01-25' 0.09014547415327845 0.08614310780979728 0.08987641142737518
  0.2979527908391507 0.08688036960937046]
 ['2010-01-26' 0.08296911901043297 0.0868026350623452 0.08964255187321651
  0.29361032780218266 0.08890335874978927]
 ['2010-01-27' 0.08571647698844548 0.08597402982560692 0.0880137249258986
  0.2672194456301445 0.08863586999644141]
 ['2010-01-28' 0.08839717420940857 0.08199153158735578 0.08747078205286135
  0.21760714153673447 0.08210705433889715]
 ['2010-01-29' 0.08340203722006617 0.07777230421517684 0.08245073972967898
  0.2791007202707653 0.0784706812207096]
 ['2010-02-01' 0.08016356534314178 0.08173786303199809 0.07812398590165426
  0.15159968675111837 0.08104539528269883]
 ['2010-02-02' 0.08046326194124154 0.07946339353321047 0.07741401516789387
  0.27611275615322306 0.07945710734118588]
 ['2010-02-03' 0.07522667399436062 0.07998761764097534 0.08337793789759645
  0.20208586111512566 0.0875658129908668]]

y_train[0]:

0.0758290820137


- train[0] has our 22 time step mini-time series of stock prices.
- x_train[0] has the last day left off (so it only contains the 21 time steps)
- y_train[0] is then the 'close' price for the 22nd day

So what we are doing is splitting the training data into lots of 22 time step series, and 
then teaching the model to predict the 22nd time step value.

At the end of the training we will predict the next time step on from the data, i.e. we'll
predict based on the last 21 days in the data set.
Then on the next prediction we will have 20 days of dataset data, and one day of prediction 
data to be fed into the model, and so on.
Eventually our model will only be predicting based on values it has previously predicted.

"""

# we can test the accuracy of predicting the 22nd time step for our mini time series using the below
x_test = result[int(row):, :-1]
y_test = result[int(row):, -1][:,-1]
# but ultimately we want to use the trained model to predict close for new data


# the LSTM architecture requires 3-d inputs
# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
"""
- Samples. One sequence is one sample. A batch is comprised of one or more samples.
- Time Steps. One time step is one point of observation in the sample.
- Features. One feature is one observation at a time step.
"""
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))



# build structure of model
def build_model(layers):
	d = 0.3
	model = Sequential()

	model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=True))
	model.add(Dropout(d))
	    
	model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=False))
	model.add(Dropout(d))
	    
	model.add(Dense(32,kernel_initializer="uniform",activation='relu'))        
	model.add(Dense(1,kernel_initializer="uniform",activation='linear'))

	# adam = keras.optimizers.Adam(decay=0.2)
	    
	start = time.time()
	model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
	print "Compilation Time : ", time.time() - start
	return model


model = build_model( [5, window, 1] )


# fit model
model.fit(x_train, y_train, batch_size=512, epochs=90, validation_split=0.1, verbose=1)


# get some predictions
p = model.predict(X_test)

print p
print y_test








