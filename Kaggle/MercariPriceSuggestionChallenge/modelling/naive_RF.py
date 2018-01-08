# naive RandomForest regression model

"""
Features to use:

* item_condition
* shipping
* brands that have over 50 distinct representations (800 length cat. vec.)
* category_name split by '/' [0]
* category_name split by '/' [1] ([2] will give us too many inputs)
* length(name)
* length(item_description)
"""




#-----------------------#
"""	Data Wrangling """
#-----------------------#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data into pandas
datafile = '/Users/admin/Documents/Projects/MachineLearning/Kaggle/MercariPriceSuggestionChallenge/data/train.tsv'
df = pd.read_csv(datafile, delimiter='\t')

# some stats
#print df.head(5)
#print df.shape
#print df.columns

# handle missing data
for feature in df.columns: df[feature].fillna(value="missing_val", inplace=True)
#print df.head(5)
#print df.shape

# create split category features
l_cat1, l_cat2 = [], []
for val in df.category_name:
	val = val.split('/')
	if len(val) == 3:
		l_cat1.append(val[0])
		l_cat2.append(val[1])
	else:
		l_cat1.append('missing_val')
		l_cat2.append('missing_val')

df.loc[:, 'cat1'] = pd.Series( l_cat1, index=df.index )
df.loc[:, 'cat2'] = pd.Series( l_cat2, index=df.index )

# create length of name and item description features
df['len_name'] = df['name'].apply(len)
df['len_itm_desc'] = df['item_description'].apply(len)

# clean the DataFrame a little
df = df[[
	'train_id',
	'item_condition_id',
	'shipping',
	'brand_name',
	'cat1',
	'cat2',
	'len_name',
	'len_itm_desc',
	'price'
]]
#print df.head(10)
#print df.shape

# limit the brand feature to only brands which have more than 75
# products in the dataset
ser_brands = df['brand_name'].value_counts()

# limits to 648 distinct brands (from 4,808)
ser_brands = ser_brands[ser_brands.values>75]
s_brands = set()
for brand in ser_brands.index: s_brands.add(brand)

# now check the brands in df, if the brand is in s_brands
# then keep it intact, otherwise, replace it with 'brand_replaced'
ser_brands = df['brand_name']
ser_bool_check = ser_brands.isin(s_brands)
df.loc[:, 'brand_check_repr'] = ser_bool_check

# use boolean column to create brand_repr column
df['brand_repr'] = np.where(df['brand_check_repr'] == False, 'brand_replaced', df['brand_name'])

# tidy df up
df = df[[
	'train_id',
	'item_condition_id',
	'shipping',
	'brand_repr',
	'cat1',
	'cat2',
	'len_name',
	'len_itm_desc',
	'price'
]]
print df.head(20)




#-----------------------#
"""	Data Encoding """
#-----------------------#

import sklearn.preprocessing

# iterate through our features and encode the discrete ones
i = -1
for feature in list(df.columns):
	if feature in ['train_id', 'len_name', 'len_itm_desc', 'price']:
		continue
	else:
		i += 1
		print feature
		ser_feature = df[feature]

		# encode the feature space
		le = sklearn.preprocessing.LabelEncoder()
		le.fit(ser_feature)
		trans1 = le.transform(ser_feature)
		print trans1

		# one-hot/binarize the encoded feature space
		lb = sklearn.preprocessing.LabelBinarizer()
		lb.fit(range(max(trans1) + 1))
		trans2 = lb.transform(trans1)
		print trans2
		print trans2.shape
		print type(trans2)
		print '\n'

		# concatenate numpy arrays to form our input data
		if i == 0:
			encoded_data = trans2
		else:
			encoded_data = np.concatenate( (encoded_data, trans2), axis=1 )

print encoded_data.shape	# (1482535, 782) looks good

# now append the continuous features and the target
npa_len_name = df['len_name'].as_matrix()	# convert series to numpy-array representation
npa_len_name = np.expand_dims(npa_len_name, axis=1)	# make dimensionality the same ready for concatenation
encoded_data = np.concatenate( (encoded_data, npa_len_name), axis=1 )	# add to master input array

npa_len_itm_desc = df['len_itm_desc'].as_matrix()
npa_len_itm_desc = np.expand_dims(npa_len_itm_desc, axis=1)
encoded_data = np.concatenate( (encoded_data, npa_len_itm_desc), axis=1 )

npa_price = df['price'].as_matrix()
npa_price = np.expand_dims(npa_price, axis=1)
encoded_data = np.concatenate( (encoded_data, npa_price), axis=1 )

print encoded_data.shape	# (1482535, 785)
print encoded_data[:10, :]




#-----------------------#
"""	Modelling """
#-----------------------#

#import x
#import y
#import z
import time
import datetime
print 'Begin modelling...', datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%dT%H:%M:%SZ')

# very naive regression approach just to get going
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

X = encoded_data[:, :-1]
y = encoded_data[:, -1]

regr = RandomForestRegressor(verbose=2)	# this takes some time...
regr.fit(X, y)

for i in range(10, -1, -1):
	print regr.predict( X[i] )
	print y[i]
	print '\n'


# need to add in
# - stratified k-fold
# 	http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
#
# - some kind of scoring
#	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
#	http://www.blopig.com/blog/2017/07/using-random-forests-in-python-with-scikit-learn/

# essentially do:
'''
# modelling
seed = 7
np.random.seed(seed)
print 'Starting modelling...'
kfold = StratifiedKFold(n_splits = 3, shuffle = True, random_state = seed)
cross_val_scores = []
model_input_dims = np.shape(X_reduced)[1]

for train, test in kfold.split(X_reduced, Y):
    
    print 'Training indexes:'
    print train
    print '\n'
    print 'Testing indexes:'
    print test
    print '\n'

'''
# and inside each fold iteration:
#	- fit a new random forest model. (this takes some time just like with the nn fitting)
#	- print off some accuracy scores using y_actual and y_pred (rmse)





















"""
Future Work

Future work will be to dig deeper into 'name' and 'item_description'

This kernel:
https://www.kaggle.com/knowledgegrappler/a-simple-nn-solution-with-keras-0-48611-pl
deals with the words.
Firstly it encodes them (see the bit where it encodes the item_description and name
into seq_item_description and seq_name).

He then visualises the lengths of the sequences in histograms, and picks a max
length for each of the sequences.

He then uses (from keras.preprocessing.sequence import) pad_sequences to make each
of the sequences the same length (https://stackoverflow.com/questions/42943291/what-does-keras-io-preprocessing-sequence-pad-sequences-do).
They can then be inputted into the network.

He then uses Keras' Embedding layer when he builds the neural network which 
creates a vector in space for each word. Similar words are similiarly positioned
within the vector space:
https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/


****
We could also use the Word2Vec Embedding such as done here:
https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

And just pass in the inputs as features.
****
"""


'''
# word embed the item_description
ser_itm_desc = df.item_description
print ser_itm_desc

# word2vec expects a list of lists where the inner list is the sentence, split
# by word, lowercase and with all punctuation removed
l_sentences = []
for val in ser_itm_desc.values:
	if len(val) > 0:
		# lower
		val = val.lower()
		# remove punctuation and split
		sentence = val.translate(None, punctuation).split(' ')
		l_sentences.append(sentence)

print len(l_sentences)
print l_sentences[220]

# let's use the gensim library to create our embeddings in vector-space
from gensim.models import Word2Vec

# train model
model = Word2Vec(l_sentences, min_count=1)

# summarize the loaded model
print model

# summarize vocabulary
words = list(model.wv.vocab)
print words
'''