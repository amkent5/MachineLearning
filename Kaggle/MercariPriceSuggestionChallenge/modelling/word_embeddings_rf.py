# Add word embeddings of the item_description field into the model

"""
In addition to these features:

* item_condition
* shipping
* brands that have over 50 distinct representations (800 length cat. vec.)
* category_name split by '/' [0]
* category_name split by '/' [1] ([2] will give us too many inputs)
* length(name)
* length(item_description)

use Word2Vec to embed the words in the name field in a vector-space
and use the vector-space as additional input.

"""




#-----------------------#
"""	Data Wrangling """
#-----------------------#

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data into pandas
datafile = '/Users/Ash/Projects/MachineLearning/Kaggle/MercariPriceSuggestionChallenge/data/train.tsv'
#datafile = '/Users/admin/Documents/Projects/MachineLearning/Kaggle/MercariPriceSuggestionChallenge/data/train.tsv'
df = pd.read_csv(datafile, delimiter='\t')

# some stats
print df.head(5)
print df.shape
print df.columns

# handle missing data
for feature in df.columns: df[feature].fillna(value="missing_val", inplace=True)
#print df.head(5)
#print df.shape

# store name series and item_description series for later
ser_name = df['name']
ser_item_description = df['item_description']

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






#-----------------------------------------------#
""" Use Gensim to model the 'name' field """
#-----------------------------------------------#

'''
- pick max num of words for name field (5)
- pick max vector length for word embedding (100)
- then create a new input of length 500 for each name field
- add it into our input matrix
- visualise context between words with PCA
'''

from gensim.models import Word2Vec
from string import punctuation

# Use the gensim library to encode the words in the name field.
# This wont only encode the words, but also the words meaning.
# Use the 'Word2Vec' method to do this. It uses a neural network to generate a 
# vector representation of the word and its associated meaning. Within the
# vector-space that is generated we can then do things like:
# king - man + royalty
# The 'closest' vector in the vector-space to this vector result is 'queen'.
# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

# Word2Vec expects a list of lists where the inner list is the sentence, split
# by word, made lowercase and with all punctuation removed
l_vocab = []
for i, val in enumerate(ser_name.values):
	if len(val) > 0:
		# lower
		val = val.lower()
		# remove punctuation and split
		sentence = val.translate(None, punctuation).split(' ')
		l_vocab.append(sentence)
	else:
		l_vocab.append([''])

# train neural network model
# Word2Vec params:
# - size: (default 100) The number of dimensions of the embedding, e.g. the length of the dense vector to represent each token (word).
# - window: (default 5) The maximum distance between a target word and words around the target word.
# - min_count: (default 5) The minimum count of words to consider when training the model; words with an occurrence less than this count will be ignored.
# - workers: (default 3) The number of threads to use while training.
model = Word2Vec(l_vocab, min_count = 1)

# summarize the loaded model
print model

# summarize vocabulary
words = list(model.wv.vocab)
#print words

# take a look at a vector representation for a word
print model['gold']
print type(model['gold'])


# form input array (choose maximum of 5 words in sentence and pad sentences with less)

'''
# 1st attempt
# This is too slow as each time we do a numpy concatenate numpy has to find 
# a new contiguous block of RAM that can fit the array on each iteration.
# This is extremely intensive on memory and slows the execution down massively.

prev_time = time.time()
for k, sentence in d_ix_to_sentence.items():

	#print k, sentence
	if k % 1000 == 0:
		print k
		print time.time() - prev_time
		prev_time = time.time()

	for i, word in enumerate(sentence):

		if i == 0:
			array = model[word]
		elif i < 5:
			array2 = model[word]
			array = np.concatenate((array, array2))

	# normalise sentence lengths
	if len(array) != 500:
		diff = 500 - len(array)
		array = np.concatenate((array, np.zeros(diff)))

	# reshape for vertical concatenation
	array = array.reshape(1, 500)

	#print array
	#print array.shape

	if k == 0:
		input_array = array
	else:
		input_array = np.concatenate((input_array, array), axis=0)

print len(input_array)
print input_array.shape

np.save('word_embedding_input_matrix', input_array)

# now append this array to the rest of the input array (check it has the same number of rows..1.4mill)
quit()
'''

# 2nd attempt
# (use lists more than arrays)
# (see comment: http://akuederle.com/create-numpy-array-with-for-loop)

#### This is now maxing out my 16Gb of memory

# I posted in SO and someone suggested writing to a file so that nothing has to get stored
# in memory, then reading the file into another script.

# See word_embeddings_write.py for this code
# It does indeed save on memory, but generates a massive file which I wouldn't be able to
# load into memory.

# So:
#	- keep in memory
#	- cut down the vector representation from 100 to 50
#	- maybe just take the last 800,000 rows of data.



prev_time = time.time()
l_encodings = []
for k, sentence in enumerate(l_vocab):	# l_vocab is a list of lists

	#print k, sentence
	if k % 100000 == 0:
		print k
		print time.time() - prev_time
		prev_time = time.time()

	# form list of 5-word-embedding to append to l_encodings
	for i, word in enumerate(sentence):

		if i == 0:
			l_array = model[word].tolist()
			#l_array = np.round(model[word].tolist(), decimals=7).tolist()
		elif i < 5:
			l_array2 = model[word].tolist()
			#l_array2 = np.round(model[word].tolist(), decimals=7).tolist()
			l_array += l_array2

	# normalise sentence lengths
	if len(l_array) != 500:
		diff = 500 - len(l_array)
		l_array += [0 for _ in range(diff)]

	print l_array
	print len(l_array)

	l_encodings.append(l_array)


# check len of the array is the same length as encoded_data
# if it is, then add this into the input array

print len(l_encodings)
print l_encodings[100]

#df_embeddings = pd.DataFrame(l_encodings)
#print df_embeddings.shape


'''
Can add in like:
>>> l = [[1,2,1], [5,4,3], [1,0,0]]
>>> for _ in l: print _
...
[1, 2, 1]
[5, 4, 3]
[1, 0, 0]
>>> df = pd.DataFrame(l)
>>> df
   0  1  2
0  1  2  1
1  5  4  3
2  1  0  0
>>> df.loc[:, 2]
0    1
1    3
2    0
Name: 2, dtype: int64
'''

# then see if it improves the accuracy of the classifier (run both, 
# and compare the y_pred for the 10 vals below as a first step, 
# then do crossvalidation on rmse).



""" Fin code """
'''
prev_time = time.time()

# init machine learning model

def sentance_to_array(sentance):

	l_array = []
	for word in sentance:
		l_array.extend(model[word].tolist())

	if len(l_array) < 500:
		l_array.extend([0] * (500 - len(l_array)))

	return l_array


for count, sentance in enumerate(l_vocab):

	if not count % 100000:
		print k, time.time() - prev_time
		prev_time = time.time()
	
	l_array = sentance_to_array(sentance)

	# machine learning model.ingest(l_array, some_score)
'''



quit()











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

# output:
'''
Begin modelling... 2018-01-08 09:50:24
building tree 1 of 10
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed: 13.5min remaining:    0.0s
building tree 2 of 10
building tree 3 of 10
building tree 4 of 10
building tree 5 of 10
building tree 6 of 10
building tree 7 of 10
building tree 8 of 10
building tree 9 of 10
building tree 10 of 10
[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed: 148.2min finished

[ 8.1]
8.0

[ 12.2]
8.0

[ 16.]
19.0

[ 9.9]
6.0

[ 79.4]
64.0

[ 47.1]
59.0

[ 34.3]
44.0

[ 32.1]
35.0

[ 9.5]
10.0

[ 43.]
52.0

[ 12.89333333]
10.0
'''


# Need to add in
# - stratified k-fold (do we want stratified?)
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

