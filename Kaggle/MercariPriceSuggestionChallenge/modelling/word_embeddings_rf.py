"""

*** Kaggle: Mercari price suggestion challenge ***



Combine appropriate categorical and continuous features with
gensim word2vec encoded features.

Features to model:
* item_condition	(discrete)
* shipping	(discrete)
* brands that have over 75 distinct representations	(discrete)
* category_name split by '/' [0]	(discrete)
* category_name split by '/' [1] ([2] will give us too many inputs)	(discrete)
* length(name)	(continuous)
* length(item_description)	(continuous)
* vector representation of the name field	(high-dimensional vector)


Notes:
Use the last 800,000 rows of data to build the model.
Use KFold cross validation to train and test the model within the context of this subset, 
k times, with different hyperparameters (note: could use GridSearchCV)
If happy with the results then I could use dask to train the model on the entirety
of the train.tsv file (1.4million rows) and then use the fully trained model to create 
the submission file against the provided test.tsv file.

"""






#-----------------------#
"""	Data Wrangling """
#-----------------------#

import pandas as pd
import numpy as np

# load data into pandas
#datafile = '/Users/Ash/Projects/MachineLearning/Kaggle/MercariPriceSuggestionChallenge/data/train.tsv'
datafile = '/Users/admin/Documents/Projects/MachineLearning/Kaggle/MercariPriceSuggestionChallenge/data/train.tsv'
df = pd.read_csv(datafile, delimiter='\t')

# due to memory constraints, limit to last 800k rows
df = df.iloc[len(df) - 800000:, :]

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
def split_cat(text):
	try:
		split = text.split('/')
		if len(split) >= 2:
			return split
		else:
			return ['missing_val', 'missing_val']
	except:
		return ['missing_val', 'missing_val']

df['cat1'] = df['category_name'].apply(lambda x: split_cat(x)[0])
df['cat2'] = df['category_name'].apply(lambda x: split_cat(x)[1])

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

print encoded_data.shape	# (800000, 590) looks good

# now append the continuous features
npa_len_name = df['len_name'].as_matrix()	# convert series to numpy-array representation
npa_len_name = np.expand_dims(npa_len_name, axis=1)	# make dimensionality the same ready for concatenation
encoded_data = np.concatenate( (encoded_data, npa_len_name), axis=1 )	# add to master input array

npa_len_itm_desc = df['len_itm_desc'].as_matrix()
npa_len_itm_desc = np.expand_dims(npa_len_itm_desc, axis=1)
encoded_data = np.concatenate( (encoded_data, npa_len_itm_desc), axis=1 )

print encoded_data.shape	# (800000, 592)
print encoded_data[:10, :]







#-----------------------------------------------#
""" Use NLTK to clean vocabulary """
#-----------------------------------------------#

import os
import string

# This process takes a few minutes so write our vocabulary to disk.
# Then load into memory to train word2vec.
#vocab_file = '/Users/Ash/Projects/MachineLearning/Kaggle/MercariPriceSuggestionChallenge/data/vocab_file.csv'
vocab_file = '/Users/admin/Documents/Projects/MachineLearning/Kaggle/MercariPriceSuggestionChallenge/data/vocab_file.csv'
if os.path.isfile(vocab_file):
	print 'Vocabulary already available for use.'

else:

	# create English stop words list
	from stop_words import get_stop_words
	en_stop = get_stop_words('en')

	# import document tokenizer
	from nltk.tokenize import RegexpTokenizer
	tokenizer = RegexpTokenizer(r'\w+')

	# import stemming algorithm (PorterStemmer algorithm)
	from nltk.stem.porter import PorterStemmer
	p_stemmer = PorterStemmer()

	def create_vocab(document):
		raw = document.lower()
		raw = raw.replace("'", "")	# handle apostrophe's

		# Apply the tokenizer
		# this converts the document to its atomic elements (i.e. no punctuation etc.)
		tokens = tokenizer.tokenize(raw)

		# Remove stop words from tokens
		# Certain parts of English speech, like conjunctions ("for", "or") or the word "the" are meaningless to a topic model. 
		# These terms are called stop words and need to be removed from our token list.
		stopped_tokens = [i for i in tokens if not i in en_stop]

		# Remove all non-ascii characters from tokens
		stopped_tokens_ascii = []
		for token in stopped_tokens: stopped_tokens_ascii.append( ''.join(x for x in token if x in string.printable) )

		# Apply 'stemming'
		# Stemming words is another common NLP technique to reduce topically similar words to their root. For example, "stemming," 
		# "stemmer," "stemmed," all have similar meanings; stemming reduces those terms to "stem." This is important for topic modeling, 
		# which would otherwise view those terms as separate entities and reduce their importance in the model.
		texts = [p_stemmer.stem(i) for i in stopped_tokens_ascii]

		# remove empty strings ('') from list
		texts = [x for x in texts if x]

		# convert unicode strings to python strings
		texts = map(str, texts)

		return texts

	print 'Wrting vocabulary to disk...'
	with open(vocab_file, 'w') as f:
		for document in ser_name:
			f.write( ','.join( create_vocab(document) ) + '\n' )
	print 'Vocabulary available for use.'







#-------------------------------------------------------------------------#
""" Use the word2vec NN model within Gensim to model the vocabulary """
#-------------------------------------------------------------------------#

'''
- pick max num of words for name field (5)
- pick max vector length for word embedding (100)
- then create a new input of length 500 for each name field
- add it into our input matrix
- optionally visualise context between words with PCA
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

# Word2Vec expects a list of lists where the inner list is the document
# also normalise the input to length 5
l_vocab = []
for doc in open(vocab_file, 'r'): l_vocab.append(doc.rstrip().split(',')[:5])

# sample
print len(l_vocab)
for i in range(1000, 1010): print l_vocab[i]

# Train neural network model
# Word2Vec params:
# - size: (default 100) The number of dimensions of the embedding, e.g. the length of the dense vector to represent each token (word).
# - window: (default 5) The maximum distance between a target word and words around the target word.
# - min_count: (default 5) The minimum count of words to consider when training the model; words with an occurrence less than this count will be ignored.
# - workers: (default 3) The number of threads to use while training.
model = Word2Vec(l_vocab, min_count=1, size=75)

# summarize the loaded model
print model

# summarize vocabulary
#words = list(model.wv.vocab)
#print words

# take a look at a vector representation for a word
print model['gold']
print type(model['gold'])








#--------------------------------------#
""" Prepare/finalise model input """
#--------------------------------------#

import time
import gc

# memory management
gc.collect()

def doc_to_array(document):
	l_array = []
	for word in document:
		l_array.extend(model[word].tolist())

	if len(l_array) < 375:
		l_array.extend([0] * (375 - len(l_array)))

	return l_array

# extend encoded_data by the length of the document embeddings
# and then overwrite with the dense vector
z = np.zeros(800000 * 375).reshape(800000, 375)
encoded_data = np.append(encoded_data, z, 1)
prev_time = time.time()
print 'Creating final input matrix...'
for i, row in enumerate(encoded_data):

	if not i % 100000:
		print i, time.time() - prev_time
		prev_time = time.time()

	row[len(row) - 375:] = doc_to_array(l_vocab[i])

# finally, append the target variable
npa_price = df['price'].as_matrix()
npa_price = np.expand_dims(npa_price, axis=1)
encoded_data = np.concatenate( (encoded_data, npa_price), axis=1 )








#--------------------------------------#
""" Random Forest Models """
#--------------------------------------#

import cPickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

seed = 7
np.random.seed(seed)
kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
cross_val_scores = []

X = encoded_data[:, :-1]
y = encoded_data[:, -1]

print 'Begin building models...'
i = 0
for train_ixs, test_ixs in kfold.split(X, y):

	i += 1
	print 'Building model on fold %i...' % i
	print '\n'*2
	print 'Training indexes:'
	print train_ixs
	print '\n'
	print 'Testing indexes:'
	print test_ixs

	# compile model
	if i == 1:
		# (Note: even though this looks like it exceeds memory, it does complete)
		rf_regr = RandomForestRegressor(verbose=2, n_estimators=10)
	elif i == 2:
		rf_regr = RandomForestRegressor(verbose=2, n_estimators=20)
	else:
		rf_regr = RandomForestRegressor(verbose=2, n_estimators=40)
	
	# fit model
	rf_regr.fit(X[train_ixs], y[train_ixs])

	# spot check model
	j = 0
	for ix in test_ixs:
		j+= 1
		if j > 100:
			break
		print ix
		print rf_regr.predict( X[ix].reshape(1, -1) )[0], '\t', y[ix]

	# evaluate model
	preds = rf_regr.predict( X[test_ixs] )
	mse = mean_squared_error(y[test_ixs], preds)
	rmse = np.sqrt(mse)
	print 'mse: ', mse
	print 'rmse: ', rmse
	cross_val_scores.append(mse)
	cross_val_scores.append(rmse)

	# save model to disk
	with open('model_%i' % i, 'wb') as f: cPickle.dump(rf_regr, f)
	f.close()


print cross_val_scores


"""
Output sample:

794
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.0s finished
[43.4] 	40.0
797
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.0s finished
[13.2] 	17.0
798
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.0s finished
[36.2] 	25.0
799
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.0s finished
[26.3] 	26.0
802
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.0s finished
[23.2] 	31.0
808
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.0s finished
[6.4] 	6.0
809
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.0s finished
[21.2] 	19.0
810
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.0s finished
[17.4] 	13.0
"""




