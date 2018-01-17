# Word Embedding part (write to file)

# Add word embeddings of the name field into the model

# Python does not use any memory using this writing approach.
# However, processing 500k rows generated a 3.1Gb file
# So I guess the resulting file would be around 10Gb.
# Who knows if this file could be read back into memory for the model training..

# I think I need to:

#	- keep in memory
#	- cut down the vector representation from 100 to 50
#	- maybe just take the last 800,000 rows of data.




#-----------------------#
"""	Data Wrangling """
#-----------------------#

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data into pandas
#datafile = '/Users/Ash/Projects/MachineLearning/Kaggle/MercariPriceSuggestionChallenge/data/train.tsv'
datafile = '/Users/admin/Documents/Projects/MachineLearning/Kaggle/MercariPriceSuggestionChallenge/data/train.tsv'
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

# 2nd attempt
# (use lists more than arrays)
# (see comment: http://akuederle.com/create-numpy-array-with-for-loop)

#### This is now maxing out my 16Gb of memory
### I need to use generators instead of lists:
### https://realpython.com/blog/python/introduction-to-python-generators/

# I posted in SO and someone suggested writing to a file so that nothing has to get stored
# in memory, then reading the file into another script. I could try this:
"""
>>> import numpy as np
>>> with open('ash.csv', 'a') as f:
...     for i in range(10, 100):
...             myarray = np.arange(i)
...             strarray = ','.join(map(str, myarray))
...             f.write(strarray + '\n')
...     f.close()
"""

with open('word_embeddings.csv', 'a') as f:

	prev_time = time.time()
	for k, sentence in enumerate(l_vocab):	# l_vocab is a list of lists

		if k % 10000 == 0:
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

		# convert to strings and write
		f.write(','.join([str(elt) for elt in l_array]) + '\n')

	f.close()
quit()




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

quit()

