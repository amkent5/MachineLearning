# Text processing (NLP)

# https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html

import os
import string
import time
import pandas as pd
import numpy as np


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




#-----------------------------------------------#
""" Use NLTK to clean vocabulary """
#-----------------------------------------------#

# This process takes a few minutes so write our vocabulary to disk.
# Then load into memory to train word2vec.
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

# Word2Vec expects a list of lists, so let's load our vocabulary into memory
# and create the object.
