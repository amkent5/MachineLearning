# Text processing (NLP)

# https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html

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



# Clean documents

# create English stop words list
from stop_words import get_stop_words
en_stop = get_stop_words('en')

# import document tokenizer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

# import stemming algorithm (PorterStemmer algorithm)
from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()


l_vocab = []
for doc in ser_name:
	print doc

	raw = doc.lower()
	raw = raw.replace("'", "")	# handle apostrophe's

	# Apply the tokenizer
	# this converts the document to its atomic elements (i.e. no punctuation etc.)
	tokens = tokenizer.tokenize(raw)
	print tokens

	# Remove stop words from tokens
	# Certain parts of English speech, like conjunctions ("for", "or") or the word "the" are meaningless to a topic model. 
	# These terms are called stop words and need to be removed from our token list.
	stopped_tokens = [i for i in tokens if not i in en_stop]
	print stopped_tokens

	# Remove all non-ascii characters from tokens
	stopped_tokens_ascii = []
	for token in stopped_tokens:
		stopped_tokens_ascii.append( ''.join(x for x in token if x in string.printable) )

	# Apply 'stemming'
	# Stemming words is another common NLP technique to reduce topically similar words to their root. For example, "stemming," 
	# "stemmer," "stemmed," all have similar meanings; stemming reduces those terms to "stem." This is important for topic modeling, 
	# which would otherwise view those terms as separate entities and reduce their importance in the model.
	texts = [p_stemmer.stem(i) for i in stopped_tokens_ascii]
	print texts

	# remove empty strings ('') from list
	texts = [x for x in texts if x]
	print texts

	# convert all strings to non-unicode


	l_vocab.append(texts)

'''
An example of the cleaning process:

RETIRED BEAUTIFUL DAY SET- BAG INCLUDED
['retired', 'beautiful', 'day', 'set', 'bag', 'included']
['retired', 'beautiful', 'day', 'set', 'bag', 'included']
[u'retir', u'beauti', 'day', 'set', 'bag', u'includ']
'''