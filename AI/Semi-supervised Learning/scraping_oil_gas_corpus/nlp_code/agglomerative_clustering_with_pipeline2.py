### agglomerative_clustering.py with scikit-learn pipeline

"""
Possible pipeline steps:
	- the tokenize_and_stem function
	- the TfidfVectorizer step
	- the creation of the distance matrix

"""

##### Agglomerative clustering of Oil and Gas jargon

### Load scraped data
import pickle

filename = '/Users/admin/Documents/Projects/MachineLearning/AI/Semi-supervised Learning/scraping_oil_gas_corpus/scraping_code/d_oil_and_gas_terms.pickle'
with open(filename, 'rb') as handle:
	d_data = pickle.load(handle)

# sample
for k, v in d_data.items(): print k, '\n', v, '\n'*2
"""

forward multiple-contact test
A laboratory test to determine the phase envelope between lean gas and oil by equilibrating a gas sample 
several times with fresh samples of oil. In a forward-contact test, light and intermediate components are 
stripped from the oil by multiple contacts with the gas. The test also indicates how many contacts are required 
before the gas with added components becomes miscible with the oil. The molar ratios at each contact step are 
typically designed using PVT simulation software that incorporates the fluid composition at each contact.


standing valve
A downhole valve assembly that is designed to hold pressure from above while allowing fluids to flow from 
below. Standing valves generally are run and retrieved on slickline with the valve assembly located in an 
appropriate nipple. Applications for standing valves include testing the tubing string, setting packers, or 
other applications in which it is desirable to maintain fluid in the tubing string.


wellbore orientation
Wellbore direction. Wellbore orientation may be described in terms of inclination and azimuth. Inclination 
refers to the vertical angle measured from the down direction-the down, horizontal and up directions have 
inclinations of 0, 90 and 180, respectively. Azimuth refers to the horizontal angle measured clockwise 
from true north-the north, east, south and west directions have azimuths of 0, 90, 180 and 270, respectively.

"""



### Process text
import random

# randomly sample 400 elements of the data dictionary
print len(d_data)	# 4931
d_data = dict( (k, d_data[k]) for k in random.sample(d_data, 400) )
print len(d_data)	# 400

# form keyword list and description list
keywords, descriptions = [], []
for i in range(len(d_data)):
	keywords.append( d_data.keys()[i] )
	descriptions.append( d_data.values()[i] )



### Create machine learning pipelearn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# pipeline imports
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# define custom transformer to sanity check a pipeline steps input and output
class PipelineDebugger(TransformerMixin):

	def __init__(self, transformer):
		self.transformer = transformer

	def fit(self, X, y=None):
		self.transformer.fit(X, y)
		return self

	def transform(self, X):
		print '='*30, self.transformer.__class__.__name__, '='*30

		#rand_ix = random.randint(0, len(X))
		print '*'*5, 'Before', '*'*5
		print type(X)
		print X[0]
		print type(X[0]), '\n'

		X = self.transformer.transform(X)
		print '*'*5, 'After', '*'*5
		print type(X)
		print X[0]
		print type(X[0]), '\n'*2

		return X

# define custom transformer to apply NLP cleaning of text
class nlp_doc_clean(BaseEstimator, TransformerMixin):
	"""
	Input:	a list of document strings
	Output: a list of cleaned document strings

	"""
	def __init__(self):
		self.stopwords = nltk.corpus.stopwords.words('english')
		self.stemmer = SnowballStemmer('english')

	def fit(self, X, y=None):
		return self

	def doc_clean(self, doc):

		# create list of tokens from the document (a token being a individual component of the vocabulary
		# i.e. a single word, or single punctuation)
		tokens = [word for sent in nltk.sent_tokenize(doc) for word in nltk.word_tokenize(sent)]

		# make all words lower case
		lowers = [ token.lower() for token in tokens ]

		# remove stop words
		stopped = [ lower for lower in lowers if lower not in self.stopwords ]

		# filter out any tokens not containing letters (numeric tokens, raw punctuation etc.)
		filtered_tokens = []
		for token in stopped:
			if re.search('[a-zA-Z]', token):
				filtered_tokens.append(token)

		# reduce tokens to base / stemmed form
		stems = [ self.stemmer.stem(token) for token in filtered_tokens ]

		# then create sentence string as input for the tf-idf vectorizer
		return ' '.join( stems )

	def transform(self, jargon_descs):
		return [ self.doc_clean(desc) for desc in jargon_descs ]

# define custom transformer to apply cosine similarity metric
class cosine_metric(BaseEstimator, TransformerMixin):
	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return 1 - cosine_similarity(X)





### Build pipeline

tfidf_vectorizer = TfidfVectorizer(
	min_df=0.01,	# ignore terms that appear in less than 1% of the documents
	max_df = 0.8,	# ignore terms that appear in more than 80% of the documents
	stop_words='english',
	use_idf=True,
	ngram_range=(1,3)	# model unigrams, bigrams, and trigrams
	)

pipeline = Pipeline([
		('nlp_clean_docs', PipelineDebugger( nlp_doc_clean() )),
		('tf_vectorizer_and_idf', PipelineDebugger( tfidf_vectorizer )),
		('create_cosine_dist_matrix', PipelineDebugger( cosine_metric() ))
#		('create_ward_linkage_matrix', PipelineDebugger(ward()))
	])


dist = pipeline.fit_transform(descriptions)
print dist

quit()



### Use Pipeline
# call fit_transform to use the pipeline
tfidf_matrix = pipeline.fit_transform(descriptions)
print tfidf_matrix.toarray()

# we can also access individual methods of steps in the pipeline using the named_steps function
print pipeline.named_steps

# Note: as we are using the PipelineDebugger class the below uncommented line doesn't work as 
# the named step for tf_vectorizer_and_idf is a PipelineDebugger class object. So we have to 
# call the .transformer init from the PipelineDebugger object to access the tf_vectorizer_and_idf methods
#vocabulary = pipeline.named_steps['tf_vectorizer_and_idf'].get_feature_names()
tfidf_object = pipeline.named_steps['tf_vectorizer_and_idf'].transformer
vocabulary = tfidf_object.get_feature_names()

print 'Size of vocabulary: %i' % len(vocabulary)



### Create distance matrix
dist = 1 - cosine_similarity(tfidf_matrix)


quit()


### Create Ward Linkage matrix
linkage_matrix = ward(dist)



### Run clustering algorithm to understand hidden structure within the keywords / descriptions
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import ward, dendrogram

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=keywords)

plt.tick_params(\
    axis= 'x',			# changes apply to the x-axis
    which='both',		# both major and minor ticks are affected
    bottom='off',		# ticks along the bottom edge are off
    top='off',			# ticks along the top edge are off
    labelbottom='off'
    )

plt.tight_layout() # show plot with tight layout
plt.show()

