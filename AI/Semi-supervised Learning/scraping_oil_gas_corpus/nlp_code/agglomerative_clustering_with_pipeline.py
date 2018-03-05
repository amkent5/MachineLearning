##### Agglomerative clustering of Oil and Gas jargon with scikit-learn pipeline and further cluster analysis

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
from scipy.cluster.hierarchy import ward

# define custom transformer to sanity check a pipeline steps input and output
class pipeline_debugger(TransformerMixin):
	"""
	Input: 	a transformer object
	Output: console info on data flow before and after input transformer
	"""
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
	Output: a list of cleaned token strings

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
	"""
	Input: 	a matrix of vector representations of words/docs (i.e. output of CountVectorizer or TfidfVectorizer classes)
	Output:	a matrix where each element contains all cosine distances between the element and other elements
	"""
	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return 1 - cosine_similarity(X)	# https://en.wikipedia.org/wiki/Cosine_similarity

# define custom transformer to apply agglomerative clustering using the Ward linkage method of Scipy's hierarchy class
class ward_linkage(BaseEstimator, TransformerMixin):
	"""
	Input: 	a pre-computed distance matrix
	Output: a linkage matrix where each row has the format [ix1, ix2, dist, sample_count] where ix1 and ix2 are the
			indices of the clusters the algorithm has decided to merge, dist is the distance between the clusters and 
			sample_count is the number of samples created within the (hierarchical) cluster
	"""
	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return ward(X)	# https://en.wikipedia.org/wiki/Ward%27s_method



### Build pipeline
tfidf_vectorizer = TfidfVectorizer(
	min_df=0.01,	# ignore terms that appear in less than 1% of the documents
	max_df = 0.8,	# ignore terms that appear in more than 80% of the documents
	stop_words='english',
	use_idf=True,
	ngram_range=(1,3)	# model unigrams, bigrams, and trigrams
	)

pipeline = Pipeline([
		('nlp_clean_docs', pipeline_debugger( nlp_doc_clean() )),
		('tf_vectorizer_and_idf', pipeline_debugger( tfidf_vectorizer )),
		('create_cosine_dist_matrix', pipeline_debugger( cosine_metric() )),
		('create_ward_linkage_matrix', pipeline_debugger( ward_linkage() ))
	])



### Use pipeline (call inherited fit_transform method)
linkage_matrix = pipeline.fit_transform(descriptions)

# we can also access individual methods of steps in the pipeline using the named_steps function
print pipeline.named_steps

# Note: as we are using the pipeline_debugger wrapper class we must call the transformer method to access the inner class object
tfidf_object = pipeline.named_steps['tf_vectorizer_and_idf'].transformer
vocabulary = tfidf_object.get_feature_names()
print 'Size of vocabulary: %i' % len(vocabulary)



### Plot results of agglomerative clustering pipeline
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(
	linkage_matrix,
	labels=keywords,
	leaf_font_size=12.,
	leaf_rotation=45.
	)

plt.tight_layout() # show plot with tight layout
plt.ylabel('Ward distance')
plt.show()



### Derive main clusters from results of agglomerative clustering and visualise in 2D

## Determining the number of clusters...
# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

# we have 400 merges in our dendrogram linkage matrix
# linkage_matrix[-1] is the last merge, which should exhibit a high distance value and a sample_count equal to the total number of samples
print linkage_matrix[-1]	# high distance value of 8.52 as expected
							# sample_count param is 400 as expected

# create a truncated dendrogram showing only the last 12 merges
# this will cut away the noisey micro-clusters and enable us to consider macro-clusters
fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(
	linkage_matrix,
	truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  					# last 12 merges
    leaf_font_size=12.,
    show_contracted=True, 	# to get a distribution impression in truncated branches
	)

plt.xlabel('Number of merges in cluster')
plt.ylabel('Ward distance')
plt.show()

# a large jump in distance is typically what we're interested in if we want to argue for
# a certain number of clusters.

# by inspecting the cut down dendrogram we can see that a distance cut-off at around 
# ward distance 6 maximises jumps in distance for each cluster tree
max_dist = 6
fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(
	linkage_matrix,
	truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  					# last 12 merges
    leaf_font_size=12.,
    show_contracted=True, 	# to get a distribution impression in truncated branches
	)

plt.axhline(y=max_dist, color='r', linestyle='--')
plt.xlabel('Number of merges in cluster')
plt.ylabel('Ward distance')
plt.show()

# knowing max_dist (and therefore our number of clusters) we can use the fcluster class to map each observation to a cluster id
from scipy.cluster.hierarchy import fcluster

clusters = fcluster(linkage_matrix, max_dist, criterion='distance')
print clusters

# and use t-SNE (with precomputed distance metric) to reduce the data into 2D so we can visualise the clusters
from sklearn.manifold import TSNE

dim_redn = TSNE(metric="precomputed")

# extract our cosine_similarity distance metric from the pipeline classes
nlp_clean_docs_object = pipeline.named_steps['nlp_clean_docs'].transformer
cleaned_vocab = nlp_clean_docs_object.transform( descriptions )

cosine_dist_object = pipeline.named_steps['create_cosine_dist_matrix'].transformer
distance_metric = cosine_dist_object.transform( tfidf_object.fit_transform( cleaned_vocab ) )

X_reduced = dim_redn.fit_transform( abs(distance_metric) )	# https://github.com/scikit-learn/scikit-learn/issues/5772

print X_reduced[:, 0], X_reduced[:, 1]

plt.scatter( X_reduced[:, 0], X_reduced[:, 1], c=clusters, s=20*2**4 )
plt.show()

# We can see that by analysing the dendrogram we have managed to create well-defined clusters and visualise them for human interaction.


















