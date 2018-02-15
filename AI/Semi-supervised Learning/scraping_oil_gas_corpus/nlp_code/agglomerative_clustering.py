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
import numpy as np
import random
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# randomly sample 400 elements of the data dictionary
print len(d_data)	# 4931
d_data = dict( (k, d_data[k]) for k in random.sample(d_data, 400) )
print len(d_data)	# 400

# form keyword list and description list
keywords, descriptions = [], []
for i in range(len(d_data)):
	keywords.append( d_data.keys()[i] )
	descriptions.append( d_data.values()[i] )

# define a tokenizer and stemmer
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

# perform stop word removal, ngram (bi and tri) modelling, tokenisation and
# tf-idf transformation with TfidVectorizer
tfidf_vectorizer = TfidfVectorizer(
	min_df=0.01,	# ignore terms that appear in less than 1% of the documents
	max_df = 0.8,	# ignore terms that appear in more than 80% of the documents
	stop_words='english',
	use_idf=True, 
	tokenizer=tokenize_and_stem, 
	ngram_range=(1,3)	# model unigrams, bigrams, and trigrams
	)

tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)

# inspect
print type(tfidf_matrix)
print tfidf_matrix.shape
print tfidf_matrix.toarray()
for elt in tfidf_matrix.toarray()[0]: print elt

terms = tfidf_vectorizer.get_feature_names()
print 'Model vocabulary:', '\n', terms



### Create distance matrix
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)



### Run clustering algorithm to understand hidden structure within the keywords / descriptions
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import ward, dendrogram

# define the linkage_matrix using ward clustering pre-computed distances
linkage_matrix = ward(dist)

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





