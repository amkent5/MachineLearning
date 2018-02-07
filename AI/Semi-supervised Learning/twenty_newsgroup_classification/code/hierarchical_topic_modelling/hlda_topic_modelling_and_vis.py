### Worked quickly with n_samples = 50 and 1000 docs...

# Implementation of hlda with respect to the 20-NG dataset

# Resources:
# https://github.com/joewandy/hlda/blob/master/notebooks/bbc_test.ipynb
# https://github.com/joewandy/hlda

# Question mark over how I get the probability distribution for each LDA hierarchy (I will need
#	these to visulaise the clusters)



### Load data
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups



# we only want to keep the body of the documents (makes the problem much harder)
remove = ('headers', 'footers', 'quotes')
d_twenty_all = fetch_20newsgroups(remove=remove, subset='all', shuffle=True, random_state=42)
df = pd.DataFrame(d_twenty_all['data'], columns=['doc'])
df['target'] = d_twenty_all['target']
print df.shape



### Quick clean of vocab
import string

# remove punctuation, numbers, non-ascii characters and make lowercase
df['doc'] = df['doc'].apply( lambda x: ''.join( [char for char in x if char not in string.punctuation] ) )
df['doc'] =df['doc'].apply( lambda x: ''.join( [char for char in x if char not in ('0','1','2','3','4','5','6','7','8','9')] ) )
df['doc'] = df['doc'].apply( lambda x: ''.join( [char for char in x if char in string.printable] ) )
df['doc'] = df['doc'].apply( lambda x: x.lower() )



### Sample data
pd.options.display.max_colwidth = 100000

print '*** DOCUMENT ***', '\n', df.loc[0]['doc']
print '*** ASSOCIATED LABEL ***', '\n', d_twenty_all['target_names'][df.loc[0]['target']]		# comp.graphics
print '\n'*3
print '*** DOCUMENT ***', '\n', df.loc[1000]['doc']
print '*** ASSOCIATED LABEL ***', '\n', d_twenty_all['target_names'][df.loc[1000]['target']]	# alt.atheism



### Dataset preprocessing
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
vocab = set()
stemmer = PorterStemmer()
stopset = stopwords.words('english') + ['will', 'also', 'said']

# the author of hLDA has not optimised the sampling method so limit the
# number of documents
for doc in df['doc'][:5000]:
	tokens = word_tokenize(doc)
	
	filtered = []
	for word in tokens:
		# stem the words
		word = stemmer.stem(word)

		# remove short tokens
		if len(word) < 3:
			continue

		# remove tokens larger than the largest word
		if len(word) > 30:
			continue

		# remove stop words
		if word in stopset:
			continue

		filtered.append(word)

	# add cleansed words to our vocab
	vocab.update(filtered)

	# add the cleansed document to our corpus
	corpus.append(filtered)



# LDA takes as input a bag of words matrix, so you would use CountVectorizer
# However, hlda takes as input the index of words in the corpus.
# So we sort the vocab, and create an index lookup
vocab = sorted(list(vocab))
word_to_ix = {}
for ix, word in enumerate(vocab):
	word_to_ix[word] = ix

print df['doc'].shape[0]
print len(corpus)
print len(vocab)
print vocab[:100]



### Visualise word frequency with word cloud package
"""
from wordcloud import WordCloud
import matplotlib.pyplot as plt

all_words = []
for doc in corpus:
	for word in doc:
		all_words.append(word)

wordcloud = WordCloud(background_color='white').generate(' '.join(all_words))
plt.figure(figsize=(12, 12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
"""



### Perform hierarchical LDA
from hlda.sampler import HierarchicalLDA

# convert words in corpus into their respective indices
corpus_of_ixs = []

for doc in corpus:

	doc_of_ixs = []
	for word in doc:
		word_ix = word_to_ix[word]
		doc_of_ixs.append(word_ix)

	corpus_of_ixs.append(doc_of_ixs)

# check ix'ing
print corpus[0][0:10]
print corpus_of_ixs[0][0:10]

# create hierarchical LDA object and run the sampler
n_samples = 50       	# no of iterations for the sampler
alpha = 10.0          	# smoothing over level distributions
gamma = 1.0           	# Chinese Restaurant Problem smoothing parameter; number of imaginary customers at next, as yet unused table
eta = 0.1             	# smoothing over topic-word distributions
num_levels = 3        	# the number of levels in the tree
display_topics = 5   	# the number of iterations between printing a brief summary of the topics so far
n_words = 5           	# the number of most probable words to print for each topic after model estimation
with_weights = True		# whether to print the words with the weights

hlda = HierarchicalLDA(corpus_of_ixs, vocab, alpha=alpha, gamma=gamma, eta=eta, num_levels=num_levels)
hlda.estimate(n_samples, display_topics=display_topics, n_words=n_words, with_weights=with_weights)



### Store trained hLDA model for reuse
import cPickle
import gzip

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)
        
def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object

save_zipped_pickle(hlda, 'hLDA_model.p')



