##### Check whether the twitter data set example meets the SSL assumptions



### Load data
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

neg_datafile = '/Users/admin/Documents/Projects/MachineLearning/AI/Semi-supervised Learning/twitter_sentiment_classification/data/train_neg.txt'
pos_datafile = '/Users/admin/Documents/Projects/MachineLearning/AI/Semi-supervised Learning/twitter_sentiment_classification/data/train_pos.txt'

df_neg = pd.read_csv(neg_datafile, names=['tweet_str'], skiprows=0, delimiter='\n', error_bad_lines=False, encoding='latin-1')
df_neg['target'] = 0
print df_neg.shape	# (98954, 2)

df_pos = pd.read_csv(pos_datafile, names=['tweet_str'], skiprows=0, delimiter='\n', error_bad_lines=False, encoding='latin-1')
df_pos['target'] = 1
print df_pos.shape	# (97718, 2)

df = shuffle( pd.concat([df_neg, df_pos]) )
df.reset_index(inplace=True)
print df.shape	# (196672, 2)



### Preliminary cleaning of vocab
import string

# remove apostrophes and non-ascii characters
df['docs'] = df['tweet_str'].apply( lambda x: x.replace("'", "") )
df['docs'] = df['docs'].apply( lambda x: ''.join( [char for char in x if char in string.printable] ) )
print df.shape

# form numpy arrays
docs = df['docs'].values
labels = df['target'].values



### Define a Keras tokenizer object for our vocabulary
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# train tokenizer on our docs
t = Tokenizer()
t.fit_on_texts(docs)

vocab_size = len(t.word_index) + 1
print vocab_size	# 101,630



### Use gloVe implementation of word2vec
# we can load in 400k pre-trained 100-dimensional word vectors to use	(https://nlp.stanford.edu/projects/glove/)
glove_data = '/Users/admin/Documents/Projects/MachineLearning/Kaggle/spam_classification_word_embeddings/data/glove.6B.100d.txt'

embeddings_index = {}
with open(glove_data, 'r') as f:
	for line in f:
		values = line.split()
		word = values[0]
		value = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = value
f.close()
print 'Loaded %s word vectors' % len(embeddings_index)

# now apply these word embeddings to the words in our vocabulary
# create embedding_matrix which has {key: ix of word in our tokenizer object, val: word_vector}
d_word_to_ix = t.word_index
embedding_dimension = 75
embedding_matrix = np.zeros((len(d_word_to_ix) + 1, embedding_dimension))

for word, ix in d_word_to_ix.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros as we initialised embedding_matrix to zeros
		embedding_matrix[ix] = embedding_vector[:embedding_dimension]

# we now have a vector representation matrix for our vocabulary
print embedding_matrix.shape	# (101630, 75)



### Define a Keras Embedding layer and use our embedding_matrix inside it
from keras.layers.embeddings import Embedding

# define our embedding layer for the network
embedding_layer = Embedding(embedding_matrix.shape[0],
	embedding_matrix.shape[1],
	weights=[embedding_matrix],
	input_length=20)

# In order to use the embedding layer we now need to reshape the training data (docs) into word-to-index
# sequences of length 20
# So we utilise our tokenizer construct again...
from keras.preprocessing.sequence import pad_sequences

X = t.texts_to_sequences(docs)
X = pad_sequences(X, maxlen=20)
print X.shape	# (196672, 20)



### Just generate the embedded data for visualisation
from keras.models import Sequential
from keras.layers import Flatten

model = Sequential()
model.add(embedding_layer)
model.add(Flatten())

X = model.predict(X)
print X
print X.shape
print X[0].shape
print X[0][0]

y = labels
print y.shape

'''
### Use PCA
from sklearn.decomposition import PCA


"""
### See how many PCA components we should use (i.e. how many components are required to represent the variance in the data)
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.rc("font", size=14)
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.ylabel('cumulative explained variance');
plt.xlabel('number of components')
plt.show()		# we see we require around 400 (of 1,500) pca components to explain ~75% of the cumulative variance
"""


### Create low-dimensional representation scatter to test SSL assumptions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X_reduced = PCA(n_components=3).fit_transform(X)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()
'''



### Use t-SNE
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


# subset data
X = X[:1000, :]
y = y[:1000]
print X.shape
print y.shape


# 2d
tsne = TSNE(n_components=2, verbose=1, perplexity=40)
tsne_results = tsne.fit_transform(X)
X_tsne = tsne_results[:, 0]
Y_tsne = tsne_results[:, 1]

print X_tsne
print Y_tsne

colors = ['red', 'blue']

plt.scatter(X_tsne, Y_tsne, c=y, cmap=mpl.colors.ListedColormap(colors), s=22**2)
plt.title('t-SNE twitt')
plt.show()


# Spoke to John and the reason I am not seeing clustering is because it doesn't make
# sense to use the Euclidean distance metric with word embeddings. It does make sense to
# use Euclidean distance metric for bag of words (count-frequencies), but try other metrics
# for this word embeddings example:

from sklearn.neighbors import DistanceMetric

#dist = DistanceMetric.get_metric('chebyshev')

tsne2 = TSNE(n_components=2, verbose=1, perplexity=20, metric='cosine')
tsne_results = tsne2.fit_transform(X)
X_tsne = tsne_results[:, 0]
Y_tsne = tsne_results[:, 1]

plt.scatter(X_tsne, Y_tsne, c=y, cmap=mpl.colors.ListedColormap(colors), s=22**2)
plt.title('t-SNE hamming')
plt.show()


















