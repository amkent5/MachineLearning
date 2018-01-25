# apply this:
# https://stackoverflow.com/questions/44909134/how-to-avoid-overfitting-on-a-simple-feed-forward-network

### Keras word embeddings classifier for tweet sentiment
### Benchmark classifier for SSL experiments
"""
The two smileys :) and :( have been removed from the tweets in the data set.
The task is to classify where a :) has been removed (class 1), and where a :(
has been removed (class -1)

"""



### Load data
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# there are 'full' datasets in the same location
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



### Look at class distribution
import matplotlib.pyplot as plt

num_negs = df['target'].value_counts()[0]	# 50.3% representation
num_pos = df['target'].value_counts()[1]	# 49.7% representation
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar([1,2], [num_negs, num_pos])
#plt.show()



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



### Build and compile the network
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

# build the model
model = Sequential()
model.add(embedding_layer)
model.add(Flatten())

# adding the following hidden layer boosts accuracy by 3%, and reduces training
# loss dramatically, but there must be significant overfitting as training accuracy
# ends up at 95%, try and reduce overfitting...
model.add(Dense(256, init='uniform', activation='relu'))

# adding the following dropout layer reduces the training accuracy signficiantly, but
# increases the validation accuracy by 1.5%
model.add(Dropout(0.5, name='dropout_1'))

model.add(Dense(1, activation='sigmoid'))
model.layers[0].trainable=False

# compile the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

# summarize the model
print model.summary()



### Fit, train and evaluate the network
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# fit model
model.fit(X_train, y_train, epochs=10)

# evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print 'Accuracy: %f' % (accuracy*100)	# Accuracy: 75.495106






