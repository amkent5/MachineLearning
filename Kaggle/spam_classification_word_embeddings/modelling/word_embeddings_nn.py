### Kaggle: Spam Classification Data set

'''
Approach:
Use gloVe word embeddings within keras' embedding layer in a neural network model to
classify whether spam or not.

Motivation:
This is SMS data, so arguably it is its own vocabulary (like industry jargon would be).
Word embeddings perform very well.
We will use this model as a benchmark for semi-supervised learning experiments.

To Check:
This classifier does very well on 80% labelled data (97%)
Check that it does worse on 20% labelled data (otherwise it may not be a good candidate
for SSL experiments...)

Resources:
http://www.orbifold.net/default/2017/01/10/embedding-and-tokenizer-in-keras/
https://www.clsp.jhu.edu/~sbergsma/Pubs/bergsmaPhDThesis.pdf

SSL NLP Idea:
5,500 samples of SMS messages. If we de-label 90% of them then we have a semi-supervised learning problem.
The idea is to train the below model on 10% of the labelled data and use that accuracy as a benchmark to
improve by implementing the following semi-supervised methods:

1. Use "Self-learning" (page 27 of Bergsma's PhD thesis)
Use the classifier that has been built on 10% labelled data to predict a large number of unlabelled feature
vectors (say 50% of the unlabelled data). Then re-train the system on both the original 10% labelled examples
and the 'automatically-labelled' examples. and then evaluate the classifier on the same test data set (i.e.
the original 80% unlabelled data). Is it now more accurate?

2. Use 'Bootstrapping'
Self-learning breaks down as the errors in training are compounded on the re-train. Bootstrapping aims to
avoid this by exploiting different views of the problem.

3. Create new features from the unlabelled data to 'boost' our 10%-classifier


'''



### Load data into pandas
import pandas as pd
import numpy as np

datafile = '/Users/admin/Documents/Projects/MachineLearning/Kaggle/spam_classification_word_embeddings/data/spam.csv'
df = pd.read_csv(datafile, names=['label_str', 'text'], skiprows=1, usecols=[0,1], encoding='latin-1')
print df

# convert labels to 0's and 1's
df['labels'] = df['label_str'].apply( lambda x: 1 if x == 'spam' else 0 )



### Preliminary cleaning of vocabulary
import string

# remove non-ascii characters
df['docs'] = df['text'].apply( lambda x: ''.join([i for i in x if i in string.printable]) )

# deal with apostrophes
df['docs'] = df['docs'].apply( lambda x: x.replace("'", "") )

# form numpy arrays
docs = df['docs'].values
labels = df['labels'].values
print docs.shape
print labels.shape



### Define a Keras tokenizer object for our vocabulary
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# train tokenizer on our docs
# the tokenizer essentially maps our vocabulary into convenient dicts accessible through the below attributes
t = Tokenizer()
t.fit_on_texts(docs)

# we can then use the tokenizers attributes:
#	- t.word_index		returns a dict of the indexes each word in our vocabulary is stored at in the tokenizer model
#						i.e. {..., u'happiness': 871, u'elaborating': 6373, u'disturbance': 6233, u'console': 3690, ...}
#	- t.word_counts		returns a dict counting the number of times each word appears in our vocabulary as stored in the tokenizer model
#	- t.document_count	returns the number of text sequences the tokenizer was trained on

vocab_size = len(t.word_index) + 1
print vocab_size	# 8,734



### Use gloVe implementation of word2vec
# we can load in 400k pre-trained 100-dimensional word vectors to use	(https://nlp.stanford.edu/projects/glove/)
glove_data = '/Users/admin/Documents/Projects/MachineLearning/Kaggle/spam_classification_word_embeddings/data/glove.6B.100d.txt'

f = open(glove_data)
embeddings_index = {}
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

# we now have an embedding matrix for our 8,734 words into 75 dimensions
print embedding_matrix.shape	# (8734, 75)



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
print X.shape	# (5572, 20)



### Build and compile the network
from keras.models import Sequential
from keras.layers import Dense, Flatten

# build the model
model = Sequential()
model.add(embedding_layer)
model.add(Flatten())
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
model.fit(X_train, y_train, epochs=10, verbose=0)

# evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print 'Accuracy: %f' % (accuracy*100)	# Accuracy: 97.668161







