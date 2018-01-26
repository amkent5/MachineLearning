### Keras word embeddings classifier for tweet sentiment
### Benchmark classifier for SSL experiments


# The two smileys :) and :( have been removed from the tweets in the data set.
# The benchmark task is to classify where a :) has been removed (class 1), and where a :( has been removed (class 0)


# Different test_size params in train_test_split:
# test_size = 0.2 		num_training_rows: 157,337		Accuracy: 75.950172
# test_size = 0.7 		num_training_rows: 59,001		Accuracy: 74.395479
# test_size = 0.9 		num_training_rows: 19,667 		Accuracy: 71.615491
# test_size = 0.99 		num_training_rows: 1,966		Accuracy: 66.239869
# test_size = 0.995 	num_training_rows: 983			Accuracy: 64.838596


# We will assume a human has laboriously and begrudgingly labelled 983 rows of the data set.
# The benchmark classifier achieves an accuracy of 64.8% when applied to the other 195,689 unlabelled
# rows. Can we use ssl techniques to up the accuracy without using more labelled rows...

# This code implements a self-learning approach, where a classifier is built on 1,000 rows of labelled data
# then used to pseudo-label an additional 25% of the unlabelled dataset. The classifier is then re-trained on
# both the original 1,000 of labelled data and the additional pseudo-labelled data and then evaluated once
# more on the original test data set.



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

# build the benchmark architecture
def build_model():
	model = Sequential()
	model.add(embedding_layer)
	model.add(Flatten())

	# adding the following hidden layer boosts accuracy by 3%, and reduces training
	# loss dramatically, but there must be significant overfitting as training accuracy
	# ends up at 95%, try and reduce overfitting...
	model.add(Dense(256, init='uniform', activation='relu'))

	# adding the following dropout layer reduces the training accuracy signficiantly, but
	# increases the validation accuracy by 1.5% (hence is limiting overfitting)
	model.add(Dropout(0.5, name='dropout_1'))

	model.add(Dense(1, activation='sigmoid'))
	model.layers[0].trainable=False

	# compile the model
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

	# summarize the model
	print model.summary()

	return model



### Fit, train and evaluate the benchmark network with ~1000 labels
from sklearn.model_selection import train_test_split

base_X_train, base_X_test, base_y_train, base_y_test = train_test_split(X, labels, test_size=0.995, random_state=42)

# fit and train the benchmark model
base_model = build_model()
base_model.fit(base_X_train, base_y_train, epochs=10)

# evaluate model
loss, accuracy = base_model.evaluate(base_X_test, base_y_test, verbose=0)
print 'Accuracy: %f' % (accuracy*100)



### Use the base_model to pseudo-label an additional 25% of the 'unlabelled' data (i.e. 25% of base_X_test)
# Try to build up to 25% where we only include predictions that the model is sure about (i.e. p<10% & p>90%)
auto_X_train = base_X_train
auto_y_train = base_y_train

proportion = len(base_X_test)/ 4	# 25% of unlabelled
for i in range(len(base_X_test)):

	# append vectors that we use in the 25% of base_X_test onto auto_X_train
	# and for the associated predictions we create, append (the rounded version) onto auto_y_train
	# We will then have training data that includes both the original 10% and 25% of pseudo-labelled data

	vector = base_X_test[i].reshape(1, 20)
	pred = base_model.predict(vector)[0][0]

	if pred < 0.05 or pred > 0.95:
		auto_X_train = np.concatenate((auto_X_train, vector), axis=0)
		auto_y_train = np.concatenate((auto_y_train, [round(pred)]))

		if auto_y_train.shape[0] > proportion:
			break

print len(base_X_train)							# 983
print len(base_X_test)/ 4 						# 48922
print auto_X_train.shape, auto_y_train.shape	# (48923, 20) (48923,)



### Fit and train the same network architecture but include the auto-labelled additional inputs
auto_model = build_model()
auto_model.fit(auto_X_train, auto_y_train, epochs=10)



### Use auto_model on the original, full, unlabelled set and compare accuracy to benchmark
loss, accuracy = auto_model.evaluate(base_X_test, base_y_test)
print 'Accuracy: %f' % (accuracy*100)



# base_model accuracy:	63.340811
# auto_model accuracy:	63.219701

# Consider changing the proportion of unlabelled data and retesting?
# Prune predict results only choosing to use results the model is sure on?
# Bring a human into the loop incrementally to help correct classification?