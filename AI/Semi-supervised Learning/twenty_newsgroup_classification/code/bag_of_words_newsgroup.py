### Bag of Words + SVM implementation of the 20-newsgroups dataset

"""

Bag of Words:
We have used word embeddings to encode the meaning of words in high-dimensional
vector spaces for classifiction problems.
The disadvantage of this method is that it generates a large feature space for
large documents (usually 1:~100 expansion ratio for each word).
The bag of words method encodes documents into a vector of length the size of
vocabulary in the corpus (typically around 1000-2000 distinct words).
So this is preferable for large document inputs.

Problem Motivation:
I wanted to build a text classifier using a Scikit-learn model so that I could experiment
with Tamas Madl's semi-supervised learning classes (which are Scikit-learn wrapper functions):
https://github.com/tmadl/semisup-learn

"""



### Load data
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups

# consider 4 of the available 20 categories
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

d_twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
d_twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
df_train, df_test = pd.DataFrame(d_twenty_train['data'], columns=['doc']), pd.DataFrame(d_twenty_test['data'], columns=['doc'])
df_train['target'], df_test['target'] = d_twenty_train['target'], d_twenty_test['target']
print df_train.shape	# (2257, 2)
print df_test.shape		# (1502, 2)



### Quick look at class distribution
import matplotlib.pyplot as plt

class_0 = df_train['target'].value_counts()[0]	# 480
class_1 = df_train['target'].value_counts()[1]	# 584
class_2 = df_train['target'].value_counts()[2]	# 594
class_3 = df_train['target'].value_counts()[3]	# 599
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar([0,1,2,3], [class_0, class_1, class_2, class_3])
#plt.show()



### Quick clean of vocab
import string

# remove apostrophes and non-ascii characters
df_train['doc'], df_test['doc'] = df_train['doc'].apply( lambda x: x.replace("'", "") ), df_test['doc'].apply( lambda x: x.replace("'", "") )
df_train['doc'], df_test['doc'] = df_train['doc'].apply( lambda x: ''.join( [char for char in x if char in string.printable] ) ), df_test['doc'].apply( lambda x: ''.join( [char for char in x if char in string.printable] ) )



### Sample data
pd.options.display.max_colwidth = 100000

print '*** DOCUMENT ***', '\n', df_train.loc[0]['doc']
print '*** ASSOCIATED LABEL ***', '\n', d_twenty_train['target_names'][df_train.loc[0]['target']]		# comp.graphics
print '\n'*3
print '*** DOCUMENT ***', '\n', df_train.loc[1000]['doc']
print '*** ASSOCIATED LABEL ***', '\n', d_twenty_train['target_names'][df_train.loc[1000]['target']]	# alt.atheism

X_train, X_test = df_train['doc'].values, df_test['doc'].values
y_train, y_test = df_train['target'].values, df_test['target'].values



### Use bag-of-words to model this problem
"""
* Simple example of bag of words *

Given two documents:
(1) John likes to watch movies. Mary likes movies too.
(2) John also likes to watch football games.

A list (or corpus) of distinct words is formed (it is orderless):
[
    "John",
    "likes",
    "to",
    "watch",
    "movies",
    "Mary",
    "too",
    "also",
    "football",
    "games"
]

We can then create a feature representation of the number of times the word index
appears in the document. I.e. our new feature representations are:
(1) [1, 2, 1, 1, 2, 1, 1, 0, 0, 0]
(2) [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]

I.e. the input at index 1 is 2 as the word in index 1 in our corpus (likes)
appears twice in the document.

Words like 'a' and 'the' appear a lot in documents, so we often normalise the appearance counts
by the tf-idf numeric (term freq - inverse doc freq).
"""



### Use scikit-learns CountVectorizer class to create bag of words
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

# form term-document matrix
X_train_counts = count_vect.fit_transform(X_train)
print X_train_counts.toarray()
print X_train_counts.toarray().shape
print X_train_counts.toarray()[0].shape



### Downscale / normalise common word appearances using tf-idf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print X_train_tfidf.shape

"""
Example:

for i in range(X_train_counts.toarray().shape[1]): print X_train_counts.toarray()[0][i], '\t', X_train_tfidf.toarray()[0][i]
0   0.0
0   0.0
0   0.0
0   0.0
0   0.0
0   0.0
0   0.0
2   0.256120262391
0   0.0
0   0.0
0   0.0
... ...
"""



### Fit SVM classifier
from sklearn.linear_model import SGDClassifier

text_clf = SGDClassifier()
text_clf.fit(X_train_tfidf, y_train)



### Test classifier on test set
# go through same steps with test data (note could build a Pipeline to do this)
# only difference is we call transform instead of fit_transform as we have
# already fitted our vectorizer
test_counts = count_vect.transform(X_test)
test_tfidf = tfidf_transformer.transform(test_counts)

predicted = text_clf.predict(test_tfidf)
print np.mean(predicted == y_test) # 0.9281



