'''
For tomorrow:

- changing the CountVectorizer to:
        count_vect = CountVectorizer(min_df=5, stop_words='english', lowercase=True)
   limits the words in the corpus and greatly reduces the dimensionality

- running t-SNE on the lowered dimension document term matrix does show exhibitions of clustering
    (do this in this code / and add it into the others)
    (see tSNE_clustering.py)

- run LDA on whole X data into 4 topics

- include the 4 probabilities in the 100 labels as additional inputs

- re-run through the SVM and see if accuracy is increased.

'''


### Semi-supervised learning.
### Bag of Words + SVM implementation of the 20-newsgroups dataset

"""

Using the ssl_data_maker function we generate:

X_train/y_train size    test_size (constant)    SVM accuracy
2257 (full)             1502                    81.29%
1000                    1502                    76.83%
500                     1502                    70.97%
250                     1502                    70.23%
100                     1502                    57.72%      * we will apply SSL to these initial conditions
50                      1502                    52.23%
25                      1502                    42.47%
5                       1502                    29.02%

X_train/y_train size    test_size (backfilled)  SVM accuracy
2257 (full)             1502                    %
1000                    2759                    %
500                     3259                    %
250                     3509                    %
100                     3659                    %
50                      3709                    %
25                      3734                    %
5                       3754                    %

"""



### Load data
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups

# consider 4 of the available 20 categories
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

# we only want to keep the body of the documents (makes the problem much harder)
remove = ('headers', 'footers', 'quotes')

d_twenty_train = fetch_20newsgroups(remove=remove, subset='train', categories=categories, shuffle=True, random_state=42)
d_twenty_test = fetch_20newsgroups(remove=remove, subset='test', categories=categories, shuffle=True, random_state=42)
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



### Create a semi-supervised learning problem
def ssl_data_maker(df_train, df_test, num_training_instances, fill_test_data=False):

    global X_train, y_train, X_test, y_test

    if fill_test_data:
        X_train = df_train['doc'].values[:num_training_instances]
        y_train = df_train['target'].values[:num_training_instances]

        X_test = df_test['doc'].values
        X_test = np.append( X_test, df_train['doc'].values[num_training_instances:] )

        y_test = df_test['target'].values
        y_test = np.append( y_test, df_train['target'].values[num_training_instances:] )

        return X_train, y_train, X_test, y_test

    else:
        X_train = df_train['doc'].values[:num_training_instances]
        y_train = df_train['target'].values[:num_training_instances]

        X_test = df_test['doc'].values
        y_test = df_test['target'].values

        return X_train, y_train, X_test, y_test

#ssl_data_maker(df_train, df_test, len(df_train))    # creates full, supervised learning problem
#ssl_data_maker(df_train, df_test, 500)              # creates semi-supervised learning problem with constant test set
#ssl_data_maker(df_train, df_test, 500, True)         # creates semi-supervised learning problem with increased test set

ssl_data_maker(df_train, df_test, 250)

print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape



# Note: I could apply stemming to the corpus before calling the CountVectorizer
# to lower the dimensionality of the resulting document term matrix.



### Use scikit-learns CountVectorizer class to create bag of words
# Convert the collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(min_df=5, stop_words='english', lowercase=True)

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



### Fit SVM classifier
from sklearn.linear_model import SGDClassifier

text_clf = SGDClassifier()
text_clf.fit(X_train_tfidf, y_train)



### Test classifier on test set
# go through same steps with test data (note: could build a Pipeline to do this)
# only difference is we call transform instead of fit_transform as we have already fitted our vectorizer
test_counts = count_vect.transform(X_test)
test_tfidf = tfidf_transformer.transform(test_counts)

predicted = text_clf.predict(test_tfidf)
print np.mean(predicted == y_test) # 0.9281



### Manual test
docs_new = [
    'Does God really exist?',
    'Who would win in a fight, Darth Sidious or the Pope?',
    'Deep Learning training should be done on the GPU',
    'Out-of-core learning is where we manage training data outside of system memory constraints']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = text_clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, d_twenty_train['target_names'][category]))

"""
'Does God really exist?' => alt.atheism
'Who would win in a fight, Darth Sidious or the Pope?' => soc.religion.christian
'Deep Learning training should be done on the GPU' => comp.graphics
'Out-of-core learning is where we manage training data outside of system memory constraints' => comp.graphics
"""