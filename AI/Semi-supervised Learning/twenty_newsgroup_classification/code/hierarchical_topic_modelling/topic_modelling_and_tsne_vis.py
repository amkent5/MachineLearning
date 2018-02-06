### Resources
# https://www.kaggle.com/ykhorramz/lda-and-t-sne-interactive-visualization
# https://shuaiw.github.io/2016/12/22/topic-modeling-and-tsne-visualzation.html
# https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
# https://towardsdatascience.com/improving-the-interpretation-of-topic-models-87fd2ee3847d



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

# remove apostrophes, non-ascii characters and make lowercase
df['doc'] = df['doc'].apply( lambda x: x.replace("'", "") )
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
# LDA takes as input a bag of words matrix, so use CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

no_features = 1000
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')	# perform some tokenisation in the CountVec call

tf = tf_vectorizer.fit_transform(df['doc'].values)
tf_feature_names = tf_vectorizer.get_feature_names()
print tf 	# we now have the term frequency matrix (i.e. the bag of words)



### Build LDA model
import pickle
import os
from sklearn.decomposition import LatentDirichletAllocation

# save LDA model to disk to speed up experimentation
filename = 'topic_modelling_and_tsne_vis__LDA_model.sav'
if not os.path.isfile(filename):

	n_topics = 20 # number of topics
	max_iter = 50 # number of iterations

	lda_model = LatentDirichletAllocation(n_topics=n_topics, max_iter=max_iter, n_jobs=-1, verbose=10)
	lda_model_fit = lda_model.fit(tf)

	pickle.dump(lda_model_fit, open(filename, 'wb'))

else:

	lda_model_fit = pickle.load(open(filename, 'rb'))


X_topics = lda_model_fit.transform(tf)
print X_topics




### Display the most frequent words that make up each topic in our fit
def display_topics(model, feature_names, no_top_words):
	for topic_idx, topic in enumerate(model.components_):
		print "Topic %d:" % (topic_idx)
		print " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])

display_topics(lda_model_fit, tf_feature_names, 10)



### Visualise our topic distribution with t-SNE
# We have a learned LDA model. But we cannot visually inspect how good our model is.
# t-SNE comes to the rescue
from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')

# 20-D -> 2-D
tsne_lda = tsne_model.fit_transform(X_topics)
x_tsne = tsne_lda[:, 0]
y_tsne = tsne_lda[:, 1]

# create 20 colours
from random import randint
colours = []
for i in range(20): colours.append('#%06X' % randint(0, 0xFFFFFF))

# for each of the rows in X_topics, there are 20 probabilities representing each of the 20 classes
# form the topic grouping by taking the index of the max probability for each row
_lda_keys = []
for i in range(X_topics.shape[0]): _lda_keys.append( X_topics[i].argmax() )

# plot
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

plt.scatter(X_tsne, Y_tsne, c=_lda_keys, cmap=mpl.colors.ListedColormap(colours), s=22**2)
plt.title('20-NG LDA t-SNE')
plt.show()




### Visualise our topic distriubtion with pyLDAvis















